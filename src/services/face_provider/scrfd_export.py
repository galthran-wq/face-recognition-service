"""Re-export a SCRFD detection graph with a dynamic batch dimension.

Stock insightface SCRFD models (``det_10g.onnx``, ...) are exported with batch
fixed at 1 and dynamic spatial dims (``[1, 3, ?, ?]``), so every image costs one
``session.run``. This module rewrites the graph in place to ``[N, 3, H, W]`` —
dynamic batch, static spatial dims (the service letterboxes every image to
``det_size`` anyway). The head outputs stay flat 2-D ``[N*anchors, C]``, but
the head-tail transposes are fixed so images fold into contiguous row blocks —
batch 1 stays bit-identical to the stock graph (insightface's own single-image
path keeps working), and a batched caller just slices rows per image.

Used two ways:

- automatically at startup by ``InsightFaceProvider.load_model()`` when the
  ``face_det_dynamic_batch`` setting is on (the default);
- as a CLI::

      uv run python -m src.services.face_provider.scrfd_export \\
          ~/.insightface/models/buffalo_l/det_10g.onnx --validate

The original graph is kept next to the model as ``<name>.onnx.bak`` so the
conversion can be redone for a different ``det_size`` or rolled back.
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from onnx import ModelProto

ConvertOutcome = Literal["converted", "already_dynamic", "unsupported"]

_BATCH_DIM_PARAM = "batch"


def _load_model(model_path: str) -> ModelProto:
    import onnx  # noqa: PLC0415

    return onnx.load(model_path)


def _input_dims(model: ModelProto) -> list[Any] | None:
    """Return the dims of the sole graph input, or None if the layout is unexpected."""
    graph = model.graph
    if len(graph.input) != 1:
        return None
    dims = graph.input[0].type.tensor_type.shape.dim
    if len(dims) != 4:
        return None
    return list(dims)


def _is_dynamic_batch(model: ModelProto, det_size: tuple[int, int]) -> bool:
    """True if the graph already has a dynamic batch and spatial dims == det_size."""
    dims = _input_dims(model)
    if dims is None:
        return False
    batch, channels, height, width = dims
    return bool(
        batch.dim_value == 0
        and channels.dim_value == 3
        and height.dim_value == det_size[1]
        and width.dim_value == det_size[0]
    )


def _rewrite_graph(model: ModelProto, det_size: tuple[int, int]) -> bool:
    """Apply the dynamic-batch surgery in place. Returns False if the graph
    does not look like a stock SCRFD export (in which case it is untouched)."""
    import onnx  # noqa: PLC0415
    from onnx import numpy_helper  # noqa: PLC0415

    graph = model.graph
    dims = _input_dims(model)
    if dims is None:
        return False
    batch, channels = dims[0], dims[1]
    if batch.dim_value != 1 or channels.dim_value != 3:
        return False

    # Every graph output must trace back (through shape-preserving elementwise
    # ops — the score heads end in Sigmoid) to the stock SCRFD head tail:
    #   Transpose(perm=(2,3,0,1)) -> Reshape([-1, C])
    # For batch 1 that yields [anchors, C] rows ordered (h, w, anchor), but the
    # transpose parks the batch dim third, so with batch N the flatten would
    # interleave images per spatial cell. Fixing each perm to (0, 2, 3, 1)
    # makes the same Reshape fold images into contiguous [K, C] row blocks —
    # batch 1 output stays bit-identical, so insightface's own single-image
    # detect keeps working on the converted graph.
    producers = {output: node for node in graph.node for output in node.output}
    initializers = {init.name: init for init in graph.initializer}
    elementwise_ops = {"Sigmoid", "Identity", "Relu"}

    planned: list[Any] = []  # transposes whose perm needs the fix
    for out in graph.output:
        node = producers.get(out.name)
        while node is not None and node.op_type in elementwise_ops and len(node.input) == 1:
            node = producers.get(node.input[0])
        if node is None or node.op_type != "Reshape" or len(node.input) < 2:
            return False
        shape_init = initializers.get(node.input[1])
        if shape_init is None:
            return False
        shape = numpy_helper.to_array(shape_init)
        if shape.ndim != 1 or shape.shape[0] != 2 or shape[0] != -1:
            return False

        transpose = producers.get(node.input[0])
        if transpose is None or transpose.op_type != "Transpose" or len(transpose.attribute) != 1:
            return False
        perm = list(transpose.attribute[0].ints)
        if perm == [2, 3, 0, 1]:
            planned.append(transpose)
        elif len(perm) != 4 or perm[0] != 0:
            return False  # batch already leads for perm[0] == 0 — nothing to fix

    # Input: [1, 3, ?, ?] -> [batch, 3, H, W]
    in_dims = graph.input[0].type.tensor_type.shape.dim
    in_dims[0].ClearField("dim_value")
    in_dims[0].dim_param = _BATCH_DIM_PARAM
    in_dims[2].ClearField("dim_param")
    in_dims[2].dim_value = det_size[1]
    in_dims[3].ClearField("dim_param")
    in_dims[3].dim_value = det_size[0]

    for transpose in planned:
        del transpose.attribute[0].ints[:]
        transpose.attribute[0].ints.extend([0, 2, 3, 1])

    # Output value_info: [K, C] -> [N*anchors, C]. The row count now scales
    # with the batch, so it must be symbolic (the stock static K was only
    # valid for one input size anyway).
    for i, out in enumerate(graph.output):
        out_dims = out.type.tensor_type.shape.dim
        if len(out_dims) != 2:
            return False
        chans = out_dims[1].dim_value
        out.type.tensor_type.shape.ClearField("dim")
        new_dims = out.type.tensor_type.shape.dim
        new_dims.add().dim_param = f"batch_anchors_{i}"
        new_dims.add().dim_value = chans

    # Drop stale intermediate shape annotations: the stock export baked
    # batch=1 into value_info, and the CUDA EP's memory-pattern planner trusts
    # them — batch N then fails with "Shape mismatch attempting to re-use
    # buffer {1,...}". ORT re-infers shapes at session load.
    graph.ClearField("value_info")

    onnx.checker.check_model(model)
    return True


def convert_scrfd_to_dynamic_batch(model_path: str, det_size: tuple[int, int] = (640, 640)) -> ConvertOutcome:
    """Convert a SCRFD ONNX file to dynamic batch in place (atomic, with backup).

    Idempotent and safe to run concurrently from multiple processes sharing a
    model dir: writers produce identical bytes and swap via ``os.replace``.
    Whenever a ``.bak`` of the original graph exists, the conversion is redone
    from it — so a model converted by an older/buggier version of this module
    (or for a different ``det_size``) self-heals on the next startup; the
    rewrite is skipped only when the resulting bytes already match the file.
    """
    backup_path = model_path + ".bak"
    have_backup = os.path.exists(backup_path)
    model = _load_model(backup_path if have_backup else model_path)

    dims = _input_dims(model)
    if dims is not None and dims[0].dim_value == 0:
        # No backup and the file itself is already dynamic: nothing to redo
        # from, and re-running the rewrite needs the batch-1 original.
        return "already_dynamic" if _is_dynamic_batch(model, det_size) else "unsupported"

    if not _rewrite_graph(model, det_size):
        return "unsupported"

    serialized = model.SerializeToString()
    with open(model_path, "rb") as f:
        if f.read() == serialized:
            return "already_dynamic"

    if not have_backup:
        shutil.copy2(model_path, backup_path)
    tmp_path = f"{model_path}.tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        f.write(serialized)
    os.replace(tmp_path, model_path)
    return "converted"


def validate_dynamic_batch(
    original_path: str,
    converted_path: str,
    det_size: tuple[int, int] = (640, 640),
    batch: int = 3,
    atol: float = 1e-4,
) -> None:
    """Check the converted graph against the original on random inputs.

    Runs the original model image-by-image and the converted model as one
    batch (rows ``[b*K:(b+1)*K]`` of each flat output belong to image ``b``);
    raises ``AssertionError`` on any mismatch beyond ``atol``.
    """
    import numpy as np  # noqa: PLC0415
    import onnxruntime as ort  # type: ignore[import-untyped]  # noqa: PLC0415

    rng = np.random.default_rng(0)
    blob = rng.standard_normal((batch, 3, det_size[1], det_size[0]), dtype=np.float32)

    orig = ort.InferenceSession(original_path, providers=["CPUExecutionProvider"])
    conv = ort.InferenceSession(converted_path, providers=["CPUExecutionProvider"])
    input_name = orig.get_inputs()[0].name
    output_names = [o.name for o in orig.get_outputs()]

    batched_outs = conv.run(output_names, {input_name: blob})
    for b in range(batch):
        single_outs = orig.run(output_names, {input_name: blob[b : b + 1]})
        for name, single, batched in zip(output_names, single_outs, batched_outs, strict=True):
            rows = single.shape[0]
            image_slice = batched[b * rows : (b + 1) * rows]
            if not np.allclose(single, image_slice, atol=atol):
                max_diff = float(np.max(np.abs(single - image_slice)))
                msg = f"output {name} mismatch at image {b}: max diff {max_diff}"
                raise AssertionError(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-export a SCRFD ONNX graph with a dynamic batch dimension")
    parser.add_argument("model", help="Path to the SCRFD .onnx file (converted in place, original kept as .bak)")
    parser.add_argument("--det-size", default="640,640", help="Static spatial size W,H to bake in (default: 640,640)")
    parser.add_argument("--validate", action="store_true", help="Compare converted outputs against the original")
    args = parser.parse_args()

    width, height = (int(part.strip()) for part in args.det_size.split(","))
    outcome = convert_scrfd_to_dynamic_batch(args.model, det_size=(width, height))
    print(f"{args.model}: {outcome}")
    if outcome == "unsupported":
        raise SystemExit(1)
    if args.validate:
        validate_dynamic_batch(args.model + ".bak", args.model, det_size=(width, height))
        print("validation passed: batched outputs match per-image originals")


if __name__ == "__main__":
    main()
