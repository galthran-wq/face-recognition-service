import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from src.services.face_provider.scrfd_export import (
    convert_scrfd_to_dynamic_batch,
    validate_dynamic_batch,
)

_REAL_DET_MODEL = Path("~/.insightface/models/buffalo_l/det_10g.onnx").expanduser()


def _make_scrfd_like_model(path: str) -> None:
    """Minimal graph with the stock SCRFD head tail: input [1, 3, ?, ?], each
    output produced by Transpose(perm=(2,3,0,1)) -> Reshape([-1, C]) — the
    batch dim parked third, exactly like det_10g."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    inp = helper.make_tensor_value_info("input.1", TensorProto.FLOAT, [1, 3, "?", "?"])
    shape_init = numpy_helper.from_array(np.array([-1, 3], dtype=np.int64), name="flat_shape")
    transpose = helper.make_node("Transpose", ["input.1"], ["hwnc"], perm=[2, 3, 0, 1])
    reshape = helper.make_node("Reshape", ["hwnc", "flat_shape"], ["scores"], name="Reshape_0")
    out = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [192, 3])
    graph = helper.make_graph([transpose, reshape], "scrfd_like", [inp], [out], initializer=[shape_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


class TestConvertScrfdToDynamicBatch:
    def test_converts_and_batches_fold_correctly(self, tmp_path: Path) -> None:
        import onnxruntime as ort

        model_path = str(tmp_path / "det_like.onnx")
        _make_scrfd_like_model(model_path)

        outcome = convert_scrfd_to_dynamic_batch(model_path, det_size=(8, 8))
        assert outcome == "converted"
        assert os.path.exists(model_path + ".bak")

        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        shape = sess.get_inputs()[0].shape
        assert shape[0] == "batch"
        assert shape[1:] == [3, 8, 8]

        rng = np.random.default_rng(1)
        blob = rng.standard_normal((2, 3, 8, 8), dtype=np.float32)
        (batched,) = sess.run(None, {"input.1": blob})
        # Flat 2-D output with contiguous per-image row blocks, not interleaved.
        assert batched.shape == (128, 3)
        (single,) = sess.run(None, {"input.1": blob[1:2]})
        np.testing.assert_allclose(batched[64:], single)

        # Batch-1 rows must be identical to the original graph's output.
        orig = ort.InferenceSession(model_path + ".bak", providers=["CPUExecutionProvider"])
        (orig_out,) = orig.run(None, {"input.1": blob[0:1]})
        np.testing.assert_allclose(batched[:64], orig_out)

    def test_idempotent(self, tmp_path: Path) -> None:
        model_path = str(tmp_path / "det_like.onnx")
        _make_scrfd_like_model(model_path)

        assert convert_scrfd_to_dynamic_batch(model_path, det_size=(8, 8)) == "converted"
        assert convert_scrfd_to_dynamic_batch(model_path, det_size=(8, 8)) == "already_dynamic"

    def test_reconverts_from_backup_on_det_size_change(self, tmp_path: Path) -> None:
        import onnxruntime as ort

        model_path = str(tmp_path / "det_like.onnx")
        _make_scrfd_like_model(model_path)

        assert convert_scrfd_to_dynamic_batch(model_path, det_size=(8, 8)) == "converted"
        assert convert_scrfd_to_dynamic_batch(model_path, det_size=(16, 16)) == "converted"

        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        assert sess.get_inputs()[0].shape[1:] == [3, 16, 16]

    def test_unsupported_graph_left_untouched(self, tmp_path: Path) -> None:
        import onnx
        from onnx import TensorProto, helper

        # Output comes straight from a Relu — no flattening Reshape to fix up.
        inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, "?", "?"])
        out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, "?", "?"])
        relu = helper.make_node("Relu", ["x"], ["y"])
        graph = helper.make_graph([relu], "not_scrfd", [inp], [out])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
        model_path = str(tmp_path / "other.onnx")
        onnx.save(model, model_path)
        before = Path(model_path).read_bytes()

        assert convert_scrfd_to_dynamic_batch(model_path, det_size=(8, 8)) == "unsupported"
        assert Path(model_path).read_bytes() == before
        assert not os.path.exists(model_path + ".bak")


@pytest.mark.skipif(not _REAL_DET_MODEL.exists(), reason="det_10g.onnx not downloaded")
class TestRealDetectorConversion:
    def test_convert_and_validate_det_10g(self, tmp_path: Path) -> None:
        # The installed model may already be converted (load_model does it in
        # place, keeping the original as .bak) — always start from a batch-1 graph.
        source = _REAL_DET_MODEL.with_suffix(".onnx.bak")
        if not source.exists():
            source = _REAL_DET_MODEL
        model_path = str(tmp_path / "det_10g.onnx")
        shutil.copy2(source, model_path)
        if convert_scrfd_to_dynamic_batch(model_path, det_size=(640, 640)) == "already_dynamic":
            pytest.skip("no batch-1 det_10g graph available to convert")

        # Batched outputs must match the original graph run image-by-image.
        validate_dynamic_batch(model_path + ".bak", model_path, det_size=(640, 640), batch=2)
