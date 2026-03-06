"""Re-export SCRFD 10G with dynamic batch support at a fixed spatial size.

This script is a modified version of insightface's scrfd2onnx.py that exports
the SCRFD detection model with ONLY the batch dimension dynamic (spatial dims
are fixed). This prevents cross-frame contamination caused by reshape operations
that break when both batch and spatial dims are symbolic.

Environment setup (use a SEPARATE venv, NOT the project's .venv):
    NOTE: No CUDA/GPU needed — export is CPU-only graph tracing.

    python3.9 -m venv /tmp/scrfd-export
    source /tmp/scrfd-export/bin/activate
    pip install torch==1.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
    pip install Cython==0.29.33 numpy==1.23.1 onnx onnx-simplifier==0.3.6 opencv-python

    # mmcv-full (compiles C++ extensions, takes a few minutes)
    pip install mmcv-full==1.3.3

    # IMPORTANT: Do NOT install mmdet separately. The insightface scrfd directory
    # bundles its own mmdet with the SCRFD model registered in the registry.
    git clone --depth 1 https://github.com/deepinsight/insightface.git /tmp/insightface
    cd /tmp/insightface/detection/scrfd
    pip install -v -e .

    # Known issue (github.com/deepinsight/insightface/issues/1573):
    # If you get "from mmdet.core import ..." errors, the import path may need
    # adjustment. This script handles both import paths automatically.

Download SCRFD_10G_KPS PyTorch weights from:
    https://1drv.ms/u/s!AswpsDO2toNKqycsF19UbaCWaLWx?e=F6i5Vm

Usage:
    python scripts/export_scrfd_batched.py \\
        --checkpoint /path/to/scrfd_10g_kps.pth \\
        --config /tmp/insightface/detection/scrfd/configs/scrfd/scrfd_10g_bnkps.py \\
        --shape 640 \\
        --output scrfd_10g_640_batch.onnx
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys


def check_dependencies() -> bool:
    """Check that all required packages are available."""
    missing = []
    for pkg in ["torch", "onnx", "numpy", "cv2"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    # Check mmdet with SCRFD registration
    try:
        import mmdet  # noqa: F401
    except ImportError:
        missing.append("mmdet (install via insightface/detection/scrfd)")

    try:
        from onnxsim import simplify  # noqa: F401
    except ImportError:
        missing.append("onnx-simplifier")

    if missing:
        print(f"ERROR: Missing packages: {', '.join(missing)}")
        print("See the docstring at the top of this script for setup instructions.")
        return False
    return True


def get_mmdet_imports():
    """Import mmdet functions, handling different import paths."""
    try:
        from mmdet.core import (
            build_model_from_cfg,
            generate_inputs_and_wrap_model,
            preprocess_example_input,
        )
        return build_model_from_cfg, generate_inputs_and_wrap_model, preprocess_example_input
    except ImportError:
        pass

    try:
        from mmdet.core.export import (
            build_model_from_cfg,
            generate_inputs_and_wrap_model,
            preprocess_example_input,
        )
        return build_model_from_cfg, generate_inputs_and_wrap_model, preprocess_example_input
    except ImportError:
        pass

    print("ERROR: Cannot import mmdet export utilities.")
    print("Make sure you installed the bundled mmdet from insightface/detection/scrfd/")
    print("See: github.com/deepinsight/insightface/issues/1573")
    sys.exit(1)


def export_scrfd_batched(
    config_path: str,
    checkpoint_path: str,
    input_shape: tuple[int, ...],
    output_file: str,
    input_img: str | None = None,
    opset_version: int = 11,
) -> None:
    """Export SCRFD model to ONNX with batch-only dynamic axes."""
    import onnx
    import torch

    _, generate_inputs_and_wrap_model, _ = get_mmdet_imports()

    mean = [127.5, 127.5, 127.5]
    std = [128.0, 128.0, 128.0]
    normalize_cfg = {"mean": mean, "std": std}

    input_config = {
        "input_shape": input_shape,
        "input_path": input_img or "",
        "normalize_cfg": normalize_cfg,
    }

    # Load checkpoint and strip optimizer state
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    tmp_ckpt_file = None
    if "optimizer" in checkpoint:
        del checkpoint["optimizer"]
        tmp_ckpt_file = checkpoint_path + "_slim.pth"
        torch.save(checkpoint, tmp_ckpt_file)
        print(f"Stripped optimizer state, saved to {tmp_ckpt_file}")
        checkpoint_path = tmp_ckpt_file

    model, tensor_data = generate_inputs_and_wrap_model(
        config_path, checkpoint_path, input_config
    )

    if tmp_ckpt_file is not None:
        os.remove(tmp_ckpt_file)

    # Define input/output names
    input_names = ["input.1"]
    output_names = [
        "score_8", "score_16", "score_32",
        "bbox_8", "bbox_16", "bbox_32",
    ]

    # Check if model has keypoints
    if "stride_kps" in str(model):
        output_names += ["kps_8", "kps_16", "kps_32"]
        print(f"Model has keypoints, {len(output_names)} outputs")
    else:
        print(f"Model without keypoints, {len(output_names)} outputs")

    # KEY DIFFERENCE from original scrfd2onnx.py:
    # Only batch dimension (dim 0) is dynamic. Spatial dimensions are FIXED.
    # This prevents cross-frame contamination from reshape operations.
    dynamic_axes = {input_names[0]: {0: "batch"}}
    for name in output_names:
        dynamic_axes[name] = {0: "batch"}

    print(f"Input shape: {input_shape}")
    print(f"Dynamic axes: batch only (spatial fixed at {input_shape[2]}x{input_shape[3]})")

    torch.onnx.export(
        model,
        tensor_data,
        output_file,
        keep_initializers_as_inputs=False,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
    )

    print(f"Raw ONNX exported to {output_file}")
    print("NOTE: onnxsim skipped — it collapses dynamic batch internally")

    # Verify the exported model
    print(f"\nExported: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")

    loaded = onnx.load(output_file)
    for inp in loaded.graph.input:
        dims = [
            d.dim_param if d.dim_param else str(d.dim_value)
            for d in inp.type.tensor_type.shape.dim
        ]
        print(f"  Input '{inp.name}': [{', '.join(dims)}]")
    for out in loaded.graph.output:
        dims = [
            d.dim_param if d.dim_param else str(d.dim_value)
            for d in out.type.tensor_type.shape.dim
        ]
        print(f"  Output '{out.name}': [{', '.join(dims)}]")

    print("\nDone! Validate with: python scripts/compare_det_models.py")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-export SCRFD with dynamic batch support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to SCRFD PyTorch checkpoint (.pth)",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to mmdet config (e.g. configs/scrfd/scrfd_10g_bnkps.py)",
    )
    parser.add_argument(
        "--shape", type=int, default=640,
        help="Input spatial size (default: 640 for 640x640)",
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="Output ONNX path (default: scrfd_10g_{shape}_batch.onnx)",
    )
    parser.add_argument(
        "--input-img", type=str, default=None,
        help="Optional test image for tracing",
    )
    parser.add_argument(
        "--opset-version", type=int, default=11,
        help="ONNX opset version (default: 11)",
    )
    args = parser.parse_args()

    if not check_dependencies():
        sys.exit(1)

    if not osp.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if not osp.exists(args.config):
        print(f"ERROR: Config not found: {args.config}")
        sys.exit(1)

    input_shape = (1, 3, args.shape, args.shape)
    output_file = args.output or f"scrfd_10g_{args.shape}_batch.onnx"

    export_scrfd_batched(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        input_shape=input_shape,
        output_file=output_file,
        input_img=args.input_img,
        opset_version=args.opset_version,
    )


if __name__ == "__main__":
    main()
