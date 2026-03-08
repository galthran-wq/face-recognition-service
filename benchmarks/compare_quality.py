#!/usr/bin/env python3
"""Compare embedding quality between CUDA EP (FP32) and TensorRT EP (FP16).

Loads the same model twice — once with CUDA EP, once with TRT EP — and compares
the embeddings produced for every detected face. Reports cosine similarity,
max absolute difference, and L2 distance.

Usage (inside GPU container):
    python3 benchmarks/compare_quality.py --model-dir /models
    python3 benchmarks/compare_quality.py --image path/to/photo.jpg
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def load_model(use_tensorrt: bool, model_dir: str, det_size: tuple[int, int], trt_cache: str):
    import onnxruntime as ort
    from insightface.app import FaceAnalysis
    from insightface.model_zoo.model_zoo import PickableInferenceSession

    _original_init = PickableInferenceSession.__init__

    def _patched_init(self_sess, model_path, **kwargs):
        if "sess_options" not in kwargs:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.enable_mem_pattern = True
            so.enable_mem_reuse = True
            kwargs["sess_options"] = so
        _original_init(self_sess, model_path, **kwargs)

    PickableInferenceSession.__init__ = _patched_init

    fa_kwargs: dict = {}
    if use_tensorrt:
        os.makedirs(trt_cache, exist_ok=True)
        fa_kwargs["providers"] = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        fa_kwargs["provider_options"] = [
            {
                "device_id": "0",
                "trt_fp16_enable": "True",
                "trt_engine_cache_enable": "True",
                "trt_engine_cache_path": trt_cache,
                "trt_max_workspace_size": str(2 * 1024**3),
            },
            {"device_id": "0"},
            {},
        ]
    else:
        fa_kwargs["providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        fa_kwargs["provider_options"] = [{"device_id": "0"}, {}]

    app = FaceAnalysis(name="buffalo_l", root=model_dir, **fa_kwargs)
    app.prepare(ctx_id=0, det_size=det_size)

    PickableInferenceSession.__init__ = _original_init
    return app


def load_image(image_path: str | None) -> np.ndarray:
    if image_path and os.path.isfile(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            return img

    try:
        import insightface
        data_dir = Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"
        img = cv2.imread(str(data_dir))
        if img is not None:
            print(f"Using bundled test image: {data_dir}")
            return img
    except Exception:
        pass

    print("ERROR: No test image found.", file=sys.stderr)
    sys.exit(1)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    parser = argparse.ArgumentParser(description="Compare CUDA EP vs TRT EP embedding quality")
    parser.add_argument("--model-dir", default=os.path.expanduser("~/.insightface"))
    parser.add_argument("--image", default=None)
    parser.add_argument("--det-size", default="640,640")
    parser.add_argument("--trt-cache", default="/tmp/trt_cache_quality_test")
    args = parser.parse_args()

    det_size = tuple(int(x) for x in args.det_size.split(","))

    img = load_image(args.image)
    print(f"Image: {img.shape[1]}x{img.shape[0]}")

    # Load CUDA EP model (FP32 baseline)
    print("\n--- Loading CUDA EP model (FP32) ---")
    app_cuda = load_model(use_tensorrt=False, model_dir=args.model_dir, det_size=det_size, trt_cache=args.trt_cache)

    # Run inference
    print("Running CUDA EP inference...")
    faces_cuda = app_cuda.get(img)
    print(f"  Detected {len(faces_cuda)} faces")

    # Extract embeddings and bboxes
    cuda_embeddings = [f.normed_embedding for f in faces_cuda]
    cuda_bboxes = [f.bbox for f in faces_cuda]

    # Also get age/gender for comparison
    cuda_ages = [getattr(f, "age", None) for f in faces_cuda]
    cuda_genders = [getattr(f, "gender", None) for f in faces_cuda]

    # Free CUDA EP model to release GPU memory
    del app_cuda

    # Load TRT EP model (FP16)
    print("\n--- Loading TensorRT EP model (FP16) ---")
    app_trt = load_model(use_tensorrt=True, model_dir=args.model_dir, det_size=det_size, trt_cache=args.trt_cache)

    # Run inference
    print("Running TRT EP inference...")
    faces_trt = app_trt.get(img)
    print(f"  Detected {len(faces_trt)} faces")

    trt_embeddings = [f.normed_embedding for f in faces_trt]
    trt_ages = [getattr(f, "age", None) for f in faces_trt]
    trt_genders = [getattr(f, "gender", None) for f in faces_trt]

    # Match faces by bbox IoU
    if len(faces_cuda) != len(faces_trt):
        print(f"\nWARNING: Different face counts! CUDA={len(faces_cuda)}, TRT={len(faces_trt)}")
        n = min(len(faces_cuda), len(faces_trt))
    else:
        n = len(faces_cuda)

    if n == 0:
        print("No faces to compare.")
        return

    # Compare embeddings
    print(f"\n{'='*60}")
    print(f"EMBEDDING QUALITY COMPARISON ({n} faces)")
    print(f"{'='*60}")

    cosines = []
    max_abs_diffs = []
    l2_dists = []

    for i in range(n):
        e_cuda = cuda_embeddings[i]
        e_trt = trt_embeddings[i]

        cos = cosine_sim(e_cuda, e_trt)
        max_diff = float(np.max(np.abs(e_cuda - e_trt)))
        l2 = float(np.linalg.norm(e_cuda - e_trt))

        cosines.append(cos)
        max_abs_diffs.append(max_diff)
        l2_dists.append(l2)

        print(f"\nFace {i}:")
        print(f"  Cosine similarity:    {cos:.6f}")
        print(f"  Max absolute diff:    {max_diff:.6f}")
        print(f"  L2 distance:          {l2:.6f}")
        print(f"  Norm CUDA: {np.linalg.norm(e_cuda):.6f}  TRT: {np.linalg.norm(e_trt):.6f}")

        # Age/gender comparison
        if cuda_ages[i] is not None and trt_ages[i] is not None:
            age_diff = abs(cuda_ages[i] - trt_ages[i])
            gender_match = cuda_genders[i] == trt_genders[i]
            print(f"  Age:    CUDA={cuda_ages[i]}  TRT={trt_ages[i]}  diff={age_diff}")
            print(f"  Gender: CUDA={cuda_genders[i]}  TRT={trt_genders[i]}  match={gender_match}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Faces compared:       {n}")
    print(f"  Cosine similarity:    min={min(cosines):.6f}  mean={np.mean(cosines):.6f}  max={max(cosines):.6f}")
    print(f"  Max absolute diff:    min={min(max_abs_diffs):.6f}  mean={np.mean(max_abs_diffs):.6f}  max={max(max_abs_diffs):.6f}")
    print(f"  L2 distance:          min={min(l2_dists):.6f}  mean={np.mean(l2_dists):.6f}  max={max(l2_dists):.6f}")

    # Verdict
    min_cos = min(cosines)
    if min_cos >= 0.999:
        print(f"\n  VERDICT: EXCELLENT — min cosine {min_cos:.6f} >= 0.999")
    elif min_cos >= 0.995:
        print(f"\n  VERDICT: GOOD — min cosine {min_cos:.6f} >= 0.995")
    elif min_cos >= 0.990:
        print(f"\n  VERDICT: ACCEPTABLE — min cosine {min_cos:.6f} >= 0.990")
    else:
        print(f"\n  VERDICT: CONCERNING — min cosine {min_cos:.6f} < 0.990")
        print("  FP16 may be degrading embedding quality significantly.")


if __name__ == "__main__":
    main()
