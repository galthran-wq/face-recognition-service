#!/usr/bin/env python3
"""Test whether cv2.imdecode and norm_crop release the GIL.

If they do, ThreadPoolExecutor should give near-linear speedup.

Usage (inside GPU container):
    python3 benchmarks/test_gil.py
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def load_test_image():
    try:
        import insightface
        p = Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"
        img = cv2.imread(str(p))
        if img is not None:
            return img
    except Exception:
        pass
    print("ERROR: No test image found.", file=sys.stderr)
    sys.exit(1)


def bench_decode(jpeg_bytes: bytes, n: int, workers: int) -> float:
    """Decode the same JPEG n times using `workers` threads."""
    def decode_one(_):
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    t0 = time.perf_counter()
    if workers <= 1:
        for i in range(n):
            decode_one(i)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(decode_one, range(n)))
    return time.perf_counter() - t0


def bench_align(img: np.ndarray, kpss: np.ndarray, n_repeats: int, workers: int, use_custom: bool = False) -> float:
    """Run norm_crop on all keypoints, repeated n_repeats times."""
    if use_custom:
        # Our custom numpy-based norm_crop
        _ARCFACE_DST = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
             [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)

        def align_one(kps):
            dst = _ARCFACE_DST.copy()
            n = kps.shape[0]
            A = np.zeros((n * 2, 4), dtype=np.float64)
            b = np.zeros(n * 2, dtype=np.float64)
            for i in range(n):
                A[2 * i] = [kps[i, 0], -kps[i, 1], 1, 0]
                A[2 * i + 1] = [kps[i, 1], kps[i, 0], 0, 1]
                b[2 * i] = dst[i, 0]
                b[2 * i + 1] = dst[i, 1]
            params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            a, b_val, tx, ty = params
            M = np.array([[a, -b_val, tx], [b_val, a, ty]], dtype=np.float64)
            return cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
    else:
        from insightface.utils import face_align
        def align_one(kps):
            return face_align.norm_crop(img, landmark=kps, image_size=112)

    # Build full list of keypoints (n_repeats * len(kpss))
    all_kps = []
    for _ in range(n_repeats):
        for kps in kpss:
            all_kps.append(kps)

    t0 = time.perf_counter()
    if workers <= 1:
        for kps in all_kps:
            align_one(kps)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(align_one, all_kps))
    return time.perf_counter() - t0


def main():
    img = load_test_image()
    _, jpeg_buf = cv2.imencode(".jpg", img)
    jpeg_bytes = jpeg_buf.tobytes()

    # Get keypoints for alignment test
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", root=os.path.expanduser("~/.insightface"),
                           providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        bboxes, kpss = app.det_model.detect(img, max_num=0, metric="default")
        print(f"Detected {bboxes.shape[0]} faces for alignment test")
    except Exception as e:
        print(f"Could not load model for alignment test: {e}")
        kpss = None

    n_decode = 128
    print(f"\n{'='*55}")
    print(f"IMAGE DECODE — cv2.imdecode x{n_decode}")
    print(f"{'='*55}")
    print(f"{'Workers':<10} {'Time (ms)':>10} {'Speedup':>10}")
    print(f"{'-'*10} {'-'*10} {'-'*10}")

    baseline = None
    for workers in [1, 2, 4, 8]:
        # Warmup
        bench_decode(jpeg_bytes, 8, workers)
        # Measure
        elapsed = bench_decode(jpeg_bytes, n_decode, workers)
        if baseline is None:
            baseline = elapsed
        speedup = baseline / elapsed
        print(f"{workers:<10} {elapsed*1000:>10.1f} {speedup:>9.1f}x")

    if kpss is not None and kpss.shape[0] > 0:
        n_align_repeats = 32  # repeat keypoints to get enough work
        total_crops = kpss.shape[0] * n_align_repeats

        for label, use_custom in [("skimage (original)", False), ("numpy (custom)", True)]:
            print(f"\n{'='*55}")
            print(f"FACE ALIGNMENT — {label} x{total_crops}")
            print(f"{'='*55}")
            print(f"{'Workers':<10} {'Time (ms)':>10} {'Speedup':>10}")
            print(f"{'-'*10} {'-'*10} {'-'*10}")

            baseline = None
            for workers in [1, 2, 4, 8]:
                # Warmup
                bench_align(img, kpss, 2, workers, use_custom=use_custom)
                # Measure
                elapsed = bench_align(img, kpss, n_align_repeats, workers, use_custom=use_custom)
                if baseline is None:
                    baseline = elapsed
                speedup = baseline / elapsed
                print(f"{workers:<10} {elapsed*1000:>10.1f} {speedup:>9.1f}x")

    cpu_count = os.cpu_count()
    print(f"\nCPU cores available: {cpu_count}")


if __name__ == "__main__":
    main()
