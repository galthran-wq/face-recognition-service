#!/usr/bin/env python3
"""Benchmark + equivalence check for dynamic-batch SCRFD detection (issue #126).

Loads the provider (which re-exports the detector to [N,3,640,640] on startup),
then compares, on the same images and in the same process:

- sequential: N x det_model.detect() — the pre-#126 per-image path, which still
  works on the converted graph via insightface's own `batched` handling;
- batched:    provider._detect_batch() — one session.run per chunk.

Reports per-pass latency, images/s, speedup, and per-image equivalence of the
two paths (face counts, and max bbox/score deltas).

Usage:
    python3 benchmarks/benchmark_batched_det.py                       # CPU, synthetic images
    python3 benchmarks/benchmark_batched_det.py --images "photos/*.jpg" --gpu
    python3 benchmarks/benchmark_batched_det.py --gpu --tensorrt --trt-cache /models/trt_cache
    python3 benchmarks/benchmark_batched_det.py --batch-size 32 --iters 10 --gpu
"""

from __future__ import annotations

import argparse
import glob
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.services.face_provider.insightface import InsightFaceProvider  # noqa: E402


def load_images(pattern, batch_size):
    if pattern:
        paths = sorted(glob.glob(str(Path(pattern).expanduser())))
        if not paths:
            sys.exit(f"no images match {pattern}")
        imgs = [cv2.imread(p) for p in paths]
        imgs = [im for im in imgs if im is not None]
        # cycle up to batch_size
        while len(imgs) < batch_size:
            imgs.append(imgs[len(imgs) % len(paths)].copy())
        return imgs[:batch_size], True
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 255, size=(720, 1280, 3), dtype=np.uint8).astype(np.uint8) for _ in range(batch_size)]
    print("note: synthetic noise images (no faces) — equivalence check is trivial, pass --images for a real one")
    return imgs, False


def compare(seq_results, bat_results):
    max_bbox_delta = 0.0
    max_score_delta = 0.0
    for i, ((sb, _sk), (bb, _bk)) in enumerate(zip(seq_results, bat_results)):
        if sb.shape[0] != bb.shape[0]:
            print(f"  MISMATCH image {i}: sequential {sb.shape[0]} faces vs batched {bb.shape[0]}")
            return False
        if sb.shape[0]:
            max_bbox_delta = max(max_bbox_delta, float(np.max(np.abs(sb[:, :4] - bb[:, :4]))))
            max_score_delta = max(max_score_delta, float(np.max(np.abs(sb[:, 4] - bb[:, 4]))))
    print(f"  face counts identical; max bbox delta {max_bbox_delta:.4f} px, max score delta {max_score_delta:.6f}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", help="glob of test images (defaults to synthetic noise)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--tensorrt", action="store_true")
    ap.add_argument("--trt-cache", default="/models/trt_cache")
    ap.add_argument("--model-dir", default="~/.insightface")
    ap.add_argument("--det-max-batch", type=int, default=32)
    args = ap.parse_args()

    provider = InsightFaceProvider(
        use_gpu=args.gpu,
        model_dir=args.model_dir,
        use_tensorrt=args.tensorrt,
        trt_cache_path=args.trt_cache,
        det_trt_max_batch=args.det_max_batch,
    )
    t0 = time.perf_counter()
    provider.load_model()
    print(f"model loaded in {time.perf_counter() - t0:.1f}s; batch-capable: {provider._det_batch_capable()}")
    if not provider._det_batch_capable():
        sys.exit("detector graph is not batch-capable — conversion failed?")

    imgs, real = load_images(args.images, args.batch_size)
    det_model = provider._app.det_model

    # Warmup both paths (TRT engine build / cuDNN autotune happen here).
    det_model.detect(imgs[0], max_num=0, metric="default")
    provider._detect_batch(imgs)

    print(f"\nequivalence on {len(imgs)} images:")
    seq_results = [det_model.detect(im, max_num=0, metric="default") for im in imgs]
    bat_results = provider._detect_batch(imgs)
    ok = compare(seq_results, bat_results)
    if real and not ok:
        sys.exit(1)

    seq_times, bat_times = [], []
    for _ in range(args.iters):
        t = time.perf_counter()
        for im in imgs:
            det_model.detect(im, max_num=0, metric="default")
        seq_times.append(time.perf_counter() - t)

        t = time.perf_counter()
        provider._detect_batch(imgs)
        bat_times.append(time.perf_counter() - t)

    seq = statistics.median(seq_times)
    bat = statistics.median(bat_times)
    n = len(imgs)
    print(f"\nsequential: {seq * 1000:8.1f} ms/pass  ({n / seq:7.1f} img/s)")
    print(f"batched:    {bat * 1000:8.1f} ms/pass  ({n / bat:7.1f} img/s)")
    print(f"speedup:    {seq / bat:.2f}x  (batch={n}, det_max_batch={args.det_max_batch})")


if __name__ == "__main__":
    main()
