#!/usr/bin/env python3
"""Profile batch embed pipeline to find where time is spent.

Breaks down a batch=32 embed call into individual stages with timing.

Usage (inside GPU container):
    python3 benchmarks/profile_batch.py --model-dir /models
    python3 benchmarks/profile_batch.py --model-dir /models --tensorrt
    python3 benchmarks/profile_batch.py --model-dir /models --batch-size 16
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def load_model(use_gpu, model_dir, det_size, use_tensorrt, trt_cache):
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
    elif use_gpu:
        fa_kwargs["providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        fa_kwargs["provider_options"] = [{"device_id": "0"}, {}]
    else:
        fa_kwargs["providers"] = ["CPUExecutionProvider"]

    app = FaceAnalysis(name="buffalo_l", root=model_dir, **fa_kwargs)
    app.prepare(ctx_id=0 if use_gpu else -1, det_size=det_size)
    PickableInferenceSession.__init__ = _original_init
    return app


def load_image(image_path):
    if image_path and os.path.isfile(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            return img
    try:
        import insightface
        p = Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"
        img = cv2.imread(str(p))
        if img is not None:
            print(f"Using bundled test image: {p}")
            return img
    except Exception:
        pass
    print("ERROR: No test image found.", file=sys.stderr)
    sys.exit(1)


def _estimate_norm(lmk, image_size=112):
    ARCFACE_DST = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
         [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
    ratio = float(image_size) / 112.0
    dst = ARCFACE_DST * ratio
    n = lmk.shape[0]
    A = np.zeros((n * 2, 4), dtype=np.float64)
    b = np.zeros(n * 2, dtype=np.float64)
    for i in range(n):
        A[2 * i] = [lmk[i, 0], -lmk[i, 1], 1, 0]
        A[2 * i + 1] = [lmk[i, 1], lmk[i, 0], 0, 1]
        b[2 * i] = dst[i, 0]
        b[2 * i + 1] = dst[i, 1]
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b_val, tx, ty = params
    return np.array([[a, -b_val, tx], [b_val, a, ty]], dtype=np.float64)


def _norm_crop(img, landmark, image_size=112):
    M = _estimate_norm(landmark, image_size)
    return cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)


def profile_batch(app, img, batch_size, num_iter, warmup, threaded=False):
    from concurrent.futures import ThreadPoolExecutor

    rec_model = app.models["recognition"]
    det_model = app.det_model
    thread_workers = 4

    # Prepare: encode images to JPEG bytes (simulating what the service receives)
    _, jpeg_buf = cv2.imencode(".jpg", img)
    jpeg_bytes = jpeg_buf.tobytes()

    # Timing accumulators
    timings = {
        "total": [],
        "decode": [],
        "detect": [],
        "align": [],
        "recognition": [],
        "normalize": [],
        "build_response": [],
    }

    for iteration in range(warmup + num_iter):
        t_total_start = time.perf_counter()

        # --- Stage 1: Decode images ---
        t0 = time.perf_counter()
        def _decode_one(_):
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if threaded:
            with ThreadPoolExecutor(max_workers=thread_workers) as pool:
                decoded_images = list(pool.map(_decode_one, range(batch_size)))
        else:
            decoded_images = [_decode_one(i) for i in range(batch_size)]
        t_decode = time.perf_counter() - t0

        # --- Stage 2: Detection (per image, GPU — must be sequential) ---
        t0 = time.perf_counter()
        all_bboxes = []
        all_kpss = []
        for decoded in decoded_images:
            bboxes, kpss = det_model.detect(decoded, max_num=0, metric="default")
            all_bboxes.append(bboxes)
            all_kpss.append(kpss)
        t_detect = time.perf_counter() - t0

        # --- Stage 3: Alignment (norm_crop per face) ---
        t0 = time.perf_counter()
        all_crops = []
        crop_counts = []
        align_tasks = []
        for i in range(batch_size):
            bboxes = all_bboxes[i]
            kpss = all_kpss[i]
            if bboxes.shape[0] == 0 or kpss is None:
                crop_counts.append(0)
                continue
            for kps in kpss:
                align_tasks.append((decoded_images[i], kps))
            crop_counts.append(bboxes.shape[0])

        if threaded and align_tasks:
            with ThreadPoolExecutor(max_workers=thread_workers) as pool:
                all_crops = list(pool.map(
                    lambda t: _norm_crop(t[0], landmark=t[1], image_size=rec_model.input_size[0]),
                    align_tasks))
        else:
            for img_a, kps_a in align_tasks:
                all_crops.append(_norm_crop(img_a, landmark=kps_a, image_size=rec_model.input_size[0]))
        t_align = time.perf_counter() - t0

        # --- Stage 4: Batched recognition ---
        t0 = time.perf_counter()
        if all_crops:
            all_embeddings = rec_model.get_feat(all_crops)
        else:
            all_embeddings = np.zeros((0, 512), dtype=np.float32)
        t_recognition = time.perf_counter() - t0

        # --- Stage 5: Normalize ---
        t0 = time.perf_counter()
        if all_crops:
            norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            all_embeddings = all_embeddings / norms
        t_normalize = time.perf_counter() - t0

        # --- Stage 6: Build response (simulate tolist() + dict creation) ---
        t0 = time.perf_counter()
        results = []
        emb_offset = 0
        for i in range(batch_size):
            n = crop_counts[i]
            faces = []
            for j in range(n):
                bbox = all_bboxes[i][j]
                face = {
                    "bbox": {
                        "x": float(bbox[0]),
                        "y": float(bbox[1]),
                        "width": float(bbox[2] - bbox[0]),
                        "height": float(bbox[3] - bbox[1]),
                    },
                    "det_score": float(all_bboxes[i][j, 4]),
                    "embedding": all_embeddings[emb_offset + j].tolist(),
                }
                faces.append(face)
            emb_offset += n
            results.append({"index": i, "faces": faces, "face_count": n})
        t_build = time.perf_counter() - t0

        t_total = time.perf_counter() - t_total_start

        if iteration >= warmup:
            timings["total"].append(t_total)
            timings["decode"].append(t_decode)
            timings["detect"].append(t_detect)
            timings["align"].append(t_align)
            timings["recognition"].append(t_recognition)
            timings["normalize"].append(t_normalize)
            timings["build_response"].append(t_build)

    return timings


def main():
    parser = argparse.ArgumentParser(description="Profile batch embed pipeline")
    parser.add_argument("--model-dir", default=os.path.expanduser("~/.insightface"))
    parser.add_argument("--image", default=None)
    parser.add_argument("--det-size", default="640,640")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-iter", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--gpu", action="store_true", default=True)
    parser.add_argument("--tensorrt", action="store_true", default=False)
    parser.add_argument("--trt-cache", default="/tmp/trt_cache_profile")
    parser.add_argument("--threaded", action="store_true", default=False, help="Use threaded decode + align")
    args = parser.parse_args()

    det_size = tuple(int(x) for x in args.det_size.split(","))
    img = load_image(args.image)
    print(f"Image: {img.shape[1]}x{img.shape[0]}")

    # Detect faces in the image to report face count
    app = load_model(args.gpu, args.model_dir, det_size, args.tensorrt, args.trt_cache)
    bboxes, _ = app.det_model.detect(img, max_num=0, metric="default")
    n_faces = bboxes.shape[0]
    total_faces = n_faces * args.batch_size

    print(f"Faces per image: {n_faces}")
    print(f"Batch size: {args.batch_size} images ({total_faces} faces total)")
    print(f"Iterations: {args.num_iter} (+ {args.warmup} warmup)")
    backend = "TensorRT FP16" if args.tensorrt else "CUDA EP"
    mode = "threaded" if args.threaded else "sequential"
    print(f"Backend: {backend}, mode: {mode}")
    print()

    timings = profile_batch(app, img, args.batch_size, args.num_iter, args.warmup, threaded=args.threaded)

    # Print results
    print(f"{'='*65}")
    print(f"BATCH EMBED PROFILE — {args.batch_size} images, {total_faces} faces, {backend} ({mode})")
    print(f"{'='*65}")
    print(f"{'Stage':<20} {'p50 (ms)':>10} {'mean (ms)':>10} {'% of total':>10}")
    print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10}")

    import statistics

    total_p50 = statistics.median(timings["total"]) * 1000

    for stage in ["decode", "detect", "align", "recognition", "normalize", "build_response"]:
        vals = timings[stage]
        p50 = statistics.median(vals) * 1000
        mean = statistics.mean(vals) * 1000
        pct = (p50 / total_p50) * 100
        print(f"{stage:<20} {p50:>10.1f} {mean:>10.1f} {pct:>9.1f}%")

    total_mean = statistics.mean(timings["total"]) * 1000
    print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    print(f"{'TOTAL':<20} {total_p50:>10.1f} {total_mean:>10.1f} {'100.0':>9}%")

    print(f"\nPer image:  {total_p50 / args.batch_size:.1f}ms")
    print(f"Images/sec: {args.batch_size / (total_p50 / 1000):.0f}")
    print(f"Faces/sec:  {total_faces / (total_p50 / 1000):.0f}")

    # Overhead = total - (detect + recognition)
    detect_p50 = statistics.median(timings["detect"]) * 1000
    rec_p50 = statistics.median(timings["recognition"]) * 1000
    model_time = detect_p50 + rec_p50
    overhead = total_p50 - model_time
    print(f"\nModel inference: {model_time:.1f}ms ({model_time/total_p50*100:.0f}%)")
    print(f"Python overhead: {overhead:.1f}ms ({overhead/total_p50*100:.0f}%) — decode + align + normalize + response build")


if __name__ == "__main__":
    main()
