#!/usr/bin/env python3
"""Benchmark face detection and embedding throughput.

Measures single-image latency and batched recognition throughput
across configurable batch sizes. Outputs results as a console table
and optionally as a JSON file.

Usage:
    uv run python benchmarks/benchmark.py --help
    uv run python benchmarks/benchmark.py --cpu --num-images 50
    uv run python benchmarks/benchmark.py --gpu --batch-sizes 1,4,8,16,32,64
    uv run python benchmarks/benchmark.py --gpu --image path/to/photo.jpg
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np  # noqa: TC002

# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------


@dataclass
class LatencyStats:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float


@dataclass
class BatchResult:
    batch_size: int
    total_faces: int
    total_time_s: float
    faces_per_sec: float
    latency: LatencyStats


@dataclass
class BenchmarkReport:
    timestamp: str
    device: str
    model_name: str
    det_size: list[int]
    image_path: str
    image_resolution: list[int]
    num_warmup: int
    num_images: int
    faces_per_image: int
    gpu_info: dict[str, object] | None
    single_image: dict[str, LatencyStats] | None
    recognition_batch: list[BatchResult]
    end_to_end_batch: list[BatchResult]


# ---------------------------------------------------------------------------
# GPU info helper
# ---------------------------------------------------------------------------


def get_gpu_info() -> dict[str, object] | None:
    try:
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "name": pynvml.nvmlDeviceGetName(handle),
            "memory_total_mb": round(mem.total / 1024**2),
            "memory_used_mb": round(mem.used / 1024**2),
            "memory_free_mb": round(mem.free / 1024**2),
            "driver_version": pynvml.nvmlSystemGetDriverVersion(),
        }
    except Exception:
        return None


def get_gpu_memory_mb() -> float | None:
    try:
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return round(mem.used / 1024**2, 1)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Latency measurement helpers
# ---------------------------------------------------------------------------


def compute_latency(timings_s: list[float]) -> LatencyStats:
    ms = [t * 1000 for t in timings_s]
    ms.sort()
    return LatencyStats(
        p50_ms=round(statistics.median(ms), 2),
        p95_ms=round(ms[int(len(ms) * 0.95)] if len(ms) > 1 else ms[0], 2),
        p99_ms=round(ms[int(len(ms) * 0.99)] if len(ms) > 1 else ms[0], 2),
        mean_ms=round(statistics.mean(ms), 2),
        min_ms=round(min(ms), 2),
        max_ms=round(max(ms), 2),
    )


# ---------------------------------------------------------------------------
# Benchmark routines
# ---------------------------------------------------------------------------


def load_model(use_gpu: bool, model_name: str, det_size: tuple[int, int], model_dir: str):
    """Load InsightFace FaceAnalysis model and return the app instance."""
    from insightface.app import FaceAnalysis  # type: ignore[import-untyped]

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    app = FaceAnalysis(name=model_name, root=model_dir, providers=providers)
    app.prepare(ctx_id=0 if use_gpu else -1, det_size=det_size)
    return app


def load_image(image_path: str | None) -> np.ndarray:
    """Load a test image. Falls back to insightface's bundled t1.jpg."""
    if image_path and os.path.isfile(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            return img

    # Fallback: use insightface bundled test image
    try:
        import insightface  # type: ignore[import-untyped]

        data_dir = Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"
        img = cv2.imread(str(data_dir))
        if img is not None:
            print(f"Using bundled test image: {data_dir}")
            return img
    except Exception:
        pass

    print("ERROR: No test image found. Provide one with --image.", file=sys.stderr)
    sys.exit(1)


def benchmark_single_image(app, img: np.ndarray, num_iter: int, warmup: int) -> dict[str, LatencyStats]:
    """Benchmark single-image inference for detect, embed (get), and full pipeline."""
    det_model = app.det_model
    rec_model = app.models.get("recognition")

    results: dict[str, LatencyStats] = {}

    # --- Detection only ---
    print("  Benchmarking detection...")
    timings = []
    for i in range(warmup + num_iter):
        t0 = time.perf_counter()
        det_model.detect(img, max_num=0, metric="default")
        elapsed = time.perf_counter() - t0
        if i >= warmup:
            timings.append(elapsed)
    results["detect"] = compute_latency(timings)

    # --- Full pipeline (detect + align + embed) via app.get() ---
    print("  Benchmarking full pipeline (detect + embed)...")
    timings = []
    for i in range(warmup + num_iter):
        t0 = time.perf_counter()
        app.get(img)
        elapsed = time.perf_counter() - t0
        if i >= warmup:
            timings.append(elapsed)
    results["full_pipeline"] = compute_latency(timings)

    # --- Recognition only (single aligned face) ---
    if rec_model is not None:
        from insightface.utils import face_align  # type: ignore[import-untyped]

        print("  Benchmarking recognition only (single face)...")
        faces = app.get(img)
        if faces:
            face = faces[0]
            aimg = face_align.norm_crop(img, landmark=face.kps, image_size=rec_model.input_size[0])
            timings = []
            for i in range(warmup + num_iter):
                t0 = time.perf_counter()
                rec_model.get_feat(aimg)
                elapsed = time.perf_counter() - t0
                if i >= warmup:
                    timings.append(elapsed)
            results["recognition_single"] = compute_latency(timings)

    return results


def benchmark_recognition_batch(
    app,
    img: np.ndarray,
    batch_sizes: list[int],
    num_iter: int,
    warmup: int,
) -> list[BatchResult]:
    """Benchmark batched recognition by calling rec_model.get_feat() with multiple aligned faces."""
    rec_model = app.models.get("recognition")
    if rec_model is None:
        print("  WARNING: No recognition model found, skipping batch recognition benchmark.")
        return []

    from insightface.utils import face_align  # type: ignore[import-untyped]

    # Detect faces and align them
    faces = app.get(img)
    if not faces:
        print("  WARNING: No faces detected in test image, skipping batch recognition benchmark.")
        return []

    # Create aligned face crops
    aligned = []
    for face in faces:
        aimg = face_align.norm_crop(img, landmark=face.kps, image_size=rec_model.input_size[0])
        aligned.append(aimg)

    faces_per_image = len(aligned)
    print(f"  Detected {faces_per_image} face(s) in test image for batch benchmarking.")

    results: list[BatchResult] = []

    for bs in batch_sizes:
        # Build a batch by repeating aligned faces to reach batch_size
        batch_imgs = []
        for i in range(bs):
            batch_imgs.append(aligned[i % faces_per_image])

        timings = []
        for i in range(warmup + num_iter):
            t0 = time.perf_counter()
            rec_model.get_feat(batch_imgs)
            elapsed = time.perf_counter() - t0
            if i >= warmup:
                timings.append(elapsed)

        latency = compute_latency(timings)
        total_time = sum(t for t in timings)
        total_faces = bs * num_iter
        fps = total_faces / total_time if total_time > 0 else 0

        results.append(
            BatchResult(
                batch_size=bs,
                total_faces=total_faces,
                total_time_s=round(total_time, 3),
                faces_per_sec=round(fps, 1),
                latency=latency,
            )
        )
        print(
            f"    batch_size={bs:>3d}  |  {fps:>8.1f} faces/sec"
            f"  |  p50={latency.p50_ms:.1f}ms  p95={latency.p95_ms:.1f}ms"
        )

    return results


def benchmark_end_to_end_batch(
    app,
    img: np.ndarray,
    batch_sizes: list[int],
    num_iter: int,
    warmup: int,
) -> list[BatchResult]:
    """Benchmark end-to-end (detect+embed) processing N images sequentially, simulating batch API behavior."""
    results: list[BatchResult] = []

    for bs in batch_sizes:
        timings = []
        for i in range(warmup + num_iter):
            t0 = time.perf_counter()
            total_faces_in_batch = 0
            for _ in range(bs):
                faces = app.get(img)
                total_faces_in_batch += len(faces)
            elapsed = time.perf_counter() - t0
            if i >= warmup:
                timings.append(elapsed)

        latency = compute_latency(timings)
        total_time = sum(t for t in timings)
        # faces detected per iteration * num_iter
        faces_per_iter = total_faces_in_batch  # last iteration count
        total_faces = faces_per_iter * num_iter
        fps = total_faces / total_time if total_time > 0 else 0

        results.append(
            BatchResult(
                batch_size=bs,
                total_faces=total_faces,
                total_time_s=round(total_time, 3),
                faces_per_sec=round(fps, 1),
                latency=latency,
            )
        )
        print(
            f"    batch_size={bs:>3d}  |  {fps:>8.1f} faces/sec"
            f"  |  p50={latency.p50_ms:.1f}ms  p95={latency.p95_ms:.1f}ms"
        )

    return results


def benchmark_batched_detection(
    app,
    batched_scrfd,
    img: np.ndarray,
    batch_sizes: list[int],
    num_iter: int,
    warmup: int,
) -> tuple[list[BatchResult], list[BatchResult]]:
    """Benchmark batched vs sequential detection across batch sizes."""
    seq_results: list[BatchResult] = []
    bat_results: list[BatchResult] = []

    det_model = app.det_model
    faces_in_img = len(app.get(img))

    for bs in batch_sizes:
        images = [img] * bs

        # Sequential detection (current behavior)
        timings = []
        for i in range(warmup + num_iter):
            t0 = time.perf_counter()
            for im in images:
                det_model.detect(im, max_num=0, metric="default")
            elapsed = time.perf_counter() - t0
            if i >= warmup:
                timings.append(elapsed)

        latency = compute_latency(timings)
        total_time = sum(timings)
        total_faces = faces_in_img * bs * num_iter
        fps = total_faces / total_time if total_time > 0 else 0
        seq_results.append(
            BatchResult(batch_size=bs, total_faces=total_faces, total_time_s=round(total_time, 3),
                        faces_per_sec=round(fps, 1), latency=latency)
        )

        # Batched detection
        timings = []
        for i in range(warmup + num_iter):
            t0 = time.perf_counter()
            batched_scrfd.detect_batch(images)
            elapsed = time.perf_counter() - t0
            if i >= warmup:
                timings.append(elapsed)

        latency = compute_latency(timings)
        total_time = sum(timings)
        fps = total_faces / total_time if total_time > 0 else 0
        bat_results.append(
            BatchResult(batch_size=bs, total_faces=total_faces, total_time_s=round(total_time, 3),
                        faces_per_sec=round(fps, 1), latency=latency)
        )

        speedup = seq_results[-1].latency.p50_ms / bat_results[-1].latency.p50_ms if bat_results[-1].latency.p50_ms > 0 else 0
        print(
            f"    batch_size={bs:>3d}  |  seq p50={seq_results[-1].latency.p50_ms:.1f}ms"
            f"  bat p50={bat_results[-1].latency.p50_ms:.1f}ms"
            f"  speedup={speedup:.2f}x"
        )

    return seq_results, bat_results


def benchmark_e2e_batched(
    app,
    batched_scrfd,
    img: np.ndarray,
    batch_sizes: list[int],
    num_iter: int,
    warmup: int,
) -> tuple[list[BatchResult], list[BatchResult]]:
    """End-to-end (detect+align+embed) with sequential vs batched detection."""
    from insightface.utils import face_align  # type: ignore[import-untyped]

    rec_model = app.models.get("recognition")
    if rec_model is None:
        return [], []

    faces_in_img = len(app.get(img))
    seq_results: list[BatchResult] = []
    bat_results: list[BatchResult] = []

    for bs in batch_sizes:
        images = [img] * bs

        # Sequential: detect one-by-one, then batch embed
        timings = []
        for i in range(warmup + num_iter):
            t0 = time.perf_counter()
            all_crops = []
            for im in images:
                bboxes, kpss = app.det_model.detect(im, max_num=0, metric="default")
                if bboxes.shape[0] > 0 and kpss is not None:
                    for kps in kpss:
                        aimg = face_align.norm_crop(im, landmark=kps, image_size=rec_model.input_size[0])
                        all_crops.append(aimg)
            if all_crops:
                rec_model.get_feat(all_crops)
            elapsed = time.perf_counter() - t0
            if i >= warmup:
                timings.append(elapsed)

        latency = compute_latency(timings)
        total_time = sum(timings)
        total_faces = faces_in_img * bs * num_iter
        fps = total_faces / total_time if total_time > 0 else 0
        seq_results.append(
            BatchResult(batch_size=bs, total_faces=total_faces, total_time_s=round(total_time, 3),
                        faces_per_sec=round(fps, 1), latency=latency)
        )

        # Batched: detect all at once, then batch embed
        timings = []
        for i in range(warmup + num_iter):
            t0 = time.perf_counter()
            detections = batched_scrfd.detect_batch(images)
            all_crops = []
            for b_idx, (bboxes, kpss) in enumerate(detections):
                if bboxes.shape[0] > 0 and kpss is not None:
                    for kps in kpss:
                        aimg = face_align.norm_crop(images[b_idx], landmark=kps, image_size=rec_model.input_size[0])
                        all_crops.append(aimg)
            if all_crops:
                rec_model.get_feat(all_crops)
            elapsed = time.perf_counter() - t0
            if i >= warmup:
                timings.append(elapsed)

        latency = compute_latency(timings)
        total_time = sum(timings)
        fps = total_faces / total_time if total_time > 0 else 0
        bat_results.append(
            BatchResult(batch_size=bs, total_faces=total_faces, total_time_s=round(total_time, 3),
                        faces_per_sec=round(fps, 1), latency=latency)
        )

        speedup = seq_results[-1].latency.p50_ms / bat_results[-1].latency.p50_ms if bat_results[-1].latency.p50_ms > 0 else 0
        print(
            f"    batch_size={bs:>3d}  |  seq p50={seq_results[-1].latency.p50_ms:.1f}ms"
            f"  bat p50={bat_results[-1].latency.p50_ms:.1f}ms"
            f"  speedup={speedup:.2f}x"
        )

    return seq_results, bat_results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_table(title: str, rows: list[BatchResult]) -> None:
    if not rows:
        return
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    header = (
        f"{'Batch':>6} | {'Faces/sec':>10} | {'p50 (ms)':>9} | {'p95 (ms)':>9} | {'p99 (ms)':>9} | {'Mean (ms)':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.batch_size:>6} | {r.faces_per_sec:>10.1f} | {r.latency.p50_ms:>9.1f} | "
            f"{r.latency.p95_ms:>9.1f} | {r.latency.p99_ms:>9.1f} | {r.latency.mean_ms:>9.1f}"
        )


def save_report(report: BenchmarkReport, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"benchmark_{ts}.json")

    def serialize(obj: object) -> object:
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)  # type: ignore[arg-type]
        return str(obj)

    with open(path, "w") as f:
        json.dump(asdict(report), f, indent=2, default=serialize)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark face recognition throughput")
    device = p.add_mutually_exclusive_group()
    device.add_argument("--gpu", action="store_true", help="Use GPU (CUDA)")
    device.add_argument("--cpu", action="store_true", default=True, help="Use CPU (default)")
    p.add_argument("--image", type=str, default=None, help="Path to test image (defaults to bundled t1.jpg)")
    p.add_argument("--model-name", type=str, default="buffalo_l", help="InsightFace model name")
    p.add_argument("--model-dir", type=str, default="~/.insightface", help="Model directory")
    p.add_argument("--det-size", type=int, default=640, help="Detection size (square)")
    p.add_argument("--num-images", type=int, default=50, help="Number of iterations per benchmark")
    p.add_argument("--warmup", type=int, default=5, help="Warmup iterations (discarded)")
    p.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16,32,64",
        help="Comma-separated batch sizes to test",
    )
    p.add_argument("--output-dir", type=str, default="benchmarks/results", help="Directory for JSON output")
    p.add_argument("--no-save", action="store_true", help="Skip saving JSON results")
    p.add_argument("--skip-single", action="store_true", help="Skip single-image benchmarks")
    p.add_argument("--skip-e2e", action="store_true", help="Skip end-to-end batch benchmarks (slow)")
    p.add_argument(
        "--batch-det-model",
        type=str,
        default="",
        help="Path to batched SCRFD ONNX model (enables batched detection benchmark)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    use_gpu = args.gpu
    device_str = "GPU (CUDA)" if use_gpu else "CPU"
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    det_size = (args.det_size, args.det_size)

    print("\nFace Recognition Benchmark")
    print(f"{'=' * 40}")
    print(f"  Device:     {device_str}")
    print(f"  Model:      {args.model_name}")
    print(f"  Det size:   {det_size}")
    print(f"  Iterations: {args.num_images}")
    print(f"  Warmup:     {args.warmup}")
    print(f"  Batch sizes: {batch_sizes}")

    # GPU info
    gpu_info = None
    if use_gpu:
        gpu_info = get_gpu_info()
        if gpu_info:
            print(f"  GPU:        {gpu_info['name']}")
            print(f"  VRAM:       {gpu_info['memory_total_mb']} MB")
        else:
            print("  WARNING: pynvml not available, GPU info unavailable.")
            print("           Install with: pip install pynvml")

    # Load model
    print("\nLoading model...")
    t0 = time.perf_counter()
    app = load_model(use_gpu, args.model_name, det_size, args.model_dir)
    load_time = time.perf_counter() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    # Load image
    img = load_image(args.image)
    h, w = img.shape[:2]
    print(f"  Image resolution: {w}x{h}")

    # Detect faces to get count
    faces = app.get(img)
    faces_per_image = len(faces)
    print(f"  Faces detected: {faces_per_image}")

    if faces_per_image == 0:
        print("\nERROR: No faces detected in image. Use an image with at least one face.", file=sys.stderr)
        sys.exit(1)

    # --- Single image benchmark ---
    single_results = None
    if not args.skip_single:
        print(f"\n--- Single-Image Latency ({args.num_images} iterations, {args.warmup} warmup) ---")
        single_results = benchmark_single_image(app, img, args.num_images, args.warmup)
        for mode, stats in single_results.items():
            print(f"  {mode:>20s}: p50={stats.p50_ms:.1f}ms  p95={stats.p95_ms:.1f}ms  mean={stats.mean_ms:.1f}ms")

    # --- Batched recognition benchmark ---
    print("\n--- Batched Recognition (embedding only, varying batch size) ---")
    rec_results = benchmark_recognition_batch(app, img, batch_sizes, args.num_images, args.warmup)
    print_table("Recognition Batch Results (embedding only)", rec_results)

    # --- End-to-end batch benchmark ---
    e2e_results: list[BatchResult] = []
    if not args.skip_e2e:
        print("\n--- End-to-End Batch (detect + embed per image, sequential) ---")
        e2e_results = benchmark_end_to_end_batch(app, img, batch_sizes, args.num_images, args.warmup)
        print_table("End-to-End Sequential Results", e2e_results)

    # --- Batched detection benchmark ---
    if args.batch_det_model and os.path.isfile(args.batch_det_model):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.services.face_provider.batched_scrfd import BatchedSCRFD

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        batched_scrfd = BatchedSCRFD(args.batch_det_model, det_size=det_size, providers=providers)
        print(f"\n  Loaded batched SCRFD: {args.batch_det_model}")

        print("\n--- Batched Detection (sequential vs batched SCRFD) ---")
        det_seq, det_bat = benchmark_batched_detection(
            app, batched_scrfd, img, batch_sizes, args.num_images, args.warmup
        )
        print_table("Detection: Sequential (baseline)", det_seq)
        print_table("Detection: Batched SCRFD", det_bat)

        print("\n--- End-to-End Batched (detect+align+embed: sequential vs batched det) ---")
        e2e_seq, e2e_bat = benchmark_e2e_batched(
            app, batched_scrfd, img, batch_sizes, args.num_images, args.warmup
        )
        print_table("E2E: Sequential Detection", e2e_seq)
        print_table("E2E: Batched Detection", e2e_bat)

        # Summary
        if e2e_seq and e2e_bat:
            print(f"\n{'=' * 80}")
            print("  BATCHED DETECTION SPEEDUP SUMMARY")
            print(f"{'=' * 80}")
            print(f"{'Batch':>6} | {'Seq p50':>10} | {'Bat p50':>10} | {'Speedup':>8}")
            print("-" * 50)
            for s, b in zip(e2e_seq, e2e_bat):
                sp = s.latency.p50_ms / b.latency.p50_ms if b.latency.p50_ms > 0 else 0
                print(f"{s.batch_size:>6} | {s.latency.p50_ms:>8.1f}ms | {b.latency.p50_ms:>8.1f}ms | {sp:>7.2f}x")
    elif args.batch_det_model:
        print(f"\n  WARNING: Batch det model not found: {args.batch_det_model}")

    # --- Find optimal batch size ---
    if rec_results:
        best = max(rec_results, key=lambda r: r.faces_per_sec)
        print(f"\n>>> Optimal recognition batch size: {best.batch_size} ({best.faces_per_sec:.1f} faces/sec)")

    # --- GPU memory after benchmarks ---
    if use_gpu:
        mem = get_gpu_memory_mb()
        if mem is not None:
            print(f"\n  GPU memory used after benchmarks: {mem} MB")

    # --- Save results ---
    report = BenchmarkReport(
        timestamp=datetime.now(UTC).isoformat(),
        device=device_str,
        model_name=args.model_name,
        det_size=list(det_size),
        image_path=args.image or "(bundled t1.jpg)",
        image_resolution=[w, h],
        num_warmup=args.warmup,
        num_images=args.num_images,
        faces_per_image=faces_per_image,
        gpu_info=gpu_info,
        single_image=single_results,
        recognition_batch=rec_results,
        end_to_end_batch=e2e_results,
    )

    if not args.no_save:
        path = save_report(report, args.output_dir)
        print(f"\nResults saved to: {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
