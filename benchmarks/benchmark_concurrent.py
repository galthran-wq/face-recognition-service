#!/usr/bin/env python3
"""Concurrent-load benchmark for the CPU/GPU pipeline (FACE_MAX_INFLIGHT).

Runs the real ASGI app in-process with a GPU provider and fires K concurrent
`analyze/batch` clients at it. With FACE_MAX_INFLIGHT=1 requests serialize
end-to-end (the GPU idles through every CPU stage); with >1 the CPU stages of
one request overlap another's GPU passes.

Usage:
    FACE_MAX_INFLIGHT=1 python3 benchmarks/benchmark_concurrent.py --gpu --tensorrt ...
    FACE_MAX_INFLIGHT=3 python3 benchmarks/benchmark_concurrent.py --gpu --tensorrt ...

(the setting is read at app import, hence the env var rather than a flag)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_payload(images_glob, batch_size, noface_every):
    import glob as globmod

    rng = np.random.default_rng(0)
    face_paths = sorted(globmod.glob(str(Path(images_glob).expanduser()))) if images_glob else []
    face_img = cv2.imread(face_paths[0]) if face_paths else None
    noise = rng.integers(0, 255, size=(720, 1280, 3), dtype=np.uint8)
    items = []
    for i in range(batch_size):
        img = noise if (face_img is None or i % noface_every == noface_every - 1) else face_img
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        items.append({"image_b64": base64.b64encode(buf.tobytes()).decode()})
    return json.dumps({"images": items}).encode()


async def worker(client, body, latencies, deadline):
    while time.perf_counter() < deadline:
        t = time.perf_counter()
        r = await client.post(
            "/faces/analyze/batch", content=body, headers={"content-type": "application/json"}, timeout=120
        )
        assert r.status_code == 200, r.text
        latencies.append(time.perf_counter() - t)


async def run(args, body):
    import httpx

    from src.main import app
    from src.services.face_provider.insightface import InsightFaceProvider

    provider = InsightFaceProvider(
        use_gpu=args.gpu,
        model_dir=args.model_dir,
        use_tensorrt=args.tensorrt,
        trt_cache_path=args.trt_cache,
    )
    provider.load_model()
    app.state.face_provider = provider

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        # Warmup (engines, pools, first-batch caches)
        for _ in range(2):
            r = await client.post(
                "/faces/analyze/batch", content=body, headers={"content-type": "application/json"}, timeout=300
            )
            assert r.status_code == 200, r.text

        latencies: list[float] = []
        start = time.perf_counter()
        deadline = start + args.duration
        await asyncio.gather(*(worker(client, body, latencies, deadline) for _ in range(args.clients)))
        elapsed = time.perf_counter() - start

    n = len(latencies)
    lat_sorted = sorted(latencies)
    from src.config import settings

    print(f"\nFACE_MAX_INFLIGHT={settings.face_max_inflight}  clients={args.clients}  batch={args.batch_size}")
    print(f"requests: {n} in {elapsed:.1f}s  ->  {n / elapsed:.2f} req/s,  {n * args.batch_size / elapsed:.1f} img/s")
    print(
        f"latency: p50 {statistics.median(latencies) * 1000:.0f} ms   "
        f"p95 {lat_sorted[int(n * 0.95)] * 1000:.0f} ms   max {lat_sorted[-1] * 1000:.0f} ms"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", help="glob of face images (mixed with synthetic no-face noise)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--duration", type=float, default=30.0, help="seconds of sustained load")
    ap.add_argument("--noface-every", type=int, default=3)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--tensorrt", action="store_true")
    ap.add_argument("--trt-cache", default="/models/trt_cache")
    ap.add_argument("--model-dir", default="~/.insightface")
    args = ap.parse_args()

    body = build_payload(args.images, args.batch_size, args.noface_every)
    print(f"request body: {len(body) / 1024 / 1024:.1f} MiB x {args.clients} concurrent clients")
    asyncio.run(run(args, body))


if __name__ == "__main__":
    main()
