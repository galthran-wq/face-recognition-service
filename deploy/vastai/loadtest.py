"""Async load tester for a single face-recognition-service instance (or the LB).

Hits an endpoint with a fixed image at a configurable concurrency / duration,
then prints RPS, latency percentiles, success rate.

Run from the repo root:

    .venv/bin/python deploy/vastai/loadtest.py \
        --url http://127.0.0.1:11113 --endpoint /faces/detect \
        --concurrency 16 --duration 15

Defaults pick the bundled InsightFace test image (6 faces, ~129 KiB).
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

import httpx


def default_image_path() -> Path:
    # Walk up from this file to find the repo root (pyproject.toml).
    root = Path(__file__).resolve().parent
    while not (root / "pyproject.toml").exists() and root != root.parent:
        root = root.parent
    candidate = root / ".venv/lib/python3.12/site-packages/insightface/data/images/t1.jpg"
    return candidate


async def worker(
    name: int,
    client: httpx.AsyncClient,
    url: str,
    payload: bytes,
    stop_at: float,
    latencies: list[float],
    statuses: dict[int, int],
) -> None:
    while True:
        now = time.perf_counter()
        if now >= stop_at:
            return
        t0 = time.perf_counter()
        try:
            r = await client.post(url, content=payload, headers={"Content-Type": "application/json"})
            elapsed = time.perf_counter() - t0
            statuses[r.status_code] = statuses.get(r.status_code, 0) + 1
            if r.status_code == 200:
                latencies.append(elapsed)
        except Exception:
            statuses[0] = statuses.get(0, 0) + 1


async def run(args: argparse.Namespace) -> None:
    img_path = Path(args.image) if args.image else default_image_path()
    if not img_path.exists():
        print(f"Image not found: {img_path}", file=sys.stderr)
        sys.exit(2)
    img_bytes = img_path.read_bytes()
    payload = json.dumps({"image_b64": base64.b64encode(img_bytes).decode()}).encode()
    url = args.url.rstrip("/") + args.endpoint

    async with httpx.AsyncClient(timeout=600) as client:
        if args.warmup > 0:
            print(f"warmup ({args.warmup} sequential requests)...", flush=True)
            for i in range(args.warmup):
                t0 = time.perf_counter()
                r = await client.post(url, content=payload, headers={"Content-Type": "application/json"})
                print(f"  warmup {i + 1}/{args.warmup}: status={r.status_code} latency={time.perf_counter() - t0:.3f}s")

        latencies: list[float] = []
        statuses: dict[int, int] = {}
        start = time.perf_counter()
        stop_at = start + args.duration
        print(f"\nload: concurrency={args.concurrency} duration={args.duration}s endpoint={args.endpoint}")
        await asyncio.gather(
            *[worker(i, client, url, payload, stop_at, latencies, statuses) for i in range(args.concurrency)]
        )
        wall = time.perf_counter() - start

    n_ok = len(latencies)
    n_total = sum(statuses.values())
    rps = n_ok / wall if wall > 0 else 0
    print("\n== Results ==")
    print(f"wall:           {wall:.2f}s")
    print(f"total requests: {n_total}")
    print(f"successful:     {n_ok}")
    print(f"statuses:       {statuses}")
    print(f"throughput:     {rps:.1f} req/s")
    if latencies:
        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.50)]
        p90 = latencies[int(len(latencies) * 0.90)]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)]
        mean = statistics.mean(latencies)
        print(f"latency mean:   {mean * 1000:.1f}ms")
        print(f"latency p50:    {p50 * 1000:.1f}ms")
        print(f"latency p90:    {p90 * 1000:.1f}ms")
        print(f"latency p95:    {p95 * 1000:.1f}ms")
        print(f"latency p99:    {p99 * 1000:.1f}ms")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="Base URL, e.g. http://127.0.0.1:11113 or https://fr-pool.example.com")
    p.add_argument("--endpoint", default="/faces/detect")
    p.add_argument("--image", default=None, help="Path to test image (default: bundled InsightFace t1.jpg)")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--duration", type=float, default=15)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
