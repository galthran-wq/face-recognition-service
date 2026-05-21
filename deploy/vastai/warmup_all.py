"""Warm up TRT engines on every running instance in parallel.

Reads the public_urls.txt produced by run_instances.sh (or hardcoded ports if
that's missing), then fires /faces/detect, /faces/embed, /faces/analyze
sequentially on each instance — concurrently across instances.

First cold endpoint hit on an instance triggers TensorRT engine compilation
(~30–60 s the very first time on a given GPU; ~5–10 s thereafter when the
on-disk TRT cache is reused). Subsequent endpoints reuse the same loaded
session — milliseconds.

Run from the repo root:

    .venv/bin/python deploy/vastai/warmup_all.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import sys
import time
from pathlib import Path

import httpx

ENDPOINTS = ("/faces/detect", "/faces/embed", "/faces/analyze")


def repo_root() -> Path:
    root = Path(__file__).resolve().parent
    while not (root / "pyproject.toml").exists() and root != root.parent:
        root = root.parent
    return root


def discover_ports() -> list[tuple[str, int]]:
    """Read run/public_urls.txt; fall back to default 9-instance topology."""
    urls_file = Path(__file__).resolve().parent / "run" / "public_urls.txt"
    if urls_file.exists():
        ports: list[tuple[str, int]] = []
        for line in urls_file.read_text().splitlines():
            m = re.search(r"^(\S+)\s+->\s+http[s]?://127\.0\.0\.1:(\d+)", line)
            if m:
                ports.append((m.group(1), int(m.group(2))))
            else:
                m2 = re.match(r"^(gpu\d-\w)\s", line)
                if m2:
                    # CF URL line — look in companion local map via pidfiles isn't ideal;
                    # rely on running uvicorn binding to 11100-range. Best effort: skip.
                    pass
        if ports:
            return ports
    # Fallback default (matches launcher's default INSTANCES array)
    return [
        ("gpu0-a", 11113),
        ("gpu0-b", 11114),
        ("gpu0-c", 11115),
        ("gpu1-a", 11119),
        ("gpu1-b", 11120),
        ("gpu1-c", 11121),
        ("gpu2-a", 11125),
        ("gpu2-b", 11126),
        ("gpu2-c", 11127),
    ]


async def wait_healthy(client: httpx.AsyncClient, port: int, name: str) -> None:
    for _ in range(120):
        try:
            r = await client.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        await asyncio.sleep(1)
    raise TimeoutError(f"{name} :{port} not healthy after 120s")


async def warm_instance(
    client: httpx.AsyncClient, name: str, port: int, payload: bytes
) -> tuple[str, dict[str, float]]:
    await wait_healthy(client, port, name)
    print(f"  [{name:6s} :{port}] healthy, warming...", flush=True)
    timings: dict[str, float] = {}
    for ep in ENDPOINTS:
        t0 = time.perf_counter()
        try:
            r = await client.post(
                f"http://127.0.0.1:{port}{ep}",
                content=payload,
                headers={"Content-Type": "application/json"},
                timeout=300,
            )
            elapsed = time.perf_counter() - t0
            timings[ep] = elapsed
            print(f"  [{name:6s}] {ep:18s} cold={elapsed:6.1f}s status={r.status_code}", flush=True)
        except Exception as e:
            timings[ep] = -1
            print(f"  [{name:6s}] {ep} FAILED: {e}", flush=True)
    return name, timings


async def main() -> None:
    ports = discover_ports()
    img_path = repo_root() / ".venv/lib/python3.12/site-packages/insightface/data/images/t1.jpg"
    if not img_path.exists():
        print(f"Test image not found: {img_path}", file=sys.stderr)
        sys.exit(2)
    payload = json.dumps({"image_b64": base64.b64encode(img_path.read_bytes()).decode()}).encode()

    print(f"warming up {len(ports)} instances")
    print("each cold endpoint builds/loads TRT engines (~5–60 s); 3 endpoints per instance")
    print("instances on the same GPU compile serially (GPU contention), across GPUs in parallel\n")

    start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[warm_instance(client, n, p, payload) for n, p in ports])
    wall = time.perf_counter() - start

    print(f"\n== Warmup complete in {wall:.1f}s ==")
    for name, timings in results:
        det = timings.get("/faces/detect", -1)
        emb = timings.get("/faces/embed", -1)
        ana = timings.get("/faces/analyze", -1)
        total = sum(t for t in timings.values() if t > 0)
        print(f"  {name:6s} detect={det:5.1f}s embed={emb:5.1f}s analyze={ana:5.1f}s total={total:5.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
