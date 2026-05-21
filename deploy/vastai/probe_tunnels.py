"""Probe Cloudflare-tunneled endpoints under your domain with realistic body sizes.

Sends three requests to each hostname: /metrics (small), small batch (1 × 64x64),
and big batch (32 × 640x640 ≈ 5 MiB body). Useful to verify the full external
roundtrip (DNS → CF edge → tunnel → instance) handles non-trivial payloads.

Edit DOMAIN + LABELS for your topology.
"""

from __future__ import annotations

import base64
import concurrent.futures as cf
import io
import time
from typing import Any

import httpx
from PIL import Image

DOMAIN = "infected.life"
LABELS = (
    "gpu0-a",
    "gpu0-b",
    "gpu0-c",
    "gpu1-a",
    "gpu1-b",
    "gpu1-c",
    "gpu2-a",
    "gpu2-b",
    "gpu2-c",
)


def make_b64(size: int) -> str:
    img = Image.new("RGB", (size, size))
    for x in range(0, size, 4):
        for y in range(0, size, 4):
            img.putpixel((x, y), ((x * 7) % 256, (y * 11) % 256, 50))
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


SAMPLE_SMALL = make_b64(64)
SAMPLE_LARGE = make_b64(640)


def probe(label: str) -> dict[str, Any]:
    url = f"https://{label}.{DOMAIN}"
    out: dict[str, Any] = {"label": label, "url": url}
    with httpx.Client(timeout=60.0) as c:
        for key, request in [
            ("metrics", lambda: c.get(f"{url}/metrics")),
            ("small", lambda: c.post(f"{url}/faces/analyze/batch", json={"images": [{"image_b64": SAMPLE_SMALL}]})),
            (
                "big",
                lambda: c.post(
                    f"{url}/faces/analyze/batch", json={"images": [{"image_b64": SAMPLE_LARGE} for _ in range(32)]}
                ),
            ),
        ]:
            t0 = time.time()
            try:
                r = request()
                out[key] = (r.status_code, int((time.time() - t0) * 1000))
            except Exception as e:
                out[key] = (None, int((time.time() - t0) * 1000), type(e).__name__)
    return out


def _fmt(row: dict[str, Any], key: str) -> str:
    v = row.get(key, ("?",))
    return f"{v[0]} {v[1]}ms" if v[0] else f"FAIL ({v[2] if len(v) > 2 else '?'})"


def main() -> None:
    rows: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=len(LABELS)) as pool:
        futures = [pool.submit(probe, label) for label in LABELS]
        for fut in cf.as_completed(futures):
            rows.append(fut.result())
    rows.sort(key=lambda r: r["label"])

    print(f"{'label':<8} {'metrics':<14} {'small(64x64)':<14} {'big(32 x 640x640)':<24}")
    print("-" * 70)
    for r in rows:
        print(f"{r['label']:<8} {_fmt(r, 'metrics'):<14} {_fmt(r, 'small'):<14} {_fmt(r, 'big'):<24}")

    big_ok = [r for r in rows if r.get("big", (None,))[0] == 200]
    print(f"\n{len(big_ok)}/{len(rows)} endpoints handle batch=32 x 640x640")
    if big_ok:
        ts = sorted(r["big"][1] for r in big_ok)
        print(f"big-batch latency: min={ts[0]}ms p50={ts[len(ts) // 2]}ms max={ts[-1]}ms")


if __name__ == "__main__":
    main()
