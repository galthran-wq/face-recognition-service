# Face Recognition Benchmarks

## Quick Start

```bash
# CPU benchmark with defaults (bundled test image, 50 iterations)
uv run python benchmarks/benchmark.py --cpu

# GPU benchmark
uv run python benchmarks/benchmark.py --gpu

# Custom batch sizes and iterations
uv run python benchmarks/benchmark.py --gpu --batch-sizes 1,4,8,16,32,64,128 --num-images 100

# Use your own test image
uv run python benchmarks/benchmark.py --gpu --image path/to/photo.jpg

# Fast run (skip slow end-to-end batch test)
uv run python benchmarks/benchmark.py --gpu --skip-e2e

# Skip single-image latency tests
uv run python benchmarks/benchmark.py --gpu --skip-single
```

## What It Measures

1. **Single-image latency** — p50/p95/p99 for detection, recognition, and full pipeline
2. **Recognition batch throughput** — faces/sec at different batch sizes using batched `get_feat()`
3. **End-to-end batch throughput** — images/sec processing N images sequentially (current service behavior)

## Output

- Console table with throughput and latency per batch size
- JSON results in `benchmarks/results/` (use `--no-save` to skip)
- Optimal batch size recommendation

## GPU Memory Monitoring

Install `pynvml` for GPU memory reporting:

```bash
uv pip install pynvml
```

## Model Alternatives

See [MODEL_ALTERNATIVES.md](MODEL_ALTERNATIVES.md) for research on alternative models and APIs.
