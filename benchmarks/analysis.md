# Benchmark Analysis

## Test Setup

- **GPU:** NVIDIA GeForce RTX 4090 (24 GB VRAM)
- **Driver:** 550.163.01, CUDA 12.4
- **Model:** buffalo_l (ArcFace R50, 512-dim embeddings)
- **Runtime:** onnxruntime-gpu 1.24.1, CUDAExecutionProvider
- **Detection size:** 640x640
- **Test image:** InsightFace bundled t1.jpg (1280x886, 6 faces)
- **Iterations:** 50 (10 warmup)
- **Date:** 2026-03-06

## Single-Image Latency

| Stage | p50 | p95 | p99 | Mean |
|-------|-----|-----|-----|------|
| Detection only | 8.5ms | 9.8ms | 10.3ms | 8.5ms |
| Recognition only (1 face) | 2.7ms | 3.6ms | 4.5ms | 2.8ms |
| Full pipeline (detect + embed 6 faces) | 72.5ms | 94.5ms | 97.8ms | 78.1ms |

The full pipeline is dominated by running 5 models per face sequentially (recognition, 2 landmark models, genderage).
Detection alone is fast at ~8.5ms.

## Recognition Batch Throughput

Batched embedding via `rec_model.get_feat()` — the core scalability test:

| Batch Size | Faces/sec | p50 (ms) | p95 (ms) | Speedup vs batch=1 |
|-----------|-----------|----------|----------|---------------------|
| 1 | 323 | 3.0 | 3.4 | 1.0x |
| 2 | 600 | 3.2 | 4.2 | 1.9x |
| 4 | 1,025 | 3.7 | 5.3 | 3.2x |
| 8 | 2,034 | 3.6 | 4.6 | 6.3x |
| 16 | 2,606 | 6.1 | 6.5 | 8.1x |
| **32** | **2,675** | **11.8** | **12.8** | **8.3x** |
| 64 | 2,564 | 24.7 | 27.9 | 7.9x |

### Observations

- **Optimal batch size: 32** — peak throughput at 2,675 faces/sec
- Near-linear scaling from batch=1 to batch=8 (6.3x speedup)
- Diminishing returns after batch=16, throughput plateaus at ~2,600-2,700 faces/sec
- batch=64 is slightly *slower* than batch=32, with double the latency — GPU is saturated
- batch=8 is the sweet spot for latency-sensitive workloads: 2,034 faces/sec at only 3.6ms p50

## End-to-End Sequential Throughput (Current Service Behavior)

Processing N images through `app.get()` one by one — this is what the service actually does today:

| Batch Size | Faces/sec | p50 (ms) | p95 (ms) |
|-----------|-----------|----------|----------|
| 1 | 77.7 | 75.2 | 92.2 |
| 2 | 77.7 | 147.9 | 182.1 |
| 4 | 75.7 | 306.8 | 384.6 |
| 8 | 77.8 | 603.8 | 740.0 |
| 16 | 79.0 | 1,209.9 | 1,319.8 |
| 32 | 78.0 | 2,466.2 | 2,664.4 |
| 64 | 78.1 | 4,864.2 | 5,492.5 |

### Observations

- Throughput is **flat at ~78 faces/sec** regardless of batch size — no batching benefit
- Latency scales linearly with batch size (sequential processing)
- Each image takes ~75ms for 6 faces (detect + 5 models x 6 faces)

## The Batching Gap

```
Current service (sequential):     78 faces/sec
Batched recognition (batch=32): 2,675 faces/sec
                                ─────────────────
Potential improvement:            ~34x throughput
```

The recognition model can process 34x more faces per second when batched, but the current service
calls `app.get()` per image, which runs each face through each model one at a time.

## Bottleneck Breakdown (per image, 6 faces)

| Component | Time | % of pipeline |
|-----------|------|---------------|
| Detection (SCRFD) | ~8.5ms | 11% |
| Recognition x6 (sequential) | ~17ms | 22% |
| Landmark 3D x6 | ~15ms | 19% |
| Landmark 2D x6 | ~15ms | 19% |
| Genderage x6 | ~12ms | 15% |
| Overhead (alignment, postproc) | ~10ms | 13% |

Detection is cheap. The cost is running 4 models x 6 faces = 24 sequential ONNX calls per image.

## Optimization Results (implemented)

The following optimizations were applied to the service:

1. **Bypassed `app.get()`** — each endpoint now runs only the models it needs:
   - `detect()`: detection only (skips recognition, landmarks, genderage)
   - `embed()`: detection + batched recognition (skips landmarks, genderage)
   - `analyze()`: detection + batched recognition + genderage (skips landmarks)
2. **Batched recognition** — all face crops within an image go through `rec_model.get_feat()`
   in a single call instead of one-by-one.
3. **Cross-image batching** — batch endpoints (`embed_batch`, `analyze_batch`) collect all
   face crops across all images and run a single `get_feat()` call for the entire batch.
4. **Single semaphore acquisition** — batch endpoints acquire the inference semaphore once
   per batch instead of per image.

### Single-image results (optimized vs original)

| Endpoint | Original | Optimized | Speedup |
|----------|----------|-----------|---------|
| `detect()` | 78ms | **14.5ms** | **5.4x** |
| `embed()` | 78ms | **21.5ms** | **3.6x** |
| `analyze()` | 78ms | **32.2ms** | **2.4x** |

### Batch endpoint results (embed_batch, 6 faces per image)

| Batch size | Original | Optimized | Faces/sec | Speedup |
|-----------|----------|-----------|-----------|---------|
| 1 image | 78ms | 27.6ms | 217 | 2.8x |
| 2 images | 155ms | 45.1ms | 266 | 3.4x |
| 4 images | 310ms | 87.6ms | 274 | 3.5x |
| 8 images | 616ms | 157ms | 306 | **3.9x** |
| 16 images | 1,235ms | 362ms | 265 | 3.4x |

### Cross-image batching vs sequential (8 images, 48 faces)

```
Sequential embed() x8:  177.5ms  (270 faces/sec)
embed_batch(8):         156.9ms  (306 faces/sec)
Cross-image speedup:    1.1x
```

Cross-image batching provides a modest 13% improvement over calling `embed()` per image
because **detection is now the bottleneck**. The SCRFD detection model has a fixed batch
size of 1 (~14ms per image), so each image must be detected sequentially regardless. The
recognition batching across images saves some overhead but cannot overcome the detection cost.

### New bottleneck breakdown (embed, per image, 6 faces)

| Component | Time | % of pipeline |
|-----------|------|---------------|
| Detection (SCRFD) | ~14.5ms | 67% |
| Batched recognition (6 faces) | ~5ms | 23% |
| Alignment + postprocessing | ~2ms | 9% |

Detection now dominates at 67% of pipeline time. Recognition is no longer the bottleneck.

### Embedding quality verification

Embeddings from the optimized pipeline are identical to the original `app.get()` output:
- Cosine similarity: **1.000000**
- Max absolute difference: 0.00008 (float32 precision noise)
- Embeddings are L2-normalized to unit vectors

## Round 3: TensorRT EP with FP16 (implemented)

Enabled TensorRT Execution Provider with FP16 via `FACE_USE_TENSORRT=true`. ORT automatically
builds optimized TRT engines on first run (cached for subsequent starts). Also increased
batch limit from 20 to 64.

### TensorRT FP16 vs CUDA EP baseline (RTX 4090, 50 iter, 10 warmup, t1.jpg 6 faces)

| Metric | CUDA EP | TensorRT FP16 | Speedup |
|--------|---------|---------------|---------|
| Detection p50 | 8.4ms | **5.8ms** | **1.45x** |
| Recognition (1 face) p50 | 2.5ms | **1.7ms** | **1.47x** |
| Full pipeline p50 | 80.0ms | **41.7ms** | **1.9x** |

### Recognition batch throughput

| Batch | CUDA EP | TensorRT FP16 | Speedup |
|-------|---------|---------------|---------|
| 1 | 340 faces/sec | **1,130 faces/sec** | **3.3x** |
| 4 | 1,264 faces/sec | **3,275 faces/sec** | **2.6x** |
| 8 | 2,089 faces/sec | **3,602 faces/sec** | **1.7x** |
| 16 | 2,543 faces/sec | **5,371 faces/sec** | **2.1x** |
| 32 | 2,574 faces/sec | **5,880 faces/sec** | **2.3x** |

### Setup

- Requires `FACE_USE_TENSORRT=true` env var
- TRT engine cache path: `FACE_TRT_CACHE_PATH` (default: `/models/trt_cache`)
- First startup builds TRT engines (~30-60s total), subsequent starts load from cache
- Dockerfile.gpu includes `tensorrt-cu12-libs==10.7.0` for CUDA 12.x compatibility
- Falls back to CUDA EP automatically for any unsupported ops

## Round 4: the host is the ceiling (2026-07-26, 4x RTX 4090 / dual Xeon E5-2680 v4)

A day of A/B testing on the production box established that **throughput is capped by the
host, not by the GPUs or the service code**. Every per-process optimization landed on the
same ~205-215 RPS ceiling for `/faces/analyze` via the LB (8 instances, 2/GPU):

| Change | analyze RPS @ C=96 |
|---|---:|
| Baseline (sem=1, no MPS) | 213 |
| Inference semaphore 2 + CUDA MPS | 209 |
| 3 instances/GPU (with MPS, all clients M+C) | 185 (worse) |
| Batched genderage (1 ONNX call vs 6) | 203 |
| `attributes=false` (genderage fully off) | 202 |
| orjson responses (iterencode was 24% of GIL time) | 201 |
| CPU governor schedutil→performance (1.67→2.4 GHz) | ~same |
| `analyze/batch` batch=8/16 (img/s) | 214-222 |

The smoking gun is scaling behavior with dedicated GPUs per instance:

| Active instances | RPS per instance |
|---|---:|
| 1 (solo, own GPU) | **83** |
| 4 (one per GPU) | **57** |
| 8 (two per GPU) | **30** |

Instances slow each other down even on separate GPUs — a host-global shared resource.
Ruled out: Caddy (direct-port tests), the load client (two parallel clients), CUDA context
time-slicing (MPS on/off identical), CPU cores (56 threads, ~78% idle), clock throttling
(governor fix changed nothing), MPS server. Remaining explanation: the memory subsystem —
the box runs dual Broadwell with **BIOS node interleaving (OS sees 1 NUMA node)**, so ~50%
of every memory access and all GPU DMA cross QPI; per-image work (JPEG decode, detector
pre/post, h11 body handling) contends there. GPU util never exceeds ~25% because the host
cannot feed 4x4090s.

GIL profile at saturation (py-spy --gil): GIL held only ~45% — json `iterencode` 24%
(fixed with orjson), pydantic dump ~6%, b64decode ~4%. The rest of request time is
GIL-released C code (ORT, cv2) slowed by the shared memory subsystem.

### What actually remains

1. **BIOS: disable node interleaving (enable NUMA)** + pin instances/memory per socket
   (`numactl --cpunodebind --membind`). Requires console access + reboot. This attacks the
   diagnosed bottleneck directly.
2. **GPU JPEG decode (nvJPEG/DALI).** Kills the decode CPU cost *and* shrinks PCIe/QPI DMA
   ~17x (transfer ~200 KB JPEG instead of ~3.4 MB raw BGR per image). Unusually valuable
   on this host.
3. **Cross-request micro-batching GPU worker** (one per GPU, HTTP processes feed it):
   divides per-image host overhead (Run calls, copies, launches) by the batch size.
   Recognition alone does 5,880 faces/s at batch=32 — the models have ~25x headroom over
   the ~1,300 faces/s the host currently delivers.
4. **Dynamic-batch SCRFD re-export** — prerequisite for (3) to batch detection.
5. Hardware note: a dual-Broadwell host is the wrong CPU:GPU ratio for 4x4090. On a modern
   single-socket host (or with NUMA fixed) the same code should go substantially past the
   current ceiling.

### Also shipped in round 4

- `FACE_INFERENCE_CONCURRENCY` (in-flight inferences per process, default 2).
- `FACE_CUDA_GPU_MEM_LIMIT_GB` (CUDA EP arena cap; default 2 GiB in the launcher) — bounds
  the grow-and-never-shrink arena that previously forced 2/GPU for safety.
- CUDA MPS management in `run_instances.sh` (`MPS_ENABLE`, default on when available) —
  neutral for throughput here but harmless; keeps kernels from different instances truly
  concurrent.
- Batched genderage (one `session.run` per image/batch instead of per face) — verified
  bit-identical age/gender on buffalo_l vs the per-face path.
- `attributes: false` request flag on `/faces/analyze[/batch]` to skip genderage entirely.
- orjson response serialization.

### Older items (still valid)

1. **Concurrent detection + recognition.** Detection for image N+1 can overlap with
   recognition for image N using CUDA streams. Would reduce batch latency by ~30%.

2. **INT8 quantization.** Expected 2-4x over FP32 but requires calibration dataset
   and accuracy validation. Diminishing returns given TRT FP16 already provides major gains
   — and irrelevant while the host, not the GPU, is the bottleneck.
