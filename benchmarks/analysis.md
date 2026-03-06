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

### New bottleneck breakdown (embed, per image)

Raw model-level timings from GPU benchmark (RTX 4090, 50 iter, 10 warmup):

| Component | Time | % of pipeline |
|-----------|------|---------------|
| Detection (SCRFD) | ~8ms | 49% |
| Batched recognition (9 faces) | ~3.5ms | 21% |
| Alignment + image decode + overhead | ~5ms | 30% |

Total E2E (detect + align + embed): ~16.4ms per image.

Detection is the largest single component but not as dominant as initially estimated.
The earlier 14.5ms figure was endpoint-level (including image decode + Python overhead),
not raw model time. Recognition is no longer the bottleneck thanks to batching.

### Embedding quality verification

Embeddings from the optimized pipeline are identical to the original `app.get()` output:
- Cosine similarity: **1.000000**
- Max absolute difference: 0.00008 (float32 precision noise)
- Embeddings are L2-normalized to unit vectors

## Round 2: Batched Detection Investigation

Re-exported SCRFD 10G at 640×640 with batch-only dynamic axes (spatial fixed) to avoid
cross-frame contamination. Model validated: zero contamination, bit-for-bit batch parity.

### GPU benchmark (RTX 4090, 50 iterations, 10 warmup, 9-face image)

**Detection only — sequential vs batched SCRFD:**

| Batch | Sequential | Batched | Speedup |
|-------|-----------|---------|---------|
| 1 | 8.1ms | 7.9ms | 1.02x |
| 2 | 16.9ms | 14.3ms | 1.18x |
| 4 | 32.1ms | 25.8ms | 1.24x |
| 8 | 64.8ms | 90.0ms | **0.72x** |
| 16 | 129.5ms | 182.4ms | **0.71x** |
| 32 | 255.9ms | 359.8ms | **0.71x** |

**End-to-end (detect + align + embed) — sequential vs batched detection:**

| Batch | Sequential | Batched | Speedup |
|-------|-----------|---------|---------|
| 1 | 16.4ms | 15.9ms | 1.03x |
| 2 | 30.7ms | 28.3ms | 1.09x |
| 4 | 62.4ms | 55.3ms | 1.13x |
| 8 | 126.3ms | 146.5ms | **0.86x** |
| 16 | 250.4ms | 295.2ms | **0.85x** |
| 32 | 550.2ms | 636.6ms | **0.86x** |

### Conclusion

Batched detection provides a small win at batch 2-4 (up to 1.24x detection, 1.13x E2E) but
is **counterproductive at batch 8+** (0.71x detection, 0.86x E2E). The large input tensor
(`[N, 3, 640, 640]`) hits GPU memory bandwidth limits at higher batch sizes.

On the RTX 4090, sequential detection is already fast at ~8ms/image. The GPU saturates its
compute on a single 640×640 image, so stacking more images into a batch adds memory transfer
cost without proportional compute savings.

**Batched detection is kept as an optional feature** (`FACE_BATCH_DET_MODEL` env var) for
use cases where it helps (CPU inference, weaker GPUs, batch sizes 2-4), but it is **not
recommended for production GPU deployments** with RTX-class hardware.

## Remaining optimization opportunities

### Medium effort

1. **TensorRT conversion.** Expected 2-3x speedup for each model, particularly detection.
   Would bring `detect()` to ~5-7ms and `embed()` to ~8-10ms per image.

2. **FP16 inference.** Enable via ONNX Runtime session options. Expected 1.2-1.5x speedup
   with negligible accuracy loss.

3. **Increase batch API limit.** Current limit is 20 images. With batched inference,
   higher limits are practical without proportional latency increase.

### Larger optimizations

4. **Concurrent detection + recognition.** Detection for image N+1 can overlap with
   recognition for image N using CUDA streams. Would reduce batch latency by ~30%.
