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

5. **Detection model with dynamic batch.** Replace SCRFD with a detection model that supports
   batch>1 input. This would unlock true batch-level parallelism for the detection stage.
