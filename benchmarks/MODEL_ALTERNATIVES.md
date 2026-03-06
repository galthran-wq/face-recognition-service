# Face Recognition Model Alternatives & API Research

Research conducted March 2026. Comparing alternatives to the current InsightFace buffalo_l (ArcFace R50, 512-dim embeddings).

## SOTA Open-Source Models

| Model | LFW Accuracy | Key Advantage | Params | Open Source |
|-------|-------------|---------------|--------|-------------|
| **AdaFace** (WebFace42M) | 99.29% | Quality-adaptive margins; robust on low-quality images | ~65M | [Yes](https://github.com/mk-minchul/AdaFace) |
| **ElasticFace** | Beats ArcFace 7/9 benchmarks | Elastic penalty margins for flexible class separability | ~65M | [Yes](https://github.com/fdbtrs/ElasticFace) |
| **TransFace / TransFace++** | Competitive | ViT-based; TransFace++ works on raw image bytes (privacy) | ~85M | [Yes](https://arxiv.org/abs/2308.10133) |
| **MobileFaceNet** (SAM-optimized) | 99.55% | Only 2.1M params; ideal for edge/mobile deployment | 2.1M | Yes |
| **Arc2Face** | 98.81% (0.5M train) | Identity-conditioned generation + recognition | ~65M | Yes |
| buffalo_l (current) | ~99.2% | Mature, well-tested, InsightFace ecosystem | ~65M | Yes |

### Recommendation

**AdaFace** is the strongest upgrade path — same inference cost as buffalo_l but better accuracy, especially on real-world degraded images (blur, low-res, poor lighting). Can be exported to ONNX and used as a drop-in replacement for the recognition model.

**MobileFaceNet** is worth evaluating if latency/cost is a priority over marginal accuracy gains.

## Usage-Based Face Recognition APIs

| API | Price per 1K calls | Free Tier | Embedding Access | Notes |
|-----|-------------------|-----------|-----------------|-------|
| **AWS Rekognition** | $0.75-1.00 | None | No raw embeddings | Mature; deep AWS integration; face collections for search |
| **Azure Face API** | $0.20-1.00 (volume) | 30K/mo | No raw embeddings | Requires approval for identification; best value at scale |
| **Google Cloud Vision** | $0.60-1.50 | 1K/mo | No raw embeddings | Better for attributes than identification |
| **Face++** (Megvii) | ~$0.011/call | Yes, no CC | Yes (1024-dim) | Widest SDK support; good value at small-to-medium scale |
| **Kairos** | From $19/mo | 14-day trial | Yes | Focus on ethical AI, demographic fairness |

### Key Insight

Most cloud APIs (AWS, Azure, Google) do **not** expose raw embedding vectors — they only support compare/search operations within their platform. This means you can't use them as a drop-in replacement if you need embeddings for your own vector DB.

**Face++** is the exception — it returns raw embeddings and is the cheapest option for prototyping.

For production at scale, **self-hosted models** (buffalo_l, AdaFace) are significantly cheaper than any API once you amortize GPU costs over 2-3 months.

## GPU Throughput Reference

### Expected Performance (from community benchmarks)

| GPU | Recognition batch=1 | Recognition batch=64 | Full pipeline (det+emb) |
|-----|--------------------|--------------------|------------------------|
| T4 (16GB) | ~200 faces/sec | ~800-1200 faces/sec | ~50-80 images/sec |
| A10 (24GB) | ~300 faces/sec | ~1200-1800 faces/sec | ~80-120 images/sec |
| RTX 2080 Ti | ~250 faces/sec | ~2000 faces/sec | ~70-100 images/sec |
| A100 (40GB) | ~400 faces/sec | ~2500-3500 faces/sec | ~120-180 images/sec |
| L4 (24GB) | ~250 faces/sec | ~1000-1500 faces/sec | ~60-90 images/sec |

*Full pipeline numbers assume 1 face per image. Multi-face images increase recognition time linearly.*

### Batch Size Guidelines

- **batch=1**: Maximum latency efficiency, poor GPU utilization
- **batch=4-8**: Good balance for real-time applications (<100ms latency)
- **batch=16-32**: Sweet spot for most deployments (70-85% of max throughput)
- **batch=64-128**: Near-maximum throughput; latency increases to 500ms-2s
- **>128**: Diminishing returns; may hit VRAM limits on smaller GPUs

### Optimization Opportunities

| Optimization | Expected Speedup | Effort |
|-------------|-----------------|--------|
| **TensorRT conversion** | 2-3x | Medium (export ONNX → TensorRT, hardware-specific) |
| **FP16 inference** | 1.2-1.5x | Low (ONNX Runtime flag) |
| **INT8 quantization** | 1.5-2x | Medium (needs calibration dataset) |
| **Batched recognition** | 2-5x over sequential | Low (already supported by get_feat()) |
| **Pinned GPU memory** | 16-30% | Low |

### Cost Comparison (cloud GPU rental)

| GPU | $/hour (spot) | Faces/$ at batch=32 |
|-----|--------------|-------------------|
| T4 | ~$0.35 | ~3.4M faces/$ |
| L4 | ~$0.50 | ~3.0M faces/$ |
| A10 | ~$0.60 | ~3.0M faces/$ |
| A100 | ~$3.00 | ~1.2M faces/$ |

*T4 offers the best cost-efficiency for pure throughput. A10/L4 are better when you also need low latency.*

## NIST FRTE 2026 Leaders (for reference)

Top commercial vendors in NIST Face Recognition Technology Evaluation:
1. **NEC** — 0.07% error rate on 12M identity database; #1 in aging tests
2. **IDEMIA** — Highest accuracy + fairness scores
3. **ROC** — #1 American vendor; best profile-image robustness

These are enterprise solutions (custom pricing, integration required) — not comparable to self-hosted OSS models for typical use cases.
