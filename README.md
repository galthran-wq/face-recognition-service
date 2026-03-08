# Face Recognition Service

[![CI](https://github.com/galthran-wq/face-recognition-service/actions/workflows/ci.yml/badge.svg)](https://github.com/galthran-wq/face-recognition-service/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/galthran-wq/face-recognition-service/badge.svg?branch=master)](https://coveralls.io/github/galthran-wq/face-recognition-service?branch=master)
![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/type_checker-mypy-blue)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

FastAPI microservice for face detection, embedding extraction, and demographic analysis. Powered by [InsightFace](https://github.com/deepinsight/insightface) with pluggable provider abstraction and optional GPU acceleration.

## Features

- **Face detection** — bounding boxes and confidence scores
- **Embedding extraction** — 512-d face vectors for recognition/search
- **Demographic analysis** — age, gender, and race estimation
- **Batch endpoints** — process multiple images in a single request
- **GPU support** — dedicated Dockerfile with CUDA/cuDNN for GPU inference
- **Provider abstraction** — swap face analysis backends without changing the API

## Stack

- **FastAPI** — async web framework
- **InsightFace** — face detection and analysis (buffalo_l model)
- **ONNX Runtime** — inference engine (CPU or GPU)
- **uv** — package manager
- **Pydantic v2** — validation and settings
- **structlog** — structured logging
- **Prometheus** — metrics via prometheus-fastapi-instrumentator
- **pytest + httpx** — testing
- **ruff** — linting and formatting
- **mypy** — type checking
- **Docker** — containerization (CPU and GPU images)

## Quick Start

```bash
make install
make run
# Server starts at http://localhost:8000
```

## API Endpoints

All face endpoints accept a JSON body with a base64-encoded image:

| Endpoint | Description |
|---|---|
| `POST /faces/detect` | Detect faces — returns bounding boxes and scores |
| `POST /faces/embed` | Detect + extract 512-d embedding vectors |
| `POST /faces/analyze` | Detect + embed + demographic attributes |
| `POST /faces/detect/batch` | Batch detection for multiple images |
| `POST /faces/embed/batch` | Batch embedding for multiple images |
| `POST /faces/analyze/batch` | Batch analysis for multiple images |
| `GET /health` | Health check |

## Commands

| Command | Description |
|---|---|
| `make install` | Install dependencies |
| `make run` | Run dev server with hot reload |
| `make test` | Run tests with coverage |
| `make test-gpu` | Run GPU-only tests |
| `make lint` | Run ruff + mypy |
| `make format` | Auto-format code |
| `make pre-commit` | Install pre-commit hooks |
| `make docker-build` | Build CPU Docker image |
| `make docker-run` | Run CPU Docker container |
| `make docker-build-gpu` | Build GPU Docker image (CUDA 12.4) |
| `make docker-run-gpu` | Run GPU Docker container |

## Configuration

Set via environment variables or `.env` file (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `FACE_PROVIDER` | `insightface` | Face analysis backend |
| `FACE_USE_GPU` | `false` | Enable GPU inference |
| `FACE_MODEL_NAME` | `buffalo_l` | InsightFace model pack |
| `FACE_MODEL_DIR` | `~/.insightface` | Directory for downloaded model files |
| `FACE_DET_SIZE` | `640,640` | Detection input resolution |
| `FACE_MAX_BATCH_SIZE` | `64` | Max images per batch request |
| `FACE_USE_TENSORRT` | `false` | Enable TensorRT EP with FP16 (GPU only) |
| `FACE_TRT_CACHE_PATH` | `/models/trt_cache` | TRT engine cache directory |

## GPU Performance

The default InsightFace pipeline (`app.get()`) runs all 5 models (detection, recognition, 2 landmark models, genderage) per face sequentially, regardless of what the endpoint actually needs. On an RTX 4090, a single image takes ~78ms through this pipeline no matter which endpoint you call.

Three optimizations reduce this to **5.8-22ms** depending on the endpoint:

1. **Selective model execution.** Each endpoint runs only the models it needs. `detect` runs SCRFD only (skips recognition, landmarks, genderage). `embed` runs detection + recognition (skips landmarks, genderage). This alone gives a **3.6-5.4x** speedup per request.

2. **Batched recognition.** Instead of embedding faces one-by-one, all face crops (within and across images) go through the recognition model in a single batched call. The recognition model scales near-linearly up to batch=32 on RTX 4090.

3. **TensorRT FP16 inference.** Enabling TensorRT Execution Provider with FP16 precision gives another **1.5-2.3x** on top. Embedding quality is unaffected (cosine similarity 0.9998+ vs FP32).

We also tested batched detection (running SCRFD on multiple images at once), but it provided no benefit on GPU — SCRFD is already efficient at ~6ms per image.

### Benchmarks (RTX 4090, buffalo_l, 640x640 detection)

Per-image latency (p50, lower is better):

| Endpoint | Original | Optimized (CUDA EP) | + TensorRT FP16 |
|----------|----------|---------------------|-----------------|
| `detect` | 78ms | 8.5ms | **5.8ms** |
| `embed` | 78ms | 21.5ms | **~14ms** |
| `analyze` | 78ms | 32.2ms | **~22ms** |

Recognition model throughput (isolated, not end-to-end):

| Batch size | CUDA EP (FP32) | TensorRT FP16 |
|-----------|----------------|---------------|
| 1 | 323 faces/sec | 1,130 faces/sec |
| 8 | 2,089 faces/sec | 3,602 faces/sec |
| 32 | 2,574 faces/sec | 5,880 faces/sec |

Latency numbers are per image regardless of face count — detection cost is fixed, recognition cost scales with the number of detected faces. Benchmarked on a 1280x886 image with 6 faces (InsightFace bundled t1.jpg). See [`benchmarks/analysis.md`](benchmarks/analysis.md) for full methodology and breakdown.

### Recommended GPU configuration

```bash
docker run --gpus all -p 8000:8000 \
  -v ~/.insightface:/models \
  -e FACE_USE_TENSORRT=true \
  -e FACE_TRT_CACHE_PATH=/models/trt_cache \
  face-recognition-service:gpu
```

First startup builds TensorRT engines (~30-60s), cached for subsequent runs. Requires NVIDIA driver 550+ with CUDA 12.x.

## Model Caching

By default, InsightFace downloads models on first startup. Inside Docker the model directory is `/models`. Mount a host volume to persist models across container restarts:

```bash
# CPU
docker run -p 8000:8000 -v ~/.insightface:/models face-recognition-service

# GPU
docker run --gpus all -p 8000:8000 -v ~/.insightface:/models face-recognition-service:gpu
```

Models are downloaded once into the mounted directory and reused on subsequent runs.

## GPU Support

Build and run the GPU image (requires NVIDIA Container Toolkit):

```bash
make docker-build-gpu
make docker-run-gpu
```

Override CUDA version if needed:

```bash
make docker-build-gpu CUDA_TAG=11.8.0-cudnn8-runtime-ubuntu22.04 ONNXRT_VERSION=1.17.1
```

## Project Structure

```
src/
├── main.py           — app factory, structlog config, Prometheus setup
├── config.py         — pydantic-settings based configuration
├── dependencies.py   — FastAPI dependency injection providers
├── api/
│   ├── router.py     — aggregated API router
│   └── endpoints/    — route handlers (faces.py)
├── schemas/          — Pydantic request/response models
├── services/
│   └── face_provider/
│       ├── base.py       — FaceProvider ABC + data classes
│       ├── insightface.py — InsightFace implementation
│       └── registry.py   — provider lookup by name
└── core/
    ├── exceptions.py — custom exceptions + handlers
    └── middleware.py  — CORS, request logging, request ID
```
