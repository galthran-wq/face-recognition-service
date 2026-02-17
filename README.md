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
| `FACE_MAX_BATCH_SIZE` | `20` | Max images per batch request |

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
