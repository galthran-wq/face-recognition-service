import asyncio
import base64
from collections.abc import Callable
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends

from src.config import settings
from src.core.exceptions import AppError
from src.dependencies import get_face_provider
from src.schemas.faces import (
    AnalyzeBatchResponse,
    AnalyzeBatchResultItem,
    AnalyzeFaceSchema,
    AnalyzeResponse,
    BatchRequest,
    BoundingBoxSchema,
    DetectBatchResponse,
    DetectBatchResultItem,
    DetectFaceSchema,
    DetectResponse,
    EmbedBatchResponse,
    EmbedBatchResultItem,
    EmbedFaceSchema,
    EmbedResponse,
    ImageRequest,
)
from src.services.face_provider.base import DetectedFace, FaceProvider

logger = structlog.get_logger()
router = APIRouter(prefix="/faces", tags=["faces"])

_inference_sem = asyncio.Semaphore(1)

ProviderDep = Annotated[FaceProvider, Depends(get_face_provider)]


def _decode_base64(image_b64: str) -> bytes:
    try:
        return base64.b64decode(image_b64, validate=True)
    except Exception:
        raise AppError(400, "Invalid base64-encoded image")  # noqa: B904


def _bbox_schema(face: DetectedFace) -> BoundingBoxSchema:
    return BoundingBoxSchema(x=face.bbox.x, y=face.bbox.y, width=face.bbox.width, height=face.bbox.height)


def _to_detect_schema(face: DetectedFace) -> DetectFaceSchema:
    return DetectFaceSchema(bbox=_bbox_schema(face), det_score=face.det_score)


def _to_embed_schema(face: DetectedFace) -> EmbedFaceSchema:
    return EmbedFaceSchema(bbox=_bbox_schema(face), det_score=face.det_score, embedding=face.embedding or [])


def _to_analyze_schema(face: DetectedFace) -> AnalyzeFaceSchema:
    return AnalyzeFaceSchema(
        bbox=_bbox_schema(face),
        det_score=face.det_score,
        embedding=face.embedding or [],
        age=face.age,
        gender=face.gender,
        race=face.race,
        race_probs=face.race_probs,
    )


@router.post("/detect")
async def detect(body: ImageRequest, provider: ProviderDep) -> DetectResponse:
    image_bytes = _decode_base64(body.image_b64)
    async with _inference_sem:
        faces = await asyncio.to_thread(provider.detect, image_bytes)
    return DetectResponse(faces=[_to_detect_schema(f) for f in faces], face_count=len(faces))


@router.post("/embed")
async def embed(body: ImageRequest, provider: ProviderDep) -> EmbedResponse:
    image_bytes = _decode_base64(body.image_b64)
    async with _inference_sem:
        faces = await asyncio.to_thread(provider.embed, image_bytes)
    return EmbedResponse(faces=[_to_embed_schema(f) for f in faces], face_count=len(faces))


@router.post("/analyze")
async def analyze(body: ImageRequest, provider: ProviderDep) -> AnalyzeResponse:
    image_bytes = _decode_base64(body.image_b64)
    async with _inference_sem:
        faces = await asyncio.to_thread(provider.analyze, image_bytes)
    return AnalyzeResponse(faces=[_to_analyze_schema(f) for f in faces], face_count=len(faces))


async def _process_batch[T](
    images: list[ImageRequest],
    method: Callable[[bytes], list[DetectedFace]],
    to_schema: Callable[[DetectedFace], T],
) -> tuple[list[dict[str, object]], int]:
    if len(images) > settings.face_max_batch_size:
        raise AppError(400, f"Batch size {len(images)} exceeds maximum of {settings.face_max_batch_size}")

    decoded: list[tuple[int, bytes | None, str | None]] = []
    for idx, item in enumerate(images):
        try:
            image_bytes = _decode_base64(item.image_b64)
            decoded.append((idx, image_bytes, None))
        except AppError as exc:
            decoded.append((idx, None, exc.detail))

    results: list[dict[str, object]] = []
    total_faces = 0

    async with _inference_sem:
        for idx, image_bytes, error in decoded:
            if error is not None:
                results.append({"index": idx, "faces": [], "face_count": 0, "error": error})
                continue
            try:
                faces = await asyncio.to_thread(method, image_bytes)
                face_schemas = [to_schema(f) for f in faces]
                results.append({"index": idx, "faces": face_schemas, "face_count": len(faces), "error": None})
                total_faces += len(faces)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Batch image processing failed", index=idx)
                error_message = str(exc) or "Processing failed"
                results.append({"index": idx, "faces": [], "face_count": 0, "error": error_message})

    return results, total_faces


@router.post("/detect/batch")
async def detect_batch(body: BatchRequest, provider: ProviderDep) -> DetectBatchResponse:
    results, total_faces = await _process_batch(body.images, provider.detect, _to_detect_schema)
    return DetectBatchResponse(
        results=[DetectBatchResultItem(**r) for r in results],
        total_faces=total_faces,
    )


async def _process_batch_optimized[T](
    images: list[ImageRequest],
    batch_method: Callable[[list[bytes]], list[list[DetectedFace]]],
    to_schema: Callable[[DetectedFace], T],
) -> tuple[list[dict[str, object]], int]:
    if len(images) > settings.face_max_batch_size:
        raise AppError(400, f"Batch size {len(images)} exceeds maximum of {settings.face_max_batch_size}")

    valid_indices: list[int] = []
    valid_bytes: list[bytes] = []
    results: list[dict[str, object]] = [{}] * len(images)

    for idx, item in enumerate(images):
        try:
            image_bytes = _decode_base64(item.image_b64)
            valid_indices.append(idx)
            valid_bytes.append(image_bytes)
        except AppError as exc:
            results[idx] = {"index": idx, "faces": [], "face_count": 0, "error": exc.detail}

    total_faces = 0

    if valid_bytes:
        async with _inference_sem:
            try:
                all_faces = await asyncio.to_thread(batch_method, valid_bytes)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Batch processing failed")
                error_message = str(exc) or "Processing failed"
                for idx in valid_indices:
                    results[idx] = {"index": idx, "faces": [], "face_count": 0, "error": error_message}
                return results, 0

        for i, idx in enumerate(valid_indices):
            faces = all_faces[i]
            face_schemas = [to_schema(f) for f in faces]
            results[idx] = {"index": idx, "faces": face_schemas, "face_count": len(faces), "error": None}
            total_faces += len(faces)

    return results, total_faces


@router.post("/embed/batch")
async def embed_batch(body: BatchRequest, provider: ProviderDep) -> EmbedBatchResponse:
    results, total_faces = await _process_batch_optimized(body.images, provider.embed_batch, _to_embed_schema)
    return EmbedBatchResponse(
        results=[EmbedBatchResultItem(**r) for r in results],
        total_faces=total_faces,
    )


@router.post("/analyze/batch")
async def analyze_batch(body: BatchRequest, provider: ProviderDep) -> AnalyzeBatchResponse:
    results, total_faces = await _process_batch_optimized(body.images, provider.analyze_batch, _to_analyze_schema)
    return AnalyzeBatchResponse(
        results=[AnalyzeBatchResultItem(**r) for r in results],
        total_faces=total_faces,
    )
