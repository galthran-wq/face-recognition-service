import asyncio
import base64
import functools
from collections.abc import Callable
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import Response
from pydantic import BaseModel

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
    DetectBatchRequest,
    DetectBatchResponse,
    DetectBatchResultItem,
    DetectFaceSchema,
    DetectRequest,
    DetectResponse,
    EmbedBatchResponse,
    EmbedBatchResultItem,
    EmbedFaceSchema,
    EmbedResponse,
    ImageRequest,
    LandmarkPoint,
    PoseSchema,
)
from src.services.face_provider.base import DetectedFace, FaceProvider

logger = structlog.get_logger()
router = APIRouter(prefix="/faces", tags=["faces"])

# Bounds requests in flight inside the provider. GPU passes are serialized by
# the provider's internal lock, so with >1 permit the CPU stages of one
# request (decode, letterbox, crops, serialization) overlap another request's
# GPU time instead of idling behind it. FACE_MAX_INFLIGHT=1 restores strictly
# serial behavior.
_inference_sem = asyncio.Semaphore(settings.face_max_inflight)


def _json_response(model: BaseModel) -> Response:
    """Serialize via pydantic-core's Rust path and bypass FastAPI's response
    pipeline (jsonable_encoder + json.dumps), which costs ~19x more on
    embedding-heavy payloads (~70 ms vs ~4 ms for a 66-face batch response).
    The `response_model` in each route decorator keeps the OpenAPI schema.
    """
    return Response(content=model.model_dump_json(), media_type="application/json")


ProviderDep = Annotated[FaceProvider, Depends(get_face_provider)]


def _decode_base64(image_b64: str) -> bytes:
    try:
        return base64.b64decode(image_b64, validate=True)
    except Exception:
        raise AppError(400, "Invalid base64-encoded image")  # noqa: B904


def _bbox_schema(face: DetectedFace) -> BoundingBoxSchema:
    return BoundingBoxSchema(x=face.bbox.x, y=face.bbox.y, width=face.bbox.width, height=face.bbox.height)


def _landmarks_schema(face: DetectedFace) -> list[LandmarkPoint] | None:
    if face.landmarks is None:
        return None
    return [LandmarkPoint(x=x, y=y) for x, y in face.landmarks]


def _pose_schema(face: DetectedFace) -> PoseSchema | None:
    if face.pose is None:
        return None
    return PoseSchema(pitch=face.pose.pitch, yaw=face.pose.yaw, roll=face.pose.roll)


def _to_detect_schema(face: DetectedFace) -> DetectFaceSchema:
    return DetectFaceSchema(
        bbox=_bbox_schema(face),
        det_score=face.det_score,
        landmarks=_landmarks_schema(face),
        pose=_pose_schema(face),
    )


def _to_embed_schema(face: DetectedFace) -> EmbedFaceSchema:
    return EmbedFaceSchema(
        bbox=_bbox_schema(face),
        det_score=face.det_score,
        embedding=face.embedding or [],
        landmarks=_landmarks_schema(face),
    )


def _to_analyze_schema(face: DetectedFace) -> AnalyzeFaceSchema:
    return AnalyzeFaceSchema(
        bbox=_bbox_schema(face),
        det_score=face.det_score,
        embedding=face.embedding or [],
        age=face.age,
        gender=face.gender,
        race=face.race,
        race_probs=face.race_probs,
        landmarks=_landmarks_schema(face),
    )


@router.post("/detect", response_model=DetectResponse)
async def detect(body: DetectRequest, provider: ProviderDep) -> Response:
    image_bytes = _decode_base64(body.image_b64)
    async with _inference_sem:
        faces = await asyncio.to_thread(provider.detect, image_bytes, body.pose)
    return _json_response(DetectResponse(faces=[_to_detect_schema(f) for f in faces], face_count=len(faces)))


@router.post("/embed", response_model=EmbedResponse)
async def embed(body: ImageRequest, provider: ProviderDep) -> Response:
    image_bytes = _decode_base64(body.image_b64)
    async with _inference_sem:
        faces = await asyncio.to_thread(provider.embed, image_bytes)
    return _json_response(EmbedResponse(faces=[_to_embed_schema(f) for f in faces], face_count=len(faces)))


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(body: ImageRequest, provider: ProviderDep) -> Response:
    image_bytes = _decode_base64(body.image_b64)
    async with _inference_sem:
        faces = await asyncio.to_thread(provider.analyze, image_bytes)
    return _json_response(AnalyzeResponse(faces=[_to_analyze_schema(f) for f in faces], face_count=len(faces)))


@router.post("/detect/batch", response_model=DetectBatchResponse)
async def detect_batch(body: DetectBatchRequest, provider: ProviderDep) -> Response:
    detect_fn = functools.partial(provider.detect_batch, include_pose=body.pose)
    results, total_faces = await _process_batch_optimized(body.images, detect_fn, _to_detect_schema)
    return _json_response(
        DetectBatchResponse(
            results=[DetectBatchResultItem(**r) for r in results],
            total_faces=total_faces,
        )
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

    def _decode_all() -> None:
        # base64 of a whole batch costs ~20+ ms — keep it off the event loop
        # so concurrent requests aren't serialized behind it (b64decode
        # releases the GIL).
        for idx, item in enumerate(images):
            try:
                image_bytes = _decode_base64(item.image_b64)
                valid_indices.append(idx)
                valid_bytes.append(image_bytes)
            except AppError as exc:
                results[idx] = {"index": idx, "faces": [], "face_count": 0, "error": exc.detail}

    await asyncio.to_thread(_decode_all)

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


@router.post("/embed/batch", response_model=EmbedBatchResponse)
async def embed_batch(body: BatchRequest, provider: ProviderDep) -> Response:
    results, total_faces = await _process_batch_optimized(body.images, provider.embed_batch, _to_embed_schema)
    return _json_response(
        EmbedBatchResponse(
            results=[EmbedBatchResultItem(**r) for r in results],
            total_faces=total_faces,
        )
    )


@router.post("/analyze/batch", response_model=AnalyzeBatchResponse)
async def analyze_batch(body: BatchRequest, provider: ProviderDep) -> Response:
    results, total_faces = await _process_batch_optimized(body.images, provider.analyze_batch, _to_analyze_schema)
    return _json_response(
        AnalyzeBatchResponse(
            results=[AnalyzeBatchResultItem(**r) for r in results],
            total_faces=total_faces,
        )
    )
