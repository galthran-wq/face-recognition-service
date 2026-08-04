from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Settings
    from src.services.face_provider.base import FaceProvider


def create_provider(settings: Settings) -> FaceProvider:
    name = settings.face_provider.lower()

    if name == "insightface":
        from src.services.face_provider.insightface import InsightFaceProvider

        return InsightFaceProvider(
            use_gpu=settings.face_use_gpu,
            ctx_id=settings.face_ctx_id,
            det_size=settings.face_det_size,
            model_name=settings.face_model_name,
            model_dir=settings.face_model_dir,
            use_tensorrt=settings.face_use_tensorrt,
            trt_cache_path=settings.face_trt_cache_path,
            trt_max_batch=settings.face_trt_max_batch,
            trt_opt_batch=settings.face_trt_opt_batch,
            det_dynamic_batch=settings.face_det_dynamic_batch,
            det_trt_max_batch=settings.face_det_trt_max_batch,
            det_trt_opt_batch=settings.face_det_trt_opt_batch,
            pad_fallback_border_px=settings.face_pad_fallback_border_px,
            pad_fallback_fill=settings.face_pad_fallback_fill,
        )

    msg = f"Unknown face provider: {name!r}"
    raise ValueError(msg)
