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
        )

    msg = f"Unknown face provider: {name!r}"
    raise ValueError(msg)
