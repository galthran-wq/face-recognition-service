from fastapi import Request

from src.core.exceptions import AppError
from src.services.face_provider.base import FaceProvider


def get_face_provider(request: Request) -> FaceProvider:
    provider: FaceProvider | None = getattr(request.app.state, "face_provider", None)
    if provider is None:
        raise AppError(503, "Face provider not initialized")
    return provider
