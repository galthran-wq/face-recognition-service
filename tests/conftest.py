from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from src.main import app
from src.services.face_provider.base import BoundingBox, DetectedFace, FaceProvider


class FakeFaceProvider(FaceProvider):
    """Deterministic provider for testing — no model weights required."""

    _FACE = DetectedFace(
        bbox=BoundingBox(x=10.0, y=20.0, width=100.0, height=120.0),
        det_score=0.99,
        embedding=[0.1] * 512,
        age=25.0,
        gender="male",
        race="white",
        race_probs={"white": 0.8, "black": 0.1, "asian": 0.1},
    )

    def load_model(self) -> None:
        self._loaded = True

    def detect(self, image_bytes: bytes) -> list[DetectedFace]:
        return [DetectedFace(bbox=self._FACE.bbox, det_score=self._FACE.det_score)]

    def embed(self, image_bytes: bytes) -> list[DetectedFace]:
        return [
            DetectedFace(bbox=self._FACE.bbox, det_score=self._FACE.det_score, embedding=self._FACE.embedding)
        ]

    def analyze(self, image_bytes: bytes) -> list[DetectedFace]:
        return [self._FACE]

    @property
    def provider_name(self) -> str:
        return "fake"


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    fake = FakeFaceProvider()
    fake.load_model()
    app.state.face_provider = fake
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.state.face_provider = None
