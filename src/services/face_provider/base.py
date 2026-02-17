from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class DetectedFace:
    bbox: BoundingBox
    det_score: float
    embedding: list[float] | None = None
    age: float | None = None
    gender: str | None = None
    race: str | None = None
    race_probs: dict[str, float] | None = field(default=None)


class FaceProvider(ABC):
    _loaded: bool = False

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights. Called once at startup."""

    @abstractmethod
    def detect(self, image_bytes: bytes) -> list[DetectedFace]:
        """Return faces with bounding boxes and detection scores only."""

    @abstractmethod
    def embed(self, image_bytes: bytes) -> list[DetectedFace]:
        """Return faces with bounding boxes + embedding vectors."""

    @abstractmethod
    def analyze(self, image_bytes: bytes) -> list[DetectedFace]:
        """Return faces with bounding boxes + embeddings + demographics."""

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    def is_loaded(self) -> bool:
        return self._loaded
