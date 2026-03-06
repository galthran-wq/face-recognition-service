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

    def detect_batch(self, images: list[bytes]) -> list[list[DetectedFace]]:
        """Process multiple images. Default: sequential calls to detect()."""
        return [self.detect(img) for img in images]

    def embed_batch(self, images: list[bytes]) -> list[list[DetectedFace]]:
        """Process multiple images with cross-image batched recognition."""
        return [self.embed(img) for img in images]

    def analyze_batch(self, images: list[bytes]) -> list[list[DetectedFace]]:
        """Process multiple images with cross-image batched recognition + attributes."""
        return [self.analyze(img) for img in images]

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    def is_loaded(self) -> bool:
        return self._loaded
