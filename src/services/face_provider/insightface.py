from __future__ import annotations

from typing import Any

import numpy as np

from src.services.face_provider.base import BoundingBox, DetectedFace, FaceProvider


def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class InsightFaceProvider(FaceProvider):
    def __init__(
        self,
        *,
        use_gpu: bool = False,
        ctx_id: int = 0,
        det_size: tuple[int, int] = (640, 640),
        model_name: str = "buffalo_l",
    ) -> None:
        self._use_gpu = use_gpu
        self._ctx_id = ctx_id
        self._det_size = det_size
        self._model_name = model_name
        self._app: Any = None

    def load_model(self) -> None:
        from insightface.app import FaceAnalysis  # type: ignore[import-untyped]

        providers: list[str] = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"] if self._use_gpu else ["CPUExecutionProvider"]
        )
        self._app = FaceAnalysis(name=self._model_name, providers=providers)
        self._app.prepare(ctx_id=self._ctx_id, det_size=self._det_size)
        self._loaded = True

    def _get_all_faces(self, image_bytes: bytes) -> list[DetectedFace]:
        import cv2

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return []

        raw_faces: list[Any] = self._app.get(img)
        result: list[DetectedFace] = []
        for f in raw_faces:
            # Embedding
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                emb = getattr(f, "embedding", None)
            embedding: list[float] | None = None
            if isinstance(emb, np.ndarray):
                embedding = emb.astype(np.float32).tolist()
            elif hasattr(emb, "tolist"):
                embedding = emb.tolist()  # type: ignore[union-attr]
            elif isinstance(emb, (list, tuple)):
                embedding = [float(v) for v in emb]

            # Bounding box: [x1, y1, x2, y2] -> {x, y, width, height}
            raw_bbox = getattr(f, "bbox", None)
            if raw_bbox is not None:
                x1, y1, x2, y2 = (float(v) for v in raw_bbox[:4])
                bbox = BoundingBox(x=x1, y=y1, width=max(0.0, x2 - x1), height=max(0.0, y2 - y1))
            else:
                bbox = BoundingBox(x=0.0, y=0.0, width=0.0, height=0.0)

            # Age
            age = _to_float(getattr(f, "age", None))

            # Gender
            sex = getattr(f, "sex", None)
            gender: str | None = None
            if sex is not None:
                try:
                    gender = "male" if int(sex) == 1 else "female"
                except (TypeError, ValueError):
                    gender = str(sex)
            elif getattr(f, "gender", None) is not None:
                gender = str(f.gender)

            # Detection score
            det_score = _to_float(getattr(f, "det_score", None)) or 0.0

            # Race
            race: str | None = None
            if hasattr(f, "race"):
                race_val = f.race
                race = str(race_val) if race_val is not None else None

            race_probs: dict[str, float] | None = None
            for attr in ("races", "race_probs", "race_prob"):
                if hasattr(f, attr) and race_probs is None:
                    rp = getattr(f, attr)
                    if isinstance(rp, dict):
                        race_probs = {str(k): float(v) for k, v in rp.items()}
                    elif isinstance(rp, (list, tuple)):
                        race_probs = {str(i): float(v) for i, v in enumerate(rp)}

            result.append(
                DetectedFace(
                    bbox=bbox,
                    det_score=det_score,
                    embedding=embedding,
                    age=age,
                    gender=gender,
                    race=race,
                    race_probs=race_probs,
                )
            )
        return result

    def detect(self, image_bytes: bytes) -> list[DetectedFace]:
        faces = self._get_all_faces(image_bytes)
        return [DetectedFace(bbox=f.bbox, det_score=f.det_score) for f in faces]

    def embed(self, image_bytes: bytes) -> list[DetectedFace]:
        faces = self._get_all_faces(image_bytes)
        return [DetectedFace(bbox=f.bbox, det_score=f.det_score, embedding=f.embedding) for f in faces]

    def analyze(self, image_bytes: bytes) -> list[DetectedFace]:
        return self._get_all_faces(image_bytes)

    @property
    def provider_name(self) -> str:
        return "insightface"
