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
        model_dir: str = "~/.insightface",
    ) -> None:
        self._use_gpu = use_gpu
        self._ctx_id = ctx_id
        self._det_size = det_size
        self._model_name = model_name
        self._model_dir = model_dir
        self._app: Any = None

    def load_model(self) -> None:
        from insightface.app import FaceAnalysis  # type: ignore[import-untyped]

        providers: list[str] = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"] if self._use_gpu else ["CPUExecutionProvider"]
        )
        self._app = FaceAnalysis(name=self._model_name, root=self._model_dir, providers=providers)
        self._app.prepare(ctx_id=self._ctx_id, det_size=self._det_size)
        self._loaded = True

    def _decode_image(self, image_bytes: bytes) -> np.ndarray | None:
        import cv2

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _detect_faces(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        bboxes, kpss = self._app.det_model.detect(img, max_num=0, metric="default")
        return bboxes, kpss

    @staticmethod
    def _make_bbox(raw_bbox: np.ndarray) -> BoundingBox:
        x1, y1, x2, y2 = (float(v) for v in raw_bbox[:4])
        return BoundingBox(x=x1, y=y1, width=max(0.0, x2 - x1), height=max(0.0, y2 - y1))

    def _align_and_embed(self, img: np.ndarray, kpss: np.ndarray) -> np.ndarray:
        from insightface.utils import face_align  # type: ignore[import-untyped]

        rec_model = self._app.models["recognition"]
        crops = []
        for kps in kpss:
            aimg = face_align.norm_crop(img, landmark=kps, image_size=rec_model.input_size[0])
            crops.append(aimg)
        embeddings: np.ndarray = rec_model.get_feat(crops)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized: np.ndarray = embeddings / norms
        return normalized

    def detect(self, image_bytes: bytes) -> list[DetectedFace]:
        img = self._decode_image(image_bytes)
        if img is None:
            return []

        bboxes, _kpss = self._detect_faces(img)
        if bboxes.shape[0] == 0:
            return []

        return [
            DetectedFace(bbox=self._make_bbox(bboxes[i, :4]), det_score=float(bboxes[i, 4]))
            for i in range(bboxes.shape[0])
        ]

    def embed(self, image_bytes: bytes) -> list[DetectedFace]:
        img = self._decode_image(image_bytes)
        if img is None:
            return []

        bboxes, kpss = self._detect_faces(img)
        if bboxes.shape[0] == 0 or kpss is None:
            return []

        embeddings = self._align_and_embed(img, kpss)

        return [
            DetectedFace(
                bbox=self._make_bbox(bboxes[i, :4]),
                det_score=float(bboxes[i, 4]),
                embedding=embeddings[i].astype(np.float32).tolist(),
            )
            for i in range(bboxes.shape[0])
        ]

    def analyze(self, image_bytes: bytes) -> list[DetectedFace]:
        img = self._decode_image(image_bytes)
        if img is None:
            return []

        bboxes, kpss = self._detect_faces(img)
        if bboxes.shape[0] == 0 or kpss is None:
            return []

        embeddings = self._align_and_embed(img, kpss)
        ga_model = self._app.models.get("genderage")

        results: list[DetectedFace] = []
        for i in range(bboxes.shape[0]):
            age: float | None = None
            gender: str | None = None

            if ga_model is not None:
                face_obj = _FaceProxy(bbox=bboxes[i, :4], kps=kpss[i] if kpss is not None else None)
                ga_model.get(img, face_obj)
                age = _to_float(face_obj.get("age"))
                gender_val = face_obj.get("gender")
                if gender_val is not None:
                    try:
                        gender = "male" if int(gender_val) == 1 else "female"  # type: ignore[call-overload]
                    except (TypeError, ValueError):
                        gender = str(gender_val)

            results.append(
                DetectedFace(
                    bbox=self._make_bbox(bboxes[i, :4]),
                    det_score=float(bboxes[i, 4]),
                    embedding=embeddings[i].astype(np.float32).tolist(),
                    age=age,
                    gender=gender,
                )
            )

        return results

    def embed_batch(self, images: list[bytes]) -> list[list[DetectedFace]]:
        from insightface.utils import face_align

        rec_model = self._app.models["recognition"]

        per_image: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None]] = []
        all_crops: list[np.ndarray] = []
        crop_counts: list[int] = []

        for image_bytes in images:
            img = self._decode_image(image_bytes)
            if img is None:
                per_image.append((np.zeros((0, 5)), None, None))
                crop_counts.append(0)
                continue

            bboxes, kpss = self._detect_faces(img)
            per_image.append((bboxes, kpss, img))
            if bboxes.shape[0] == 0 or kpss is None:
                crop_counts.append(0)
                continue

            for kps in kpss:
                aimg = face_align.norm_crop(img, landmark=kps, image_size=rec_model.input_size[0])
                all_crops.append(aimg)
            crop_counts.append(bboxes.shape[0])

        if all_crops:
            all_embeddings: np.ndarray = rec_model.get_feat(all_crops)
            norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            all_embeddings = all_embeddings / norms
        else:
            all_embeddings = np.zeros((0, 512), dtype=np.float32)

        results: list[list[DetectedFace]] = []
        emb_offset = 0
        for idx, (bboxes, _kpss, _img) in enumerate(per_image):
            n = crop_counts[idx]
            faces: list[DetectedFace] = []
            for i in range(n):
                faces.append(
                    DetectedFace(
                        bbox=self._make_bbox(bboxes[i, :4]),
                        det_score=float(bboxes[i, 4]),
                        embedding=all_embeddings[emb_offset + i].astype(np.float32).tolist(),
                    )
                )
            emb_offset += n
            results.append(faces)

        return results

    def analyze_batch(self, images: list[bytes]) -> list[list[DetectedFace]]:
        from insightface.utils import face_align

        rec_model = self._app.models["recognition"]
        ga_model = self._app.models.get("genderage")

        per_image: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None]] = []
        all_crops: list[np.ndarray] = []
        crop_counts: list[int] = []

        for image_bytes in images:
            img = self._decode_image(image_bytes)
            if img is None:
                per_image.append((np.zeros((0, 5)), None, None))
                crop_counts.append(0)
                continue

            bboxes, kpss = self._detect_faces(img)
            per_image.append((bboxes, kpss, img))
            if bboxes.shape[0] == 0 or kpss is None:
                crop_counts.append(0)
                continue

            for kps in kpss:
                aimg = face_align.norm_crop(img, landmark=kps, image_size=rec_model.input_size[0])
                all_crops.append(aimg)
            crop_counts.append(bboxes.shape[0])

        if all_crops:
            all_embeddings: np.ndarray = rec_model.get_feat(all_crops)
            norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            all_embeddings = all_embeddings / norms
        else:
            all_embeddings = np.zeros((0, 512), dtype=np.float32)

        results: list[list[DetectedFace]] = []
        emb_offset = 0
        for idx, (bboxes, kpss, img) in enumerate(per_image):
            n = crop_counts[idx]
            faces: list[DetectedFace] = []
            for i in range(n):
                age: float | None = None
                gender: str | None = None

                if ga_model is not None and img is not None and kpss is not None:
                    face_obj = _FaceProxy(bbox=bboxes[i, :4], kps=kpss[i])
                    ga_model.get(img, face_obj)
                    age = _to_float(face_obj.get("age"))
                    gender_val = face_obj.get("gender")
                    if gender_val is not None:
                        try:
                            gender = "male" if int(gender_val) == 1 else "female"  # type: ignore[call-overload]
                        except (TypeError, ValueError):
                            gender = str(gender_val)

                faces.append(
                    DetectedFace(
                        bbox=self._make_bbox(bboxes[i, :4]),
                        det_score=float(bboxes[i, 4]),
                        embedding=all_embeddings[emb_offset + i].astype(np.float32).tolist(),
                        age=age,
                        gender=gender,
                    )
                )
            emb_offset += n
            results.append(faces)

        return results

    @property
    def provider_name(self) -> str:
        return "insightface"


class _FaceProxy:
    def __init__(self, bbox: np.ndarray, kps: np.ndarray | None) -> None:
        self.bbox = bbox
        self.kps = kps
        self._attrs: dict[str, Any] = {}

    def __setitem__(self, key: str, value: object) -> None:
        self._attrs[key] = value

    def __getitem__(self, key: str) -> object:
        return self._attrs[key]

    def get(self, key: str, default: object = None) -> object:
        return self._attrs.get(key, default)
