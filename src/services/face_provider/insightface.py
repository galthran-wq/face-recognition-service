from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np

from src.services.face_provider.base import BoundingBox, DetectedFace, FaceProvider

_BATCH_THREAD_WORKERS = 4

# ArcFace reference landmarks for 112x112 alignment
_ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def _estimate_norm(lmk: np.ndarray, image_size: int = 112) -> np.ndarray:
    """Estimate similarity transform matrix from 5 landmarks.

    Pure numpy replacement for skimage.transform.SimilarityTransform.estimate().
    Uses np.linalg.lstsq which releases the GIL, making this thread-safe.
    """
    ratio = float(image_size) / 112.0
    dst = _ARCFACE_DST * ratio

    # Solve for similarity transform: x' = a*x - b*y + tx, y' = b*x + a*y + ty
    n = lmk.shape[0]
    coeff = np.zeros((n * 2, 4), dtype=np.float64)
    target = np.zeros(n * 2, dtype=np.float64)
    for i in range(n):
        coeff[2 * i] = [lmk[i, 0], -lmk[i, 1], 1, 0]
        coeff[2 * i + 1] = [lmk[i, 1], lmk[i, 0], 0, 1]
        target[2 * i] = dst[i, 0]
        target[2 * i + 1] = dst[i, 1]

    params, _, _, _ = np.linalg.lstsq(coeff, target, rcond=None)
    a, b_val, tx, ty = params
    mat = np.array([[a, -b_val, tx], [b_val, a, ty]], dtype=np.float64)
    return mat


def _norm_crop(img: np.ndarray, landmark: np.ndarray, image_size: int = 112) -> np.ndarray:
    """Align and crop face using similarity transform. GIL-free alternative to insightface norm_crop."""
    mat = _estimate_norm(landmark, image_size)
    return cv2.warpAffine(img, mat, (image_size, image_size), borderValue=0.0)


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
        use_tensorrt: bool = False,
        trt_cache_path: str = "/models/trt_cache",
    ) -> None:
        self._use_gpu = use_gpu
        self._ctx_id = ctx_id
        self._det_size = det_size
        self._model_name = model_name
        self._model_dir = model_dir
        self._use_tensorrt = use_tensorrt
        self._trt_cache_path = trt_cache_path
        self._app: Any = None

    def load_model(self) -> None:
        import onnxruntime as ort  # type: ignore[import-untyped]
        import structlog
        from insightface.app import FaceAnalysis  # type: ignore[import-untyped]
        from insightface.model_zoo.model_zoo import PickableInferenceSession  # type: ignore[import-untyped]

        log = structlog.get_logger()

        # Monkey-patch to inject SessionOptions into all insightface ORT sessions.
        # FaceAnalysis only forwards `providers` and `provider_options` to sessions,
        # not `sess_options` (model_zoo.py:94-96). This patch fills the gap.
        _original_init = PickableInferenceSession.__init__

        def _patched_init(self_sess: PickableInferenceSession, model_path: str, **kwargs: Any) -> None:
            if "sess_options" not in kwargs:
                so = ort.SessionOptions()
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                so.enable_mem_pattern = True
                so.enable_mem_reuse = True
                kwargs["sess_options"] = so
            _original_init(self_sess, model_path, **kwargs)

        PickableInferenceSession.__init__ = _patched_init

        cuda_ep_opts: dict[str, str] = {
            "device_id": str(self._ctx_id),
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": "1",
            "cudnn_conv_use_max_workspace": "1",
        }

        if self._use_gpu and self._use_tensorrt:
            import os

            os.makedirs(self._trt_cache_path, exist_ok=True)
            trt_ep_opts: dict[str, str] = {
                "device_id": str(self._ctx_id),
                "trt_fp16_enable": "True",
                "trt_engine_cache_enable": "True",
                "trt_engine_cache_path": self._trt_cache_path,
                "trt_max_workspace_size": str(2 * 1024**3),
            }
            providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options: list[dict[str, str]] = [trt_ep_opts, cuda_ep_opts, {}]
            log.info("tensorrt_enabled", trt_options=trt_ep_opts, cache_path=self._trt_cache_path)
        elif self._use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options = [cuda_ep_opts, {}]
        else:
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]

        fa_kwargs: dict[str, Any] = {"providers": providers, "provider_options": provider_options}

        self._app = FaceAnalysis(name=self._model_name, root=self._model_dir, **fa_kwargs)
        self._app.prepare(ctx_id=self._ctx_id, det_size=self._det_size)
        self._loaded = True

        # Restore original init to avoid side effects on other code
        PickableInferenceSession.__init__ = _original_init

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
        rec_model = self._app.models["recognition"]
        crops = []
        for kps in kpss:
            aimg = _norm_crop(img, landmark=kps, image_size=rec_model.input_size[0])
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
        rec_model = self._app.models["recognition"]
        image_size = rec_model.input_size[0]

        # Stage 1: Decode images in parallel (cv2.imdecode releases the GIL)
        with ThreadPoolExecutor(max_workers=_BATCH_THREAD_WORKERS) as pool:
            decoded = list(pool.map(self._decode_image, images))

        # Stage 2: Detect faces (GPU, must be sequential)
        per_image: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None]] = []
        for img in decoded:
            if img is None:
                per_image.append((np.zeros((0, 5)), None, None))
            else:
                bboxes, kpss = self._detect_faces(img)
                per_image.append((bboxes, kpss, img))

        # Stage 3: Align face crops in parallel (_norm_crop releases the GIL)
        all_crops: list[np.ndarray] = []
        crop_counts: list[int] = []
        align_tasks: list[tuple[np.ndarray, np.ndarray]] = []
        for bboxes, kpss, img in per_image:
            if bboxes.shape[0] == 0 or kpss is None or img is None:
                crop_counts.append(0)
                continue
            for kps in kpss:
                align_tasks.append((img, kps))
            crop_counts.append(bboxes.shape[0])

        if align_tasks:
            with ThreadPoolExecutor(max_workers=_BATCH_THREAD_WORKERS) as pool:
                all_crops = list(pool.map(lambda t: _norm_crop(t[0], t[1], image_size), align_tasks))

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
        rec_model = self._app.models["recognition"]
        ga_model = self._app.models.get("genderage")
        image_size = rec_model.input_size[0]

        # Stage 1: Decode images in parallel (cv2.imdecode releases the GIL)
        with ThreadPoolExecutor(max_workers=_BATCH_THREAD_WORKERS) as pool:
            decoded = list(pool.map(self._decode_image, images))

        # Stage 2: Detect faces (GPU, must be sequential)
        per_image: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None]] = []
        for img in decoded:
            if img is None:
                per_image.append((np.zeros((0, 5)), None, None))
            else:
                bboxes, kpss = self._detect_faces(img)
                per_image.append((bboxes, kpss, img))

        # Stage 3: Align face crops in parallel (_norm_crop releases the GIL)
        all_crops: list[np.ndarray] = []
        crop_counts: list[int] = []
        align_tasks: list[tuple[np.ndarray, np.ndarray]] = []
        for bboxes, kpss, img in per_image:
            if bboxes.shape[0] == 0 or kpss is None or img is None:
                crop_counts.append(0)
                continue
            for kps in kpss:
                align_tasks.append((img, kps))
            crop_counts.append(bboxes.shape[0])

        if align_tasks:
            with ThreadPoolExecutor(max_workers=_BATCH_THREAD_WORKERS) as pool:
                all_crops = list(pool.map(lambda t: _norm_crop(t[0], t[1], image_size), align_tasks))

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
