from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np

from src.services.face_provider.base import BoundingBox, DetectedFace, FaceProvider, HeadPose

_BATCH_THREAD_WORKERS = 4


def _trt_batch_profile(model_path: str, opt_batch: int, max_batch: int) -> dict[str, str]:
    """Build TRT optimization-profile options for a dynamic-batch model.

    Matches models whose first input dim is a dynamic batch and whose remaining
    dims are static (recognition `Nx3x112x112`, genderage `Nx3x96x96`). Returns
    ``{}`` for anything else — notably the detector, whose batch is fixed at 1
    and whose spatial dims are dynamic — so their TRT behavior is unchanged.

    With these options the TensorRT EP builds one engine spanning batch
    ``[1, max_batch]`` (cached to disk) instead of rebuilding a fresh engine the
    first time it sees each distinct batch size (~30-75 s, in-process only).
    """
    try:
        import onnx  # noqa: PLC0415

        model = onnx.load(model_path, load_external_data=False)
    except Exception:  # noqa: BLE001 — never block model load on profile probing
        return {}

    mins: list[str] = []
    opts: list[str] = []
    maxs: list[str] = []
    for inp in model.graph.input:
        dims = inp.type.tensor_type.shape.dim
        if len(dims) < 2:
            continue
        batch_dynamic = dims[0].dim_value == 0  # 0 == symbolic / unknown
        rest_static = all(d.dim_value > 0 for d in dims[1:])
        if not (batch_dynamic and rest_static):
            continue
        static = "x".join(str(d.dim_value) for d in dims[1:])
        mins.append(f"{inp.name}:1x{static}")
        opts.append(f"{inp.name}:{opt_batch}x{static}")
        maxs.append(f"{inp.name}:{max_batch}x{static}")
    if not mins:
        return {}
    return {
        "trt_profile_min_shapes": ",".join(mins),
        "trt_profile_opt_shapes": ",".join(opts),
        "trt_profile_max_shapes": ",".join(maxs),
    }


# Pad-to-square fallback: RetinaFace anchors miss faces that fill most of the
# frame. Padding to a square with a gray border restores typical face-to-frame
# ratio so anchors can match again. Applied transparently when the first
# detection pass returns zero faces; output coordinates are translated back to
# the original image space. Border/fill values are configurable per provider
# instance via `pad_fallback_border_px` / `pad_fallback_fill`.

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


def _to_pose(raw: object) -> HeadPose | None:
    """InsightFace's 1k3d68 stores pose as [pitch, yaw, roll] (degrees)."""
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float32)
    if arr.shape != (3,):
        return None
    return HeadPose(pitch=float(arr[0]), yaw=float(arr[1]), roll=float(arr[2]))


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
        trt_max_batch: int = 256,
        trt_opt_batch: int = 16,
        pad_fallback_border_px: int = 100,
        pad_fallback_fill: int = 128,
    ) -> None:
        self._use_gpu = use_gpu
        self._ctx_id = ctx_id
        self._det_size = det_size
        self._model_name = model_name
        self._model_dir = model_dir
        self._use_tensorrt = use_tensorrt
        self._trt_cache_path = trt_cache_path
        self._trt_max_batch = trt_max_batch
        self._trt_opt_batch = trt_opt_batch
        self._pad_border_px = pad_fallback_border_px
        self._pad_fill = pad_fallback_fill
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
        _trt_on = self._use_gpu and self._use_tensorrt
        _opt_batch = self._trt_opt_batch
        _max_batch = self._trt_max_batch

        # Optional per-session ORT thread caps. ORT defaults intra_op_num_threads
        # to nproc *per session*; on a many-core host running N instances each
        # loading several sessions this explodes to thousands of OS threads
        # thrashing the scheduler. Observed: 16 instances × 5 sessions × 56-core
        # host → 362 threads/instance, ~5800 total, kernel scheduler chokes,
        # per-request latency goes to seconds even with the GPU at 20% util. Set
        # FACE_INTRA_OP_THREADS / FACE_INTER_OP_THREADS (typically 2 and 1) to
        # cap; on the same box this collapsed thread count to ~70/instance and
        # restored sub-100 ms warm latency.
        import os as _os  # noqa: PLC0415

        _intra = int(_os.environ.get("FACE_INTRA_OP_THREADS", "0"))
        _inter = int(_os.environ.get("FACE_INTER_OP_THREADS", "0"))

        def _patched_init(self_sess: PickableInferenceSession, model_path: str, **kwargs: Any) -> None:
            if "sess_options" not in kwargs:
                so = ort.SessionOptions()
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                so.enable_mem_pattern = True
                so.enable_mem_reuse = True
                if _intra > 0:
                    so.intra_op_num_threads = _intra
                if _inter > 0:
                    so.inter_op_num_threads = _inter
                kwargs["sess_options"] = so
            # Give dynamic-batch models (recognition, genderage) a TRT optimization
            # profile so one engine covers batch [1, max] instead of rebuilding per
            # face-count. Scoped per session via model_path: the detector and the
            # recognizer share input name 'input.1' with incompatible shapes, so a
            # global profile would break the detector — we copy provider_options
            # and only amend the session whose model actually has a dynamic batch.
            if _trt_on and _max_batch > 0 and "providers" in kwargs and "provider_options" in kwargs:
                profile = _trt_batch_profile(model_path, _opt_batch, _max_batch)
                providers = list(kwargs["providers"])
                if profile and "TensorrtExecutionProvider" in providers:
                    idx = providers.index("TensorrtExecutionProvider")
                    opts = [dict(o) for o in kwargs["provider_options"]]
                    opts[idx] = {**opts[idx], **profile}
                    kwargs["provider_options"] = opts
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

    def _pad_to_square(self, img: np.ndarray) -> tuple[np.ndarray, int, int]:
        h, w = img.shape[:2]
        side = max(h, w) + 2 * self._pad_border_px
        canvas = np.full((side, side, 3), self._pad_fill, dtype=img.dtype)
        dy = (side - h) // 2
        dx = (side - w) // 2
        canvas[dy : dy + h, dx : dx + w] = img
        return canvas, dx, dy

    def _detect_with_pad_fallback(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, int, int]:
        """Detect faces; on miss, pad to square and retry once.

        Returns (bboxes, kpss, working_img, dx, dy). Coordinates in bboxes/kpss
        are in working_img space — alignment must use working_img, but bboxes
        and landmarks emitted to clients must be translated back via
        _translate_bbox / _translate_landmarks using (dx, dy) and the original
        frame dimensions.
        """
        bboxes, kpss = self._detect_faces(img)
        if bboxes.shape[0] > 0:
            return bboxes, kpss, img, 0, 0
        padded, dx, dy = self._pad_to_square(img)
        bboxes, kpss = self._detect_faces(padded)
        return bboxes, kpss, padded, dx, dy

    @staticmethod
    def _make_bbox(raw_bbox: np.ndarray, dx: int = 0, dy: int = 0, orig_w: int = 0, orig_h: int = 0) -> BoundingBox:
        x1, y1, x2, y2 = (float(v) for v in raw_bbox[:4])
        if dx or dy:
            x1 -= dx
            y1 -= dy
            x2 -= dx
            y2 -= dy
            # Clip to original frame; faces touching the edge can have bbox
            # extend slightly outside after translation.
            max_w = float(orig_w)
            max_h = float(orig_h)
            x1 = max(0.0, min(x1, max_w))
            y1 = max(0.0, min(y1, max_h))
            x2 = max(0.0, min(x2, max_w))
            y2 = max(0.0, min(y2, max_h))
        return BoundingBox(x=x1, y=y1, width=max(0.0, x2 - x1), height=max(0.0, y2 - y1))

    def _recognize(self, rec_model: Any, crops: list[np.ndarray]) -> np.ndarray:
        """Run recognition on face crops, chunked so the batch dimension never
        exceeds the TRT optimization-profile maximum. Without chunking a request
        with more faces than ``trt_max_batch`` would fall outside the profile and
        either error or trigger a per-shape engine rebuild.
        """
        max_b = self._trt_max_batch
        if max_b <= 0 or len(crops) <= max_b:
            return rec_model.get_feat(crops)  # type: ignore[no-any-return]
        feats = [rec_model.get_feat(crops[i : i + max_b]) for i in range(0, len(crops), max_b)]
        return np.concatenate(feats, axis=0)

    def _align_and_embed(self, img: np.ndarray, kpss: np.ndarray) -> np.ndarray:
        rec_model = self._app.models["recognition"]
        crops = []
        for kps in kpss:
            aimg = _norm_crop(img, landmark=kps, image_size=rec_model.input_size[0])
            crops.append(aimg)
        embeddings: np.ndarray = self._recognize(rec_model, crops)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized: np.ndarray = embeddings / norms
        return normalized

    @staticmethod
    def _kps_to_landmarks(kps: np.ndarray | None, dx: int = 0, dy: int = 0) -> list[tuple[float, float]] | None:
        if kps is None:
            return None
        if dx or dy:
            return [(float(p[0]) - dx, float(p[1]) - dy) for p in kps]
        return [(float(p[0]), float(p[1])) for p in kps]

    def _estimate_poses(self, img: np.ndarray, bboxes: np.ndarray, kpss: np.ndarray | None) -> list[HeadPose | None]:
        """Real head pose via the 1k3d68 landmark model (detect + pose only, no recognition)."""
        pose_model = self._app.models.get("landmark_3d_68")
        if pose_model is None:
            return [None] * bboxes.shape[0]
        poses: list[HeadPose | None] = []
        for i in range(bboxes.shape[0]):
            face_obj = _FaceProxy(bbox=bboxes[i, :4], kps=kpss[i] if kpss is not None else None)
            pose_model.get(img, face_obj)
            poses.append(_to_pose(face_obj.get("pose")))
        return poses

    def detect(self, image_bytes: bytes, include_pose: bool = False) -> list[DetectedFace]:
        img = self._decode_image(image_bytes)
        if img is None:
            return []
        return self._detect_decoded(img, include_pose)

    def detect_batch(self, images: list[bytes], include_pose: bool = False) -> list[list[DetectedFace]]:
        with ThreadPoolExecutor(max_workers=_BATCH_THREAD_WORKERS) as pool:
            decoded = list(pool.map(self._decode_image, images))
        return [[] if img is None else self._detect_decoded(img, include_pose) for img in decoded]

    def _detect_decoded(self, img: np.ndarray, include_pose: bool) -> list[DetectedFace]:
        bboxes, kpss, working, dx, dy = self._detect_with_pad_fallback(img)
        if bboxes.shape[0] == 0:
            return []

        poses: list[HeadPose | None]
        if include_pose:
            poses = self._estimate_poses(working, bboxes, kpss)
        else:
            poses = [None] * bboxes.shape[0]
        orig_h, orig_w = img.shape[:2]
        return [
            DetectedFace(
                bbox=self._make_bbox(bboxes[i, :4], dx, dy, orig_w, orig_h),
                det_score=float(bboxes[i, 4]),
                landmarks=self._kps_to_landmarks(kpss[i] if kpss is not None else None, dx, dy),
                pose=poses[i],
            )
            for i in range(bboxes.shape[0])
        ]

    def embed(self, image_bytes: bytes) -> list[DetectedFace]:
        img = self._decode_image(image_bytes)
        if img is None:
            return []

        bboxes, kpss, working, dx, dy = self._detect_with_pad_fallback(img)
        if bboxes.shape[0] == 0 or kpss is None:
            return []

        # Alignment uses padded coords + padded image so it can sample past the
        # original edges without hitting black borders.
        embeddings = self._align_and_embed(working, kpss)

        orig_h, orig_w = img.shape[:2]
        return [
            DetectedFace(
                bbox=self._make_bbox(bboxes[i, :4], dx, dy, orig_w, orig_h),
                det_score=float(bboxes[i, 4]),
                embedding=embeddings[i].astype(np.float32).tolist(),
                landmarks=self._kps_to_landmarks(kpss[i], dx, dy),
            )
            for i in range(bboxes.shape[0])
        ]

    def analyze(self, image_bytes: bytes) -> list[DetectedFace]:
        img = self._decode_image(image_bytes)
        if img is None:
            return []

        bboxes, kpss, working, dx, dy = self._detect_with_pad_fallback(img)
        if bboxes.shape[0] == 0 or kpss is None:
            return []

        embeddings = self._align_and_embed(working, kpss)
        ga_model = self._app.models.get("genderage")

        orig_h, orig_w = img.shape[:2]
        results: list[DetectedFace] = []
        for i in range(bboxes.shape[0]):
            age: float | None = None
            gender: str | None = None

            if ga_model is not None:
                face_obj = _FaceProxy(bbox=bboxes[i, :4], kps=kpss[i] if kpss is not None else None)
                ga_model.get(working, face_obj)
                age = _to_float(face_obj.get("age"))
                gender_val = face_obj.get("gender")
                if gender_val is not None:
                    try:
                        gender = "male" if int(gender_val) == 1 else "female"  # type: ignore[call-overload]
                    except (TypeError, ValueError):
                        gender = str(gender_val)

            results.append(
                DetectedFace(
                    bbox=self._make_bbox(bboxes[i, :4], dx, dy, orig_w, orig_h),
                    det_score=float(bboxes[i, 4]),
                    embedding=embeddings[i].astype(np.float32).tolist(),
                    age=age,
                    gender=gender,
                    landmarks=self._kps_to_landmarks(kpss[i], dx, dy),
                )
            )

        return results

    def embed_batch(self, images: list[bytes]) -> list[list[DetectedFace]]:
        rec_model = self._app.models["recognition"]
        image_size = rec_model.input_size[0]

        # Stage 1: Decode images in parallel (cv2.imdecode releases the GIL)
        with ThreadPoolExecutor(max_workers=_BATCH_THREAD_WORKERS) as pool:
            decoded = list(pool.map(self._decode_image, images))

        # Stage 2: Detect faces (GPU, must be sequential). On miss, retry once
        # on a padded copy — same fallback as the single-image path.
        per_image: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None, int, int, int, int]] = []
        for img in decoded:
            if img is None:
                per_image.append((np.zeros((0, 5)), None, None, 0, 0, 0, 0))
            else:
                bboxes, kpss, working, dx, dy = self._detect_with_pad_fallback(img)
                per_image.append((bboxes, kpss, working, dx, dy, img.shape[0], img.shape[1]))

        # Stage 3: Align face crops in parallel (_norm_crop releases the GIL)
        all_crops: list[np.ndarray] = []
        crop_counts: list[int] = []
        align_tasks: list[tuple[np.ndarray, np.ndarray]] = []
        for it in per_image:
            it_bboxes, it_kpss, it_working = it[0], it[1], it[2]
            if it_bboxes.shape[0] == 0 or it_kpss is None or it_working is None:
                crop_counts.append(0)
                continue
            for kps in it_kpss:
                align_tasks.append((it_working, kps))
            crop_counts.append(it_bboxes.shape[0])

        if align_tasks:
            with ThreadPoolExecutor(max_workers=_BATCH_THREAD_WORKERS) as pool:
                all_crops = list(pool.map(lambda t: _norm_crop(t[0], t[1], image_size), align_tasks))

        if all_crops:
            all_embeddings: np.ndarray = self._recognize(rec_model, all_crops)
            norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            all_embeddings = all_embeddings / norms
        else:
            all_embeddings = np.zeros((0, 512), dtype=np.float32)

        results: list[list[DetectedFace]] = []
        emb_offset = 0
        for idx, it in enumerate(per_image):
            it_bboxes, it_kpss, _, it_dx, it_dy, it_oh, it_ow = it
            n = crop_counts[idx]
            faces: list[DetectedFace] = []
            for i in range(n):
                faces.append(
                    DetectedFace(
                        bbox=self._make_bbox(it_bboxes[i, :4], it_dx, it_dy, it_ow, it_oh),
                        det_score=float(it_bboxes[i, 4]),
                        embedding=all_embeddings[emb_offset + i].astype(np.float32).tolist(),
                        landmarks=self._kps_to_landmarks(it_kpss[i] if it_kpss is not None else None, it_dx, it_dy),
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

        # Stage 2: Detect faces (GPU, must be sequential). On miss, retry once
        # on a padded copy.
        per_image: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None, int, int, int, int]] = []
        for img in decoded:
            if img is None:
                per_image.append((np.zeros((0, 5)), None, None, 0, 0, 0, 0))
            else:
                bboxes, kpss, working, dx, dy = self._detect_with_pad_fallback(img)
                per_image.append((bboxes, kpss, working, dx, dy, img.shape[0], img.shape[1]))

        # Stage 3: Align face crops in parallel (_norm_crop releases the GIL)
        all_crops: list[np.ndarray] = []
        crop_counts: list[int] = []
        align_tasks: list[tuple[np.ndarray, np.ndarray]] = []
        for it in per_image:
            it_bboxes, it_kpss, it_working = it[0], it[1], it[2]
            if it_bboxes.shape[0] == 0 or it_kpss is None or it_working is None:
                crop_counts.append(0)
                continue
            for kps in it_kpss:
                align_tasks.append((it_working, kps))
            crop_counts.append(it_bboxes.shape[0])

        if align_tasks:
            with ThreadPoolExecutor(max_workers=_BATCH_THREAD_WORKERS) as pool:
                all_crops = list(pool.map(lambda t: _norm_crop(t[0], t[1], image_size), align_tasks))

        if all_crops:
            all_embeddings: np.ndarray = self._recognize(rec_model, all_crops)
            norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            all_embeddings = all_embeddings / norms
        else:
            all_embeddings = np.zeros((0, 512), dtype=np.float32)

        results: list[list[DetectedFace]] = []
        emb_offset = 0
        for idx, it in enumerate(per_image):
            it_bboxes, it_kpss, it_working, it_dx, it_dy, it_oh, it_ow = it
            n = crop_counts[idx]
            faces: list[DetectedFace] = []
            for i in range(n):
                age: float | None = None
                gender: str | None = None

                if ga_model is not None and it_working is not None and it_kpss is not None:
                    face_obj = _FaceProxy(bbox=it_bboxes[i, :4], kps=it_kpss[i])
                    ga_model.get(it_working, face_obj)
                    age = _to_float(face_obj.get("age"))
                    gender_val = face_obj.get("gender")
                    if gender_val is not None:
                        try:
                            gender = "male" if int(gender_val) == 1 else "female"  # type: ignore[call-overload]
                        except (TypeError, ValueError):
                            gender = str(gender_val)

                faces.append(
                    DetectedFace(
                        bbox=self._make_bbox(it_bboxes[i, :4], it_dx, it_dy, it_ow, it_oh),
                        det_score=float(it_bboxes[i, 4]),
                        embedding=all_embeddings[emb_offset + i].astype(np.float32).tolist(),
                        age=age,
                        gender=gender,
                        landmarks=self._kps_to_landmarks(it_kpss[i] if it_kpss is not None else None, it_dx, it_dy),
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
