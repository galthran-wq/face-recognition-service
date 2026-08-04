from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

import cv2
import numpy as np

from src.services.face_provider.base import BoundingBox, DetectedFace, FaceProvider, HeadPose


class CvWorkPool:
    """Persistent thread pool for CPU-bound cv2/numpy work.

    JPEG decode, letterbox resize, warpAffine crops and blob fills all release
    the GIL, so they thread well — but a fresh ThreadPoolExecutor per request
    pays thread spawn/teardown on every call. This keeps one warm pool for the
    provider's lifetime (threads are created lazily by the executor, so an
    unused pool costs nothing). Sized via FACE_THREAD_WORKERS.
    """

    def __init__(self, workers: int) -> None:
        self._workers = max(1, workers)
        self._pool = ThreadPoolExecutor(max_workers=self._workers, thread_name_prefix="face-cv")

    def decode_batch(
        self, decode: Callable[[bytes], np.ndarray | None], images: list[bytes]
    ) -> list[np.ndarray | None]:
        """Decode a batch of encoded images (JPEG/PNG) in parallel."""
        return self.map(decode, images)

    def map(self, fn: Callable[[Any], Any], items: Sequence[Any]) -> list[Any]:
        if len(items) <= 1:
            return [fn(item) for item in items]
        return list(self._pool.map(fn, items))


# Per-image detection state: (bboxes, kpss, working_img, dx, dy, orig_h, orig_w).
_DetResult = tuple[np.ndarray, "np.ndarray | None", "np.ndarray | None", int, int, int, int]


def _trt_batch_profile(
    model_path: str,
    opt_batch: int,
    max_batch: int,
    det_static_dims: str | None = None,
    det_opt_batch: int = 0,
    det_max_batch: int = 0,
) -> dict[str, str]:
    """Build TRT optimization-profile options for a dynamic-batch model.

    Matches models whose first input dim is a dynamic batch and whose remaining
    dims are static (recognition `Nx3x112x112`, genderage `Nx3x96x96`, and the
    re-exported detector `Nx3x640x640` — see ``scrfd_export``). Returns ``{}``
    for anything else.

    The detector's inputs are ~30x larger than a recognition crop, so it gets
    its own batch bounds: an input whose static dims equal ``det_static_dims``
    (e.g. ``"3x640x640"``) uses ``det_opt_batch``/``det_max_batch`` instead of
    the shared bounds. Each knob disables only its own profile when <= 0 —
    ``max_batch <= 0`` skips recognition/genderage inputs, ``det_max_batch <=
    0`` skips the detector input; the two are independent.

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
        if det_static_dims is not None and static == det_static_dims:
            if det_max_batch <= 0:
                continue
            inp_opt, inp_max = det_opt_batch, det_max_batch
        else:
            if max_batch <= 0:
                continue  # shared profile disabled — never emit a 0x... shape
            inp_opt, inp_max = opt_batch, max_batch
        inp_opt = max(1, min(inp_opt, inp_max))
        mins.append(f"{inp.name}:1x{static}")
        opts.append(f"{inp.name}:{inp_opt}x{static}")
        maxs.append(f"{inp.name}:{inp_max}x{static}")
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
        det_dynamic_batch: bool = True,
        det_uint8_input: bool = True,
        det_trt_max_batch: int = 32,
        det_trt_opt_batch: int = 8,
        thread_workers: int = 8,
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
        self._det_dynamic_batch = det_dynamic_batch
        self._det_uint8_input = det_uint8_input
        self._det_trt_max_batch = det_trt_max_batch
        self._det_trt_opt_batch = det_trt_opt_batch
        self._pad_border_px = pad_fallback_border_px
        self._pad_fill = pad_fallback_fill
        self._cv_pool = CvWorkPool(thread_workers)
        # Serializes GPU passes when FACE_MAX_INFLIGHT lets several requests
        # run their CPU stages concurrently. Held only around session.run-style
        # calls, per chunk, so requests interleave at chunk granularity.
        # Everything else touched concurrently is safe: CvWorkPool is a
        # thread-safe executor, _det_center_cache worst-cases a duplicate
        # compute under the GIL, and all blobs/buffers are per-call locals.
        self._gpu_lock = threading.Lock()
        self._det_center_cache: dict[int, np.ndarray] = {}
        self._app: Any = None

    def load_model(self) -> None:
        import onnxruntime as ort  # type: ignore[import-untyped]
        import structlog
        from insightface.app import FaceAnalysis  # type: ignore[import-untyped]
        from insightface.model_zoo.model_zoo import PickableInferenceSession  # type: ignore[import-untyped]

        log = structlog.get_logger()

        # Re-export the detector with a dynamic batch dim before any session is
        # created, so one session.run covers a whole request batch (issue #126).
        # ensure_available downloads the model pack if missing — the same call
        # FaceAnalysis makes first thing in __init__, so no duplicate work.
        import glob as _glob  # noqa: PLC0415
        import os as _os  # noqa: PLC0415

        if self._det_dynamic_batch:
            from insightface.utils import ensure_available  # type: ignore[import-untyped]

            from src.services.face_provider.scrfd_export import convert_scrfd_to_dynamic_batch  # noqa: PLC0415

            pack_dir = ensure_available("models", self._model_name, root=_os.path.expanduser(self._model_dir))
            for det_path in sorted(_glob.glob(_os.path.join(pack_dir, "det_*.onnx"))):
                outcome = convert_scrfd_to_dynamic_batch(
                    det_path, det_size=self._det_size, uint8_input=self._det_uint8_input
                )
                log.info("scrfd_dynamic_batch_export", model=det_path, outcome=outcome, uint8=self._det_uint8_input)
        else:
            # True rollback: restore the stock batch-1 graphs from their .bak
            # so the flag doesn't just skip the export while a previously
            # converted file keeps running. Consuming the .bak is idempotent —
            # re-enabling the flag recreates it from the restored original.
            pack_dir = _os.path.join(_os.path.expanduser(self._model_dir), "models", self._model_name)
            for bak_path in sorted(_glob.glob(_os.path.join(pack_dir, "det_*.onnx.bak"))):
                det_path = bak_path.removesuffix(".bak")
                try:
                    _os.replace(bak_path, det_path)
                except FileNotFoundError:
                    continue  # a concurrently restoring peer won the race
                log.info("scrfd_dynamic_batch_restore", model=det_path)

        # Monkey-patch to inject SessionOptions into all insightface ORT sessions.
        # FaceAnalysis only forwards `providers` and `provider_options` to sessions,
        # not `sess_options` (model_zoo.py:94-96). This patch fills the gap.
        _original_init = PickableInferenceSession.__init__
        _trt_on = self._use_gpu and self._use_tensorrt
        _opt_batch = self._trt_opt_batch
        _max_batch = self._trt_max_batch
        _det_opt_batch = self._det_trt_opt_batch
        _det_max_batch = self._det_trt_max_batch
        # NCHW static dims of the re-exported detector — used to give it its
        # own (smaller) TRT batch bounds; det_size is (W, H).
        _det_static_dims = f"3x{self._det_size[1]}x{self._det_size[0]}"

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
            # Give dynamic-batch models (recognition, genderage, and the
            # re-exported detector) a TRT optimization profile so one engine
            # covers batch [1, max] instead of rebuilding per batch size.
            # Scoped per session via model_path: the detector and the
            # recognizer share input name 'input.1' with incompatible shapes,
            # so a global profile would break — we copy provider_options and
            # only amend the session whose model actually has a dynamic batch.
            profiles_wanted = _max_batch > 0 or _det_max_batch > 0
            if _trt_on and profiles_wanted and "providers" in kwargs and "provider_options" in kwargs:
                profile = _trt_batch_profile(
                    model_path,
                    _opt_batch,
                    _max_batch,
                    det_static_dims=_det_static_dims,
                    det_opt_batch=_det_opt_batch,
                    det_max_batch=_det_max_batch,
                )
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
        with self._gpu_lock:
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

    def _det_batch_capable(self) -> bool:
        """True if the loaded detector graph supports batched inference: a
        dynamic batch dim with static spatial dims (see ``scrfd_export``), on a
        detector exposing the SCRFD decode attributes. Anything else — stock
        batch-1 graphs included — falls back to the per-image detect loop."""
        det_model = self._app.det_model
        if not isinstance(getattr(det_model, "fmc", None), int):
            return False
        shape = det_model.session.get_inputs()[0].shape
        return (
            isinstance(shape, (list, tuple))
            and len(shape) == 4
            and not isinstance(shape[0], int)
            and isinstance(shape[2], int)
            and isinstance(shape[3], int)
        )

    def _letterbox(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """Aspect-preserving resize onto a det_size canvas (top-left anchored),
        exactly mirroring insightface's SCRFD.detect preprocessing."""
        input_w, input_h = self._app.det_model.input_size
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_h) / input_w
        if im_ratio > model_ratio:
            new_height = input_h
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_w
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized
        return det_img, det_scale

    def _det_anchor_centers(self, stride: int) -> np.ndarray:
        """Anchor-center grid for one FPN stride; input size is static in the
        batched graph so one cache entry per stride is enough."""
        cached = self._det_center_cache.get(stride)
        if cached is not None:
            return cached
        det_model = self._app.det_model
        input_w, input_h = det_model.input_size
        height, width = input_h // stride, input_w // stride
        grid = np.mgrid[:height, :width]
        centers = np.stack((grid[1], grid[0]), axis=-1).astype(np.float32)
        centers = (centers * stride).reshape((-1, 2))
        if det_model._num_anchors > 1:
            centers = np.stack([centers] * det_model._num_anchors, axis=1).reshape((-1, 2))
        self._det_center_cache[stride] = centers
        return centers

    def _decode_det_output(
        self, net_outs: list[np.ndarray], b: int, det_scale: float
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Decode one image's slice of a batched SCRFD forward pass into
        (bboxes, kpss), mirroring SCRFD.forward + SCRFD.detect post-processing.

        The converted graph keeps the stock flat 2-D outputs, so a batch of N
        yields (N*K, C) per stride with image b occupying rows [b*K:(b+1)*K].
        """
        from insightface.model_zoo.scrfd import (  # type: ignore[import-untyped] # noqa: PLC0415
            distance2bbox,
            distance2kps,
        )

        det_model = self._app.det_model
        input_w, input_h = det_model.input_size
        fmc = det_model.fmc
        scores_list: list[np.ndarray] = []
        bboxes_list: list[np.ndarray] = []
        kpss_list: list[np.ndarray] = []
        for idx, stride in enumerate(det_model._feat_stride_fpn):
            k = (input_h // stride) * (input_w // stride) * det_model._num_anchors
            rows = slice(b * k, (b + 1) * k)
            scores = net_outs[idx][rows]
            anchor_centers = self._det_anchor_centers(stride)
            pos_inds = np.where(scores >= det_model.det_thresh)[0]
            # Unlike insightface's forward(), decode only the anchors above
            # threshold — distance2bbox/kps are row-wise, so results are
            # identical and the per-image cost drops from ~25k anchors to a
            # handful.
            bbox_preds = net_outs[idx + fmc][rows][pos_inds] * stride
            bboxes = distance2bbox(anchor_centers[pos_inds], bbox_preds)
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes)
            if det_model.use_kps:
                kps_preds = net_outs[idx + fmc * 2][rows][pos_inds] * stride
                kpss = distance2kps(anchor_centers[pos_inds], kps_preds)
                kpss_list.append(kpss.reshape((kpss.shape[0], kpss.shape[1] // 2, 2)))

        scores_all = np.vstack(scores_list)
        order = scores_all.ravel().argsort()[::-1]
        bboxes_all = np.vstack(bboxes_list) / det_scale
        pre_det = np.hstack((bboxes_all, scores_all)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = det_model.nms(pre_det)
        det = pre_det[keep, :]
        if det_model.use_kps:
            kpss_all = np.vstack(kpss_list) / det_scale
            return det, kpss_all[order, :, :][keep, :, :]
        return det, None

    def _det_input_is_uint8(self) -> bool:
        """True if the detector graph carries its own normalization and takes
        raw uint8 BGR NCHW canvases (see scrfd_export uint8_input)."""
        return bool(self._app.det_model.session.get_inputs()[0].type == "tensor(uint8)")

    def _letterbox_blob(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """Letterbox one image and convert it to a (1, 3, H, W) network blob.

        Per-image ``blobFromImage`` (which releases the GIL, so it threads
        well) is ~5x faster than one ``blobFromImages`` call over the whole
        batch, and bit-identical to insightface's own preprocessing.
        """
        det_model = self._app.det_model
        det_img, det_scale = self._letterbox(img)
        input_w, input_h = det_model.input_size
        blob: np.ndarray = cv2.dnn.blobFromImage(
            det_img,
            1.0 / det_model.input_std,
            (input_w, input_h),
            (det_model.input_mean, det_model.input_mean, det_model.input_mean),
            swapRB=True,
        )
        return blob, det_scale

    def _detect_faces_batched(self, imgs: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray | None]]:
        """Detect faces in N images with one session.run per chunk instead of
        one per image. Chunk size is bounded by the detector's TRT profile max
        so every run stays inside the prebuilt engine.

        Workers write each image's blob straight into its slice of one
        preallocated (N, 3, H, W) array — a serial np.concatenate over the
        batch costs more than the forward pass itself.
        """
        det_model = self._app.det_model
        n = len(imgs)
        input_w, input_h = det_model.input_size
        uint8_input = self._det_input_is_uint8()
        blob = np.empty((n, 3, input_h, input_w), dtype=np.uint8 if uint8_input else np.float32)
        det_scales = [0.0] * n

        def _fill(i: int) -> None:
            if uint8_input:
                # Normalization lives in the graph — ship the raw letterboxed
                # canvas (uint8 CHW, 4x less PCIe than a float blob).
                det_img, det_scale = self._letterbox(imgs[i])
                blob[i] = det_img.transpose(2, 0, 1)
            else:
                img_blob, det_scale = self._letterbox_blob(imgs[i])
                blob[i] = img_blob[0]
            det_scales[i] = det_scale

        self._cv_pool.map(_fill, range(n))

        max_b = self._det_trt_max_batch if self._det_trt_max_batch > 0 else n
        results: list[tuple[np.ndarray, np.ndarray | None]] = []
        for start in range(0, n, max_b):
            chunk = blob[start : start + max_b]
            with self._gpu_lock:
                net_outs = det_model.session.run(det_model.output_names, {det_model.input_name: chunk})
            for b in range(chunk.shape[0]):
                results.append(self._decode_det_output(net_outs, b, det_scales[start + b]))
        return results

    def _detect_batch(self, imgs: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray | None]]:
        """Batched detection when the graph supports it, sequential otherwise."""
        if not imgs:
            return []
        if self._det_batch_capable():
            return self._detect_faces_batched(imgs)
        return [self._detect_faces(img) for img in imgs]

    def _detect_with_pad_fallback_batch(self, decoded: list[np.ndarray | None]) -> list[_DetResult]:
        """Detect faces across a batch; images with zero faces get one batched
        retry on a padded-to-square copy (see the pad-fallback note above).

        Coordinates in bboxes/kpss are in working_img space — alignment must
        use working_img, but bboxes and landmarks emitted to clients must be
        translated back via _make_bbox / _kps_to_landmarks using (dx, dy) and
        the original frame dimensions. Undecodable (None) images yield empty
        entries.
        """
        results: list[_DetResult | None] = [None] * len(decoded)
        valid = [i for i, img in enumerate(decoded) if img is not None]

        first_pass = self._detect_batch([decoded[i] for i in valid])  # type: ignore[misc]
        misses: list[int] = []
        for i, (bboxes, kpss) in zip(valid, first_pass, strict=True):
            img = decoded[i]
            assert img is not None
            if bboxes.shape[0] > 0:
                results[i] = (bboxes, kpss, img, 0, 0, img.shape[0], img.shape[1])
            else:
                misses.append(i)

        if misses:
            padded = [self._pad_to_square(decoded[i]) for i in misses]  # type: ignore[arg-type]
            second_pass = self._detect_batch([canvas for canvas, _, _ in padded])
            for i, (canvas, dx, dy), (bboxes, kpss) in zip(misses, padded, second_pass, strict=True):
                img = decoded[i]
                assert img is not None
                results[i] = (bboxes, kpss, canvas, dx, dy, img.shape[0], img.shape[1])

        empty: _DetResult = (np.zeros((0, 5), dtype=np.float32), None, None, 0, 0, 0, 0)
        return [r if r is not None else empty for r in results]

    def _detect_with_pad_fallback(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, int, int]:
        """Single-image wrapper over the batched pad-fallback path."""
        bboxes, kpss, working, dx, dy, _, _ = self._detect_with_pad_fallback_batch([img])[0]
        assert working is not None
        return bboxes, kpss, working, dx, dy

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
            with self._gpu_lock:
                return rec_model.get_feat(crops)  # type: ignore[no-any-return]
        feats = []
        for i in range(0, len(crops), max_b):
            with self._gpu_lock:
                feats.append(rec_model.get_feat(crops[i : i + max_b]))
        return np.concatenate(feats, axis=0)

    @staticmethod
    def _ga_batch_capable(ga_model: Any) -> bool:
        """True if the genderage model can take a batched forward pass: the
        stock graph already has a dynamic batch dim (`[None,3,96,96]`), so this
        mostly guards against test doubles and exotic model packs."""
        if getattr(ga_model, "taskname", None) != "genderage":
            return False
        shape = ga_model.session.get_inputs()[0].shape
        return (
            isinstance(shape, (list, tuple))
            and len(shape) == 4
            and not isinstance(shape[0], int)
            and isinstance(shape[2], int)
            and isinstance(shape[3], int)
        )

    @staticmethod
    def _ga_crop(img: np.ndarray, bbox: np.ndarray, size: int) -> np.ndarray:
        """Center-crop a face for the genderage model — numpy equivalent of
        insightface's face_align.transform with rotation 0 (skimage-free, and
        cv2.warpAffine releases the GIL so it threads well)."""
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        s = size / (max(w, h) * 1.5)
        mat = np.array([[s, 0.0, size / 2 - s * cx], [0.0, s, size / 2 - s * cy]], dtype=np.float64)
        return cv2.warpAffine(img, mat, (size, size), borderValue=0.0)

    def _genderage_batch(
        self, ga_model: Any, tasks: list[tuple[np.ndarray, np.ndarray]]
    ) -> list[tuple[float | None, str | None]]:
        """Run genderage over (working_img, bbox) pairs with one session.run
        per chunk instead of one per face. Crops mirror Attribute.get exactly;
        chunking is bounded by the shared TRT profile max (same as recognition).
        """
        size = ga_model.input_size[0]
        n = len(tasks)
        blob = np.empty((n, 3, size, size), dtype=np.float32)

        def _fill(i: int) -> None:
            img, bbox = tasks[i]
            aimg = self._ga_crop(img, bbox, size)
            blob[i] = cv2.dnn.blobFromImage(
                aimg,
                1.0 / ga_model.input_std,
                (size, size),
                (ga_model.input_mean, ga_model.input_mean, ga_model.input_mean),
                swapRB=True,
            )[0]

        self._cv_pool.map(_fill, range(n))

        max_b = self._trt_max_batch if self._trt_max_batch > 0 else n
        results: list[tuple[float | None, str | None]] = []
        for start in range(0, n, max_b):
            with self._gpu_lock:
                preds = ga_model.session.run(ga_model.output_names, {ga_model.input_name: blob[start : start + max_b]})[
                    0
                ]
            for pred in preds:
                age = float(int(np.round(pred[2] * 100)))
                gender = "male" if int(np.argmax(pred[:2])) == 1 else "female"
                results.append((age, gender))
        return results

    def _genderage_for_image(
        self, ga_model: Any, working: np.ndarray, bboxes: np.ndarray, kpss: np.ndarray | None
    ) -> list[tuple[float | None, str | None]]:
        """Genderage for all faces of one image: batched when the graph allows
        it, per-face Attribute.get otherwise (also the path test doubles hit)."""
        if self._ga_batch_capable(ga_model):
            return self._genderage_batch(ga_model, [(working, bboxes[i, :4]) for i in range(bboxes.shape[0])])
        results: list[tuple[float | None, str | None]] = []
        for i in range(bboxes.shape[0]):
            face_obj = _FaceProxy(bbox=bboxes[i, :4], kps=kpss[i] if kpss is not None else None)
            with self._gpu_lock:
                ga_model.get(working, face_obj)
            age = _to_float(face_obj.get("age"))
            gender_val = face_obj.get("gender")
            gender: str | None = None
            if gender_val is not None:
                try:
                    gender = "male" if int(gender_val) == 1 else "female"  # type: ignore[call-overload]
                except (TypeError, ValueError):
                    gender = str(gender_val)
            results.append((age, gender))
        return results

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
            with self._gpu_lock:
                pose_model.get(img, face_obj)
            poses.append(_to_pose(face_obj.get("pose")))
        return poses

    def detect(self, image_bytes: bytes, include_pose: bool = False) -> list[DetectedFace]:
        img = self._decode_image(image_bytes)
        if img is None:
            return []
        return self._detect_decoded(img, include_pose)

    def _detect_decoded(self, img: np.ndarray, include_pose: bool) -> list[DetectedFace]:
        bboxes, kpss, working, dx, dy = self._detect_with_pad_fallback(img)
        if bboxes.shape[0] == 0:
            return []

        poses: list[HeadPose | None] = (
            self._estimate_poses(working, bboxes, kpss) if include_pose else [None] * bboxes.shape[0]
        )
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

        demographics: list[tuple[float | None, str | None]]
        if ga_model is not None:
            demographics = self._genderage_for_image(ga_model, working, bboxes, kpss)
        else:
            demographics = [(None, None)] * bboxes.shape[0]

        orig_h, orig_w = img.shape[:2]
        return [
            DetectedFace(
                bbox=self._make_bbox(bboxes[i, :4], dx, dy, orig_w, orig_h),
                det_score=float(bboxes[i, 4]),
                embedding=embeddings[i].astype(np.float32).tolist(),
                age=demographics[i][0],
                gender=demographics[i][1],
                landmarks=self._kps_to_landmarks(kpss[i], dx, dy),
            )
            for i in range(bboxes.shape[0])
        ]

    def detect_batch(self, images: list[bytes], include_pose: bool = False) -> list[list[DetectedFace]]:
        # Stage 1: Decode images in parallel (cv2.imdecode releases the GIL)
        decoded = self._cv_pool.decode_batch(self._decode_image, images)

        # Stage 2: One batched detection pass, plus one batched pad-retry pass
        # over the zero-face subset.
        per_image = self._detect_with_pad_fallback_batch(decoded)

        results: list[list[DetectedFace]] = []
        for bboxes, kpss, working, dx, dy, orig_h, orig_w in per_image:
            if bboxes.shape[0] == 0 or working is None:
                results.append([])
                continue
            poses: list[HeadPose | None] = (
                self._estimate_poses(working, bboxes, kpss) if include_pose else [None] * bboxes.shape[0]
            )
            results.append(
                [
                    DetectedFace(
                        bbox=self._make_bbox(bboxes[i, :4], dx, dy, orig_w, orig_h),
                        det_score=float(bboxes[i, 4]),
                        landmarks=self._kps_to_landmarks(kpss[i] if kpss is not None else None, dx, dy),
                        pose=poses[i],
                    )
                    for i in range(bboxes.shape[0])
                ]
            )
        return results

    def embed_batch(self, images: list[bytes]) -> list[list[DetectedFace]]:
        rec_model = self._app.models["recognition"]
        image_size = rec_model.input_size[0]

        # Stage 1: Decode images in parallel (cv2.imdecode releases the GIL)
        decoded = self._cv_pool.decode_batch(self._decode_image, images)

        # Stage 2: One batched detection pass, plus one batched pad-retry pass
        # over the zero-face subset.
        per_image = self._detect_with_pad_fallback_batch(decoded)

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
            all_crops = self._cv_pool.map(lambda t: _norm_crop(t[0], t[1], image_size), align_tasks)

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
        decoded = self._cv_pool.decode_batch(self._decode_image, images)

        # Stage 2: One batched detection pass, plus one batched pad-retry pass
        # over the zero-face subset.
        per_image = self._detect_with_pad_fallback_batch(decoded)

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
            all_crops = self._cv_pool.map(lambda t: _norm_crop(t[0], t[1], image_size), align_tasks)

        if all_crops:
            all_embeddings: np.ndarray = self._recognize(rec_model, all_crops)
            norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            all_embeddings = all_embeddings / norms
        else:
            all_embeddings = np.zeros((0, 512), dtype=np.float32)

        # Genderage across ALL faces of the request in one batched pass (one
        # session.run per chunk instead of one per face); falls back to
        # per-image Attribute.get when the graph can't batch. Task order
        # mirrors the crop/embedding order, so emb_offset indexes both.
        demographics: list[tuple[float | None, str | None]] = []
        if ga_model is not None and all_crops:
            if self._ga_batch_capable(ga_model):
                ga_tasks: list[tuple[np.ndarray, np.ndarray]] = []
                for idx, it in enumerate(per_image):
                    it_bboxes, it_working = it[0], it[2]
                    if crop_counts[idx] and it_working is not None:
                        ga_tasks.extend((it_working, it_bboxes[i, :4]) for i in range(crop_counts[idx]))
                demographics = self._genderage_batch(ga_model, ga_tasks)
            else:
                for idx, it in enumerate(per_image):
                    it_bboxes, it_kpss, it_working = it[0], it[1], it[2]
                    if crop_counts[idx] and it_working is not None:
                        demographics.extend(
                            self._genderage_for_image(ga_model, it_working, it_bboxes[: crop_counts[idx]], it_kpss)
                        )
        else:
            demographics = [(None, None)] * len(all_crops)

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
                        age=demographics[emb_offset + i][0],
                        gender=demographics[emb_offset + i][1],
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
