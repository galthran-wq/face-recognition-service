from __future__ import annotations

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]

_STRIDES = [8, 16, 32]
_NUM_ANCHORS = 2
_FMC = 3


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def _nms(dets: np.ndarray, thresh: float) -> list[int]:
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


class BatchedSCRFD:
    def __init__(
        self,
        model_path: str,
        *,
        det_size: tuple[int, int] = (640, 640),
        max_batch: int = 32,
        det_thresh: float = 0.5,
        nms_thresh: float = 0.4,
        providers: list[str] | None = None,
    ) -> None:
        self._det_size = det_size
        self._max_batch = max_batch
        self._det_thresh = det_thresh
        self._nms_thresh = nms_thresh
        self._input_mean = 127.5
        self._input_std = 128.0

        self._session = ort.InferenceSession(model_path, providers=providers or ["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

        self._anchor_cache: dict[tuple[int, int, int], np.ndarray] = {}

    def _get_anchors(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        if key in self._anchor_cache:
            return self._anchor_cache[key]

        anchor_centers: np.ndarray = np.stack(
            np.mgrid[:height, :width][::-1], axis=-1  # type: ignore[call-overload]
        ).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        if _NUM_ANCHORS > 1:
            anchor_centers = np.stack([anchor_centers] * _NUM_ANCHORS, axis=1).reshape((-1, 2))

        self._anchor_cache[key] = anchor_centers
        return anchor_centers  # noqa: RET504

    def _preprocess(self, images: list[np.ndarray]) -> tuple[np.ndarray, list[float]]:
        import cv2

        input_h, input_w = self._det_size[1], self._det_size[0]
        det_scales: list[float] = []
        resized_imgs: list[np.ndarray] = []

        for img in images:
            im_ratio = float(img.shape[0]) / img.shape[1]
            model_ratio = float(input_h) / input_w
            if im_ratio > model_ratio:
                new_height = input_h
                new_width = int(new_height / im_ratio)
            else:
                new_width = input_w
                new_height = int(new_width * im_ratio)
            det_scale = float(new_height) / img.shape[0]
            det_scales.append(det_scale)
            resized = cv2.resize(img, (new_width, new_height))
            det_img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
            det_img[:new_height, :new_width, :] = resized
            resized_imgs.append(det_img)

        blob: np.ndarray = cv2.dnn.blobFromImages(
            resized_imgs,
            1.0 / self._input_std,
            (input_w, input_h),
            (self._input_mean, self._input_mean, self._input_mean),
            swapRB=True,
        )
        return blob, det_scales

    def _postprocess_single(
        self,
        net_outs: list[np.ndarray],
        batch_idx: int,
        det_scale: float,
        input_height: int,
        input_width: int,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        scores_list: list[np.ndarray] = []
        bboxes_list: list[np.ndarray] = []
        kpss_list: list[np.ndarray] = []

        for idx, stride in enumerate(_STRIDES):
            scores = net_outs[idx][batch_idx]
            bbox_preds = net_outs[idx + _FMC][batch_idx] * stride
            kps_preds = net_outs[idx + _FMC * 2][batch_idx] * stride

            height = input_height // stride
            width = input_width // stride
            anchor_centers = self._get_anchors(height, width, stride)

            pos_inds = np.where(scores >= self._det_thresh)[0]
            bboxes = _distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            kpss = _distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

        if not scores_list or all(s.shape[0] == 0 for s in scores_list):
            return np.zeros((0, 5), dtype=np.float32), None

        scores_arr = np.vstack(scores_list)
        scores_ravel = scores_arr.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes_arr = np.vstack(bboxes_list) / det_scale
        kpss_arr = np.vstack(kpss_list) / det_scale

        pre_det = np.hstack((bboxes_arr, scores_arr)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = _nms(pre_det, self._nms_thresh)
        det = pre_det[keep, :]

        kpss_arr = kpss_arr[order, :, :]
        kpss_arr = kpss_arr[keep, :, :]

        return det, kpss_arr if kpss_arr.shape[0] > 0 else None

    def detect_batch(
        self,
        images: list[np.ndarray],
        max_num: int = 0,
    ) -> list[tuple[np.ndarray, np.ndarray | None]]:
        if not images:
            return []

        results: list[tuple[np.ndarray, np.ndarray | None]] = []

        for start in range(0, len(images), self._max_batch):
            chunk = images[start : start + self._max_batch]
            blob, det_scales = self._preprocess(chunk)
            input_height = blob.shape[2]
            input_width = blob.shape[3]

            net_outs = self._session.run(self._output_names, {self._input_name: blob})

            for b in range(len(chunk)):
                det, kpss = self._postprocess_single(net_outs, b, det_scales[b], input_height, input_width)

                if max_num > 0 and det.shape[0] > max_num:
                    area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                    img_center = (chunk[b].shape[0] // 2, chunk[b].shape[1] // 2)
                    offsets = np.vstack(
                        [
                            (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                            (det[:, 1] + det[:, 3]) / 2 - img_center[0],
                        ]
                    )
                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                    values = area - offset_dist_squared * 2.0
                    bindex = np.argsort(values)[::-1][:max_num]
                    det = det[bindex, :]
                    if kpss is not None:
                        kpss = kpss[bindex, :]

                results.append((det, kpss))

        return results
