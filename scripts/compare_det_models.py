"""Benchmark batched SCRFD models against original on real face images.

Downloads WIDER FACE validation subset, runs detection with original (single)
as ground truth, then compares immich and alonsorobots batched models.

Checks:
1. Raw output parity: all 9 output tensors must be identical at batch=1
2. Cross-frame contamination: outputs must not change when batch neighbors change
3. Detection parity: final bboxes/scores/kps after postprocessing must match
4. Batch size sweep: verify parity holds at batch=1,2,4,8,16

Usage:
    uv run python scripts/compare_det_models.py [--gpu] [--num-images 50]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


MODELS = {
    "original": {
        "repo": None,
        "local_path": "~/.insightface/models/buffalo_l/det_10g.onnx",
    },
    "immich": {
        "repo": "immich-app/buffalo_l_batch",
        "filename": "detection/model.onnx",
        "local_dir": "models_compare/immich",
    },
    "alonsorobots": {
        "repo": "alonsorobots/scrfd_320_batched",
        "filename": "scrfd_10g_320_batch.onnx",
        "local_dir": "models_compare/alonsorobots",
    },
    "ours_640": {
        "repo": None,
        "local_dir": "models_compare",
        "filename": "scrfd_10g_640_batch.onnx",
    },
}

STRIDES = [8, 16, 32]
NUM_ANCHORS = 2
FMC = 3


def download_models(base_dir: Path) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    paths: dict[str, Path] = {}
    for name, info in MODELS.items():
        if info.get("repo") is None and "local_path" in info:
            p = Path(os.path.expanduser(info["local_path"]))
            if not p.exists():
                print(f"  [{name}] NOT FOUND at {p}")
                continue
            paths[name] = p
        elif info.get("repo") is None and "local_dir" in info:
            p = base_dir / info["local_dir"] / info["filename"]
            if not p.exists():
                print(f"  [{name}] NOT FOUND at {p}")
                continue
            paths[name] = p
        else:
            local_dir = base_dir / info["local_dir"]
            local_dir.mkdir(parents=True, exist_ok=True)
            expected = local_dir / info["filename"]
            if expected.exists():
                paths[name] = expected
            else:
                print(f"  [{name}] downloading from {info['repo']}...")
                result = hf_hub_download(
                    repo_id=info["repo"],
                    filename=info["filename"],
                    local_dir=str(local_dir),
                )
                paths[name] = Path(result)
    return paths


def load_wider_face_images(num_images: int, cache_dir: Path) -> list[np.ndarray]:
    img_dir = cache_dir / "wider_face_images"
    if img_dir.exists() and len(list(img_dir.glob("*.jpg"))) >= num_images:
        print(f"  Loading cached images from {img_dir}")
        paths = sorted(img_dir.glob("*.jpg"))[:num_images]
        return [cv2.imread(str(p)) for p in paths if cv2.imread(str(p)) is not None]

    print(f"  Downloading WIDER FACE validation images...")
    import io
    import zipfile
    from urllib.request import urlopen

    url = "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip"
    print(f"  Fetching {url} ...")
    resp = urlopen(url)  # noqa: S310
    data = resp.read()
    print(f"  Downloaded {len(data) / 1e6:.0f} MB, extracting...")

    extract_dir = cache_dir / "wider_val_raw"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(extract_dir)

    all_jpgs = sorted(extract_dir.rglob("*.jpg"))
    print(f"  Found {len(all_jpgs)} images total")

    rng = np.random.default_rng(42)
    selected = rng.choice(len(all_jpgs), size=min(num_images, len(all_jpgs)), replace=False)
    selected.sort()

    img_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for i, idx in enumerate(selected):
        img = cv2.imread(str(all_jpgs[int(idx)]))
        if img is not None:
            cv2.imwrite(str(img_dir / f"{i:04d}.jpg"), img)
            images.append(img)

    print(f"  Cached {len(images)} images to {img_dir}")
    return images


def load_session(path: Path, providers: list[str]) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=providers)


def get_model_det_size(session: ort.InferenceSession) -> tuple[int, int] | None:
    shape = session.get_inputs()[0].shape
    if isinstance(shape[2], int) and isinstance(shape[3], int):
        return (shape[2], shape[3])
    return None


def is_batched(session: ort.InferenceSession) -> bool:
    return len(session.get_outputs()[0].shape) == 3


def preprocess(img: np.ndarray, det_size: tuple[int, int]) -> tuple[np.ndarray, float]:
    im_ratio = float(img.shape[0]) / img.shape[1]
    model_ratio = float(det_size[1]) / det_size[0]
    if im_ratio > model_ratio:
        new_height = det_size[1]
        new_width = int(new_height / im_ratio)
    else:
        new_width = det_size[0]
        new_height = int(new_width * im_ratio)
    det_scale = float(new_height) / img.shape[0]
    resized = cv2.resize(img, (new_width, new_height))
    det_img = np.zeros((det_size[1], det_size[0], 3), dtype=np.uint8)
    det_img[:new_height, :new_width, :] = resized
    return det_img, det_scale


def make_blob_single(det_img: np.ndarray) -> np.ndarray:
    return cv2.dnn.blobFromImage(
        det_img, 1.0 / 128.0, tuple(det_img.shape[:2][::-1]),
        (127.5, 127.5, 127.5), swapRB=True,
    )


def make_blob_batch(det_imgs: list[np.ndarray]) -> np.ndarray:
    return cv2.dnn.blobFromImages(
        det_imgs, 1.0 / 128.0, tuple(det_imgs[0].shape[:2][::-1]),
        (127.5, 127.5, 127.5), swapRB=True,
    )


def run_inference(session: ort.InferenceSession, blob: np.ndarray) -> list[np.ndarray]:
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: blob})


def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def nms(dets: np.ndarray, thresh: float = 0.4) -> list[int]:
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
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
        ovr = (w * h) / (areas[i] + areas[order[1:]] - w * h)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def postprocess_single(
    raw_outs: list[np.ndarray], det_scale: float, input_h: int, input_w: int,
    batched_model: bool, det_thresh: float = 0.5,
) -> tuple[np.ndarray, np.ndarray | None]:
    scores_list, bboxes_list, kpss_list = [], [], []
    center_cache: dict[tuple[int, int, int], np.ndarray] = {}

    for idx, stride in enumerate(STRIDES):
        if batched_model:
            scores = raw_outs[idx][0]
            bbox_preds = raw_outs[idx + FMC][0] * stride
            kps_preds = raw_outs[idx + FMC * 2][0] * stride
        else:
            scores = raw_outs[idx]
            bbox_preds = raw_outs[idx + FMC] * stride
            kps_preds = raw_outs[idx + FMC * 2] * stride

        height = input_h // stride
        width = input_w // stride
        key = (height, width, stride)
        if key in center_cache:
            anchor_centers = center_cache[key]
        else:
            anchor_centers = np.stack(
                np.mgrid[:height, :width][::-1], axis=-1
            ).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if NUM_ANCHORS > 1:
                anchor_centers = np.stack(
                    [anchor_centers] * NUM_ANCHORS, axis=1
                ).reshape((-1, 2))
            center_cache[key] = anchor_centers

        pos_inds = np.where(scores >= det_thresh)[0]
        bboxes = distance2bbox(anchor_centers, bbox_preds)
        pos_scores = scores[pos_inds]
        pos_bboxes = bboxes[pos_inds]
        scores_list.append(pos_scores)
        bboxes_list.append(pos_bboxes)

        kpss = distance2kps(anchor_centers, kps_preds)
        kpss = kpss.reshape((kpss.shape[0], -1, 2))
        kpss_list.append(kpss[pos_inds])

    scores_arr = np.vstack(scores_list)
    scores_ravel = scores_arr.ravel()
    order = scores_ravel.argsort()[::-1]
    bboxes_arr = np.vstack(bboxes_list) / det_scale
    kpss_arr = np.vstack(kpss_list) / det_scale
    pre_det = np.hstack((bboxes_arr, scores_arr)).astype(np.float32, copy=False)
    pre_det = pre_det[order, :]
    keep = nms(pre_det)
    det = pre_det[keep, :]
    kpss_arr = kpss_arr[order, :, :]
    kpss_arr = kpss_arr[keep, :, :]
    return det, kpss_arr


def detect_single(
    session: ort.InferenceSession, img: np.ndarray,
    det_size: tuple[int, int], batched_model: bool,
) -> tuple[np.ndarray, np.ndarray | None, list[np.ndarray]]:
    det_img, det_scale = preprocess(img, det_size)
    blob = make_blob_single(det_img)
    raw_outs = run_inference(session, blob)
    det, kpss = postprocess_single(
        raw_outs, det_scale, det_size[1], det_size[0], batched_model,
    )
    return det, kpss, raw_outs


def test_raw_output_parity(
    original_session: ort.InferenceSession,
    candidate_session: ort.InferenceSession,
    candidate_name: str,
    images: list[np.ndarray],
    det_size: tuple[int, int],
    candidate_batched: bool,
) -> None:
    print(f"\n{'='*60}")
    print(f"TEST 1: Raw output parity — original vs {candidate_name} (batch=1)")
    print(f"{'='*60}")

    max_diffs = [0.0] * 9
    for i, img in enumerate(images):
        det_img, _ = preprocess(img, det_size)
        blob = make_blob_single(det_img)

        orig_outs = run_inference(original_session, blob)
        cand_outs = run_inference(candidate_session, blob)

        for j in range(9):
            orig = orig_outs[j]
            cand = cand_outs[j][0] if candidate_batched else cand_outs[j]
            diff = float(np.max(np.abs(orig - cand)))
            max_diffs[j] = max(max_diffs[j], diff)

    output_names = [
        "score_8", "score_16", "score_32",
        "bbox_8", "bbox_16", "bbox_32",
        "kps_8", "kps_16", "kps_32",
    ]
    all_zero = True
    for j, name in enumerate(output_names):
        status = "PASS" if max_diffs[j] < 1e-6 else ("WARN" if max_diffs[j] < 1e-4 else "FAIL")
        if max_diffs[j] > 0:
            all_zero = False
        print(f"  {name:<10} max_diff={max_diffs[j]:.10f}  [{status}]")

    overall_max = max(max_diffs)
    if overall_max == 0.0:
        print(f"\n  RESULT: IDENTICAL — all outputs match exactly across {len(images)} images")
    elif overall_max < 1e-6:
        print(f"\n  RESULT: PASS — max diff {overall_max:.2e} (float32 noise)")
    elif overall_max < 1e-4:
        print(f"\n  RESULT: WARN — max diff {overall_max:.2e}")
    else:
        print(f"\n  RESULT: FAIL — max diff {overall_max:.2e}")


def test_contamination(
    candidate_session: ort.InferenceSession,
    candidate_name: str,
    images: list[np.ndarray],
    det_size: tuple[int, int],
    candidate_batched: bool,
    batch_sizes: list[int],
) -> None:
    print(f"\n{'='*60}")
    print(f"TEST 2: Cross-frame contamination — {candidate_name}")
    print(f"{'='*60}")

    det_imgs = [preprocess(img, det_size)[0] for img in images]

    ref_outputs: list[list[np.ndarray]] = []
    for det_img in det_imgs:
        blob = make_blob_single(det_img)
        outs = run_inference(candidate_session, blob)
        ref_outputs.append(outs)

    rng = np.random.default_rng(123)

    for bs in batch_sizes:
        if bs > len(det_imgs):
            continue

        max_diff_all = 0.0
        num_tests = 0
        max_diff_per_pos: dict[int, float] = {}

        for trial in range(min(20, len(det_imgs) // bs)):
            indices = rng.choice(len(det_imgs), size=bs, replace=False).tolist()
            batch = [det_imgs[j] for j in indices]
            blob = make_blob_batch(batch)
            batch_outs = run_inference(candidate_session, blob)

            for pos, target_idx in enumerate(indices):
                for j in range(9):
                    ref = ref_outputs[target_idx][j]
                    if candidate_batched:
                        ref_data = ref[0]
                        batch_data = batch_outs[j][pos]
                    else:
                        ref_data = ref
                        batch_data = batch_outs[j]
                    diff = float(np.max(np.abs(ref_data - batch_data)))
                    max_diff_all = max(max_diff_all, diff)
                    max_diff_per_pos[pos] = max(max_diff_per_pos.get(pos, 0.0), diff)
                num_tests += 1

        status = "PASS" if max_diff_all < 1e-6 else ("WARN" if max_diff_all < 1e-4 else "FAIL")
        pos_detail = " ".join(f"pos{p}={d:.2e}" for p, d in sorted(max_diff_per_pos.items()))
        print(f"  batch={bs:<3} max_diff={max_diff_all:.10f}  tests={num_tests}  [{status}]")
        if max_diff_all > 1e-6:
            print(f"         per-position: {pos_detail}")


def test_detection_parity(
    original_session: ort.InferenceSession,
    candidate_session: ort.InferenceSession,
    candidate_name: str,
    images: list[np.ndarray],
    det_size: tuple[int, int],
    candidate_batched: bool,
) -> None:
    print(f"\n{'='*60}")
    print(f"TEST 3: Detection parity — original vs {candidate_name}")
    print(f"{'='*60}")

    total_faces_orig = 0
    total_faces_cand = 0
    face_count_match = 0
    bbox_diffs: list[float] = []
    score_diffs: list[float] = []
    kps_diffs: list[float] = []

    for img in images:
        det_o, kps_o, _ = detect_single(original_session, img, det_size, False)
        det_c, kps_c, _ = detect_single(candidate_session, img, det_size, candidate_batched)

        n_orig = det_o.shape[0]
        n_cand = det_c.shape[0]
        total_faces_orig += n_orig
        total_faces_cand += n_cand

        if n_orig == n_cand:
            face_count_match += 1

        n_compare = min(n_orig, n_cand)
        if n_compare > 0:
            bbox_diff = float(np.max(np.abs(det_o[:n_compare, :4] - det_c[:n_compare, :4])))
            score_diff = float(np.max(np.abs(det_o[:n_compare, 4] - det_c[:n_compare, 4])))
            bbox_diffs.append(bbox_diff)
            score_diffs.append(score_diff)

            if kps_o is not None and kps_c is not None:
                kps_diff = float(np.max(np.abs(kps_o[:n_compare] - kps_c[:n_compare])))
                kps_diffs.append(kps_diff)

    n = len(images)
    print(f"  Images tested: {n}")
    print(f"  Face count match: {face_count_match}/{n}")
    print(f"  Total faces: original={total_faces_orig}, {candidate_name}={total_faces_cand}")

    if bbox_diffs:
        max_bbox = max(bbox_diffs)
        max_score = max(score_diffs)
        status_b = "PASS" if max_bbox < 0.01 else ("WARN" if max_bbox < 1.0 else "FAIL")
        status_s = "PASS" if max_score < 1e-6 else ("WARN" if max_score < 1e-4 else "FAIL")
        print(f"  Max bbox diff: {max_bbox:.6f} px  [{status_b}]")
        print(f"  Max score diff: {max_score:.10f}  [{status_s}]")
    if kps_diffs:
        max_kps = max(kps_diffs)
        status_k = "PASS" if max_kps < 0.01 else ("WARN" if max_kps < 1.0 else "FAIL")
        print(f"  Max kps diff: {max_kps:.6f} px  [{status_k}]")

    if bbox_diffs and max(bbox_diffs) == 0.0 and max(score_diffs) == 0.0:
        print(f"\n  RESULT: IDENTICAL detections across {n} images")
    elif bbox_diffs and max(bbox_diffs) < 0.01:
        print(f"\n  RESULT: PASS — negligible differences")
    else:
        print(f"\n  RESULT: CHECK — differences found")


def test_batch_detection_parity(
    candidate_session: ort.InferenceSession,
    candidate_name: str,
    images: list[np.ndarray],
    det_size: tuple[int, int],
    candidate_batched: bool,
    batch_sizes: list[int],
) -> None:
    print(f"\n{'='*60}")
    print(f"TEST 4: Batch detection parity — {candidate_name} single vs batched")
    print(f"{'='*60}")

    single_results: list[tuple[np.ndarray, np.ndarray | None]] = []
    for img in images:
        det, kps, _ = detect_single(candidate_session, img, det_size, candidate_batched)
        single_results.append((det, kps))

    rng = np.random.default_rng(456)

    for bs in batch_sizes:
        if bs > len(images):
            continue

        max_bbox_diff = 0.0
        max_score_diff = 0.0
        face_mismatches = 0
        num_tests = 0

        for _ in range(min(20, len(images))):
            indices = rng.choice(len(images), size=bs, replace=False).tolist()
            det_imgs = [preprocess(images[idx], det_size)[0] for idx in indices]
            det_scales = [preprocess(images[idx], det_size)[1] for idx in indices]

            blob = make_blob_batch(det_imgs)
            batch_outs = run_inference(candidate_session, blob)

            for pos, idx in enumerate(indices):
                per_image_outs = []
                for j in range(9):
                    per_image_outs.append(batch_outs[j][pos:pos + 1])

                det_b, kps_b = postprocess_single(
                    per_image_outs, det_scales[pos],
                    det_size[1], det_size[0], True,
                )
                det_s, kps_s = single_results[idx]

                if det_b.shape[0] != det_s.shape[0]:
                    face_mismatches += 1
                else:
                    n = det_s.shape[0]
                    if n > 0:
                        bd = float(np.max(np.abs(det_b[:n, :4] - det_s[:n, :4])))
                        sd = float(np.max(np.abs(det_b[:n, 4] - det_s[:n, 4])))
                        max_bbox_diff = max(max_bbox_diff, bd)
                        max_score_diff = max(max_score_diff, sd)

                num_tests += 1

        status = "PASS" if max_bbox_diff < 0.01 and face_mismatches == 0 else "FAIL"
        print(
            f"  batch={bs:<3} max_bbox_diff={max_bbox_diff:.6f}px "
            f"max_score_diff={max_score_diff:.10f} "
            f"face_mismatches={face_mismatches}/{num_tests}  [{status}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark batched SCRFD models")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--num-images", type=int, default=50)
    args = parser.parse_args()

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.gpu
        else ["CPUExecutionProvider"]
    )

    base_dir = Path(__file__).parent.parent
    print("Downloading models...")
    model_paths = download_models(base_dir)

    if "original" not in model_paths:
        print("ERROR: original det_10g.onnx not found")
        sys.exit(1)

    print(f"\nLoading {args.num_images} WIDER FACE validation images...")
    images = load_wider_face_images(args.num_images, base_dir / "models_compare")

    print(f"\nLoaded {len(images)} images, shapes: {[img.shape for img in images[:3]]}...")

    original_session = load_session(model_paths["original"], providers)
    orig_det_size = get_model_det_size(original_session)
    if orig_det_size is None:
        orig_det_size = (640, 640)
    print(f"\nOriginal model det_size: {orig_det_size}")

    for cand_name in ["immich", "alonsorobots", "ours_640"]:
        if cand_name not in model_paths:
            print(f"\nSkipping {cand_name} (not downloaded)")
            continue

        cand_session = load_session(model_paths[cand_name], providers)
        cand_det_size = get_model_det_size(cand_session)
        cand_batched = is_batched(cand_session)

        print(f"\n{'#'*60}")
        print(f"# CANDIDATE: {cand_name}")
        print(f"# det_size={cand_det_size}, batched={cand_batched}")
        print(f"{'#'*60}")

        if cand_det_size != orig_det_size:
            print(f"\n  WARNING: det_size mismatch ({cand_det_size} vs {orig_det_size})")
            print(f"  Skipping raw output parity (different resolutions)")
            print(f"  Running contamination test at native {cand_det_size}...")

            test_contamination(
                cand_session, cand_name, images,
                cand_det_size, cand_batched,
                batch_sizes=[2, 4, 8, 16],
            )
            continue

        test_raw_output_parity(
            original_session, cand_session, cand_name,
            images, orig_det_size, cand_batched,
        )

        test_contamination(
            cand_session, cand_name, images,
            orig_det_size, cand_batched,
            batch_sizes=[2, 4, 8, 16],
        )

        test_detection_parity(
            original_session, cand_session, cand_name,
            images, orig_det_size, cand_batched,
        )

        test_batch_detection_parity(
            cand_session, cand_name, images,
            orig_det_size, cand_batched,
            batch_sizes=[2, 4, 8, 16],
        )

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
