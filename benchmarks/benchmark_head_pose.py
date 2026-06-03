#!/usr/bin/env python3
"""Benchmark head-pose accuracy: InsightFace 1k3d68 pose vs our old 5-landmark hack.

Runs the real buffalo_l pipeline over a head-pose dataset with ground-truth
yaw/pitch/roll (AFLW2000-3D) and reports per-axis MAE for:
  - InsightFace pose (face.pose, populated by the 1k3d68 model)
  - the legacy 5-landmark geometric estimator we used in liveness (yaw from the
    nose-vs-eye-midpoint offset, pitch from the nose's vertical fraction)

For the legacy estimator (which outputs normalized units, not degrees) we fit the
best linear map to ground-truth degrees and report the residual MAE — i.e. its
best-case accuracy. For InsightFace we also report the sign per axis that best
matches the dataset convention (this is what tells us how to set the liveness signs).

Dataset: AFLW2000-3D (http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/).
Unzip so that `<root>/imageNNNNN.jpg` and `<root>/imageNNNNN.mat` sit side by side.

Usage:
    uv run --with scipy python benchmarks/benchmark_head_pose.py --root benchmarks/data/AFLW2000
    uv run --with scipy python benchmarks/benchmark_head_pose.py --root <dir> --limit 200 --cpu
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from glob import glob

import cv2
import numpy as np

# Reuse the model loader (ORT session-options patch + provider selection).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import load_model  # noqa: E402


def _gt_pose_degrees(mat_path: str) -> tuple[float, float, float] | None:
    """AFLW2000-3D Pose_Para = [pitch, yaw, roll, ...] in radians."""
    from scipy.io import loadmat  # type: ignore[import-untyped]

    mat = loadmat(mat_path)
    if "Pose_Para" not in mat:
        return None
    p = mat["Pose_Para"][0]
    pitch, yaw, roll = (math.degrees(float(p[i])) for i in range(3))
    return pitch, yaw, roll


def _legacy_estimator(kps: np.ndarray) -> tuple[float, float]:
    """Our old liveness hack from the 5 keypoints: returns (yaw_norm, pitch_norm)."""
    left_eye, right_eye, nose, left_mouth, right_mouth = kps[:5]
    inter_eye = abs(right_eye[0] - left_eye[0]) or 1e-6
    yaw = (nose[0] - (left_eye[0] + right_eye[0]) / 2) / inter_eye
    eye_y = (left_eye[1] + right_eye[1]) / 2
    mouth_y = (left_mouth[1] + right_mouth[1]) / 2
    span = (mouth_y - eye_y) or 1e-6
    nose_frac = (nose[1] - eye_y) / span
    pitch = 0.62 - nose_frac
    return float(yaw), float(pitch)


def _mae(errs: list[float]) -> float:
    return float(np.mean(np.abs(errs))) if errs else float("nan")


def _best_sign_mae(pred: list[float], gt: list[float]) -> tuple[float, int]:
    a = np.array(pred)
    g = np.array(gt)
    pos = float(np.mean(np.abs(a - g)))
    neg = float(np.mean(np.abs(-a - g)))
    return (pos, 1) if pos <= neg else (neg, -1)


def _linear_fit_mae(pred: list[float], gt: list[float]) -> float:
    """Best linear map pred->gt (a*pred+b), residual MAE. Legacy estimator's best case."""
    a = np.array(pred)
    g = np.array(gt)
    coeffs = np.linalg.lstsq(np.vstack([a, np.ones_like(a)]).T, g, rcond=None)[0]
    fitted = coeffs[0] * a + coeffs[1]
    return float(np.mean(np.abs(fitted - g)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Head-pose accuracy benchmark (InsightFace vs legacy hack)")
    parser.add_argument("--root", required=True, help="AFLW2000-3D dir with imageNNNNN.jpg + .mat")
    parser.add_argument("--limit", type=int, default=0, help="Max images (0 = all)")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--model-name", default="buffalo_l")
    parser.add_argument("--model-dir", default=os.path.expanduser("~/.insightface"))
    parser.add_argument("--max-angle", type=float, default=99.0, help="Skip images with any |GT angle| above this")
    args = parser.parse_args()

    use_gpu = args.gpu and not args.cpu
    print(f"Loading {args.model_name} ({'GPU' if use_gpu else 'CPU'})...")
    app = load_model(use_gpu, args.model_name, (640, 640), args.model_dir)
    if app.models.get("landmark_3d_68") is None:
        print("ERROR: landmark_3d_68 (pose) model not loaded — buffalo_l pack missing 1k3d68.onnx?", file=sys.stderr)
        sys.exit(1)

    images = sorted(glob(os.path.join(args.root, "*.jpg")))
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"ERROR: no .jpg images under {args.root}", file=sys.stderr)
        sys.exit(1)
    print(f"Running over {len(images)} images...\n")

    axes = ("pitch", "yaw", "roll")
    if_pred: dict[str, list[float]] = {ax: [] for ax in axes}
    gt_vals: dict[str, list[float]] = {ax: [] for ax in axes}
    legacy_pred = {"yaw": [], "pitch": []}  # type: dict[str, list[float]]
    legacy_gt = {"yaw": [], "pitch": []}  # type: dict[str, list[float]]
    no_face = 0
    skipped = 0

    for n, img_path in enumerate(images):
        mat_path = img_path[:-4] + ".mat"
        if not os.path.isfile(mat_path):
            continue
        gt = _gt_pose_degrees(mat_path)
        if gt is None or any(abs(v) > args.max_angle for v in gt):
            skipped += 1
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        faces = app.get(img)
        if not faces:
            no_face += 1
            continue
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        pose = getattr(face, "pose", None)
        if pose is None:
            no_face += 1
            continue
        for i, ax in enumerate(axes):
            if_pred[ax].append(float(pose[i]))
            gt_vals[ax].append(gt[i])
        ly, lp = _legacy_estimator(np.asarray(face.kps))
        legacy_pred["yaw"].append(ly)
        legacy_gt["yaw"].append(gt[1])
        legacy_pred["pitch"].append(lp)
        legacy_gt["pitch"].append(gt[0])
        if (n + 1) % 200 == 0:
            print(f"  ...{n + 1}/{len(images)}")

    used = len(if_pred["yaw"])
    print(f"\nUsed {used} faces (no-face: {no_face}, skipped-by-angle: {skipped})\n")
    if used == 0:
        sys.exit(1)

    print(f"{'axis':<7}{'GT std':>9}{'InsightFace MAE':>18}{'sign':>6}{'legacy MAE (fit)':>20}")
    print("-" * 60)
    for ax in axes:
        gt_std = float(np.std(gt_vals[ax]))
        if_mae, sign = _best_sign_mae(if_pred[ax], gt_vals[ax])
        legacy = ""
        if ax in legacy_pred:
            legacy = f"{_linear_fit_mae(legacy_pred[ax], legacy_gt[ax]):>18.2f}°"
        print(f"{ax:<7}{gt_std:>8.1f}°{if_mae:>16.2f}°{sign:>6}{legacy:>20}")
    print(
        "\nLower MAE = better. The 'sign' column is the multiplier that aligns InsightFace's "
        "axis to the dataset — use it to set the liveness yaw/pitch signs."
    )


if __name__ == "__main__":
    main()
