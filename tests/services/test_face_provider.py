import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.services.face_provider.insightface import InsightFaceProvider


def _make_det_output(
    faces: list[dict[str, object]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (bboxes, kpss) arrays mimicking det_model.detect() output."""
    if faces is None:
        faces = [{"bbox": [10.0, 20.0, 110.0, 140.0], "det_score": 0.95}]

    n = len(faces)
    bboxes = np.zeros((n, 5), dtype=np.float32)
    kpss = np.zeros((n, 5, 2), dtype=np.float32)
    for i, f in enumerate(faces):
        bbox = f["bbox"]
        bboxes[i, :4] = bbox  # type: ignore[index]
        bboxes[i, 4] = f.get("det_score", 0.95)  # type: ignore[arg-type]
        # Fake 5 keypoints inside the bbox
        x1, y1, x2, y2 = bbox  # type: ignore[misc]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2  # type: ignore[operator]
        kpss[i] = [[cx - 10, cy - 10], [cx + 10, cy - 10], [cx, cy], [cx - 10, cy + 10], [cx + 10, cy + 10]]
    return bboxes, kpss


def _create_provider_with_mock() -> tuple[InsightFaceProvider, MagicMock]:
    provider = InsightFaceProvider(use_gpu=False, det_size=(640, 640))
    mock_app = MagicMock()

    # Set up det_model mock
    mock_app.det_model.detect.return_value = _make_det_output()

    # Set up recognition model mock
    mock_rec = MagicMock()
    mock_rec.input_size = (112, 112)
    mock_rec.get_feat.return_value = np.random.randn(1, 512).astype(np.float32)
    mock_app.models = {"recognition": mock_rec}

    provider._app = mock_app
    provider._loaded = True
    return provider, mock_app


def _fake_image_bytes() -> bytes:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    import cv2  # type: ignore[import-untyped]

    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


class TestInsightFaceProviderDetect:
    def test_detect_returns_bbox_and_score(self) -> None:
        provider, mock_app = _create_provider_with_mock()

        faces = provider.detect(_fake_image_bytes())

        assert len(faces) == 1
        assert faces[0].bbox.x == 10.0
        assert faces[0].bbox.y == 20.0
        assert faces[0].bbox.width == 100.0
        assert faces[0].bbox.height == 120.0
        assert faces[0].det_score == pytest.approx(0.95)
        assert faces[0].embedding is None
        assert faces[0].age is None
        assert faces[0].gender is None

    def test_detect_empty_image(self) -> None:
        provider, _mock_app = _create_provider_with_mock()
        faces = provider.detect(b"not-an-image")
        assert faces == []

    def test_detect_no_faces(self) -> None:
        provider, mock_app = _create_provider_with_mock()
        mock_app.det_model.detect.return_value = (np.zeros((0, 5), dtype=np.float32), None)

        faces = provider.detect(_fake_image_bytes())
        assert faces == []

    def test_detect_returns_pose_when_model_present(self) -> None:
        provider, mock_app = _create_provider_with_mock()

        mock_pose = MagicMock()

        def fake_pose_get(img: object, face: object) -> None:
            face["pose"] = np.array([5.0, 10.0, -3.0], dtype=np.float32)  # pitch, yaw, roll

        mock_pose.get.side_effect = fake_pose_get
        mock_app.models["landmark_3d_68"] = mock_pose

        faces = provider.detect(_fake_image_bytes(), include_pose=True)
        assert faces[0].pose is not None
        assert faces[0].pose.pitch == pytest.approx(5.0)
        assert faces[0].pose.yaw == pytest.approx(10.0)
        assert faces[0].pose.roll == pytest.approx(-3.0)

    def test_detect_skips_pose_model_by_default(self) -> None:
        provider, mock_app = _create_provider_with_mock()
        mock_pose = MagicMock()
        mock_app.models["landmark_3d_68"] = mock_pose

        faces = provider.detect(_fake_image_bytes())
        assert faces[0].pose is None
        mock_pose.get.assert_not_called()

    def test_detect_pose_none_without_model(self) -> None:
        provider, _mock_app = _create_provider_with_mock()
        faces = provider.detect(_fake_image_bytes(), include_pose=True)
        assert faces[0].pose is None

    def test_detect_batch_matches_single(self) -> None:
        provider, _mock_app = _create_provider_with_mock()
        img = _fake_image_bytes()

        batched = provider.detect_batch([img, b"not-an-image", img])

        assert len(batched) == 3
        assert batched[1] == []
        single = provider.detect(img)
        assert batched[0] == single
        assert batched[2] == single


class TestInsightFaceProviderEmbed:
    def test_embed_returns_embedding_no_demographics(self) -> None:
        provider, _mock_app = _create_provider_with_mock()

        faces = provider.embed(_fake_image_bytes())

        assert len(faces) == 1
        assert faces[0].embedding is not None
        assert len(faces[0].embedding) == 512
        assert faces[0].age is None
        assert faces[0].gender is None
        assert faces[0].race is None

    def test_embed_multiple_faces(self) -> None:
        provider, mock_app = _create_provider_with_mock()
        det_output = _make_det_output(
            [
                {"bbox": [10.0, 20.0, 110.0, 140.0], "det_score": 0.95},
                {"bbox": [200.0, 50.0, 300.0, 170.0], "det_score": 0.8},
            ]
        )
        mock_app.det_model.detect.return_value = det_output
        mock_app.models["recognition"].get_feat.return_value = np.random.randn(2, 512).astype(np.float32)

        faces = provider.embed(_fake_image_bytes())
        assert len(faces) == 2
        assert faces[0].det_score == pytest.approx(0.95)
        assert faces[1].det_score == pytest.approx(0.8)


class TestInsightFaceProviderAnalyze:
    def test_analyze_returns_all_fields(self) -> None:
        provider, mock_app = _create_provider_with_mock()

        # Add genderage model mock
        mock_ga = MagicMock()

        def fake_ga_get(img: object, face: object) -> tuple[int, int]:
            face["gender"] = 0  # female
            face["age"] = 25
            return 0, 25

        mock_ga.get.side_effect = fake_ga_get
        mock_app.models["genderage"] = mock_ga

        faces = provider.analyze(_fake_image_bytes())

        assert len(faces) == 1
        f = faces[0]
        assert f.embedding is not None
        assert len(f.embedding) == 512
        assert f.age == 25.0
        assert f.gender == "female"
        assert f.bbox.x == 10.0

    def test_analyze_male_gender(self) -> None:
        provider, mock_app = _create_provider_with_mock()

        mock_ga = MagicMock()

        def fake_ga_get(img: object, face: object) -> tuple[int, int]:
            face["gender"] = 1  # male
            face["age"] = 30
            return 1, 30

        mock_ga.get.side_effect = fake_ga_get
        mock_app.models["genderage"] = mock_ga

        faces = provider.analyze(_fake_image_bytes())
        assert faces[0].gender == "male"


class TestInsightFaceProviderLoadModel:
    @patch("insightface.app.FaceAnalysis", autospec=False)
    def test_load_model(self, mock_fa_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_fa_cls.return_value = mock_instance

        provider = InsightFaceProvider(
            use_gpu=False, det_size=(320, 320), model_name="buffalo_l", det_dynamic_batch=False
        )
        provider.load_model()

        mock_fa_cls.assert_called_once_with(
            name="buffalo_l", root="~/.insightface", providers=["CPUExecutionProvider"], provider_options=[{}]
        )
        mock_instance.prepare.assert_called_once_with(ctx_id=0, det_size=(320, 320))
        assert provider.is_loaded is True

    @patch("insightface.app.FaceAnalysis", autospec=False)
    def test_load_model_gpu(self, mock_fa_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_fa_cls.return_value = mock_instance

        provider = InsightFaceProvider(use_gpu=True, ctx_id=1, det_dynamic_batch=False)
        provider.load_model()

        mock_fa_cls.assert_called_once_with(
            name="buffalo_l",
            root="~/.insightface",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            provider_options=[
                {
                    "device_id": "1",
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": "1",
                    "cudnn_conv_use_max_workspace": "1",
                },
                {},
            ],
        )
        mock_instance.prepare.assert_called_once_with(ctx_id=1, det_size=(640, 640))

    @patch("insightface.app.FaceAnalysis", autospec=False)
    def test_load_model_converts_detector_to_dynamic_batch(self, mock_fa_cls: MagicMock) -> None:
        mock_fa_cls.return_value = MagicMock()

        with (
            patch("insightface.utils.ensure_available", return_value="/fake/pack") as mock_ensure,
            patch("glob.glob", return_value=["/fake/pack/det_10g.onnx"]) as mock_glob,
            patch(
                "src.services.face_provider.scrfd_export.convert_scrfd_to_dynamic_batch",
                return_value="converted",
            ) as mock_convert,
        ):
            provider = InsightFaceProvider(use_gpu=False, det_size=(640, 640), model_name="buffalo_l")
            provider.load_model()

        mock_ensure.assert_called_once_with("models", "buffalo_l", root=os.path.expanduser("~/.insightface"))
        mock_glob.assert_called_once_with("/fake/pack/det_*.onnx")
        mock_convert.assert_called_once_with("/fake/pack/det_10g.onnx", det_size=(640, 640))

    @patch("insightface.app.FaceAnalysis", autospec=False)
    def test_load_model_skips_conversion_when_disabled(self, mock_fa_cls: MagicMock) -> None:
        mock_fa_cls.return_value = MagicMock()

        with patch("insightface.utils.ensure_available") as mock_ensure:
            provider = InsightFaceProvider(use_gpu=False, det_dynamic_batch=False)
            provider.load_model()

        mock_ensure.assert_not_called()

    def test_provider_name(self) -> None:
        provider = InsightFaceProvider()
        assert provider.provider_name == "insightface"

    def test_is_loaded_default_false(self) -> None:
        provider = InsightFaceProvider()
        assert provider.is_loaded is False


def _empty_det_output() -> tuple[np.ndarray, None]:
    return np.zeros((0, 5), dtype=np.float32), None


class TestPadSquareFallback:
    """Pad-to-square fallback when the first detector pass returns zero faces.

    _fake_image_bytes() returns a 100x100 image. _pad_to_square() adds a
    100px gray border on every side, so the padded image is 300x300 with
    offset (dx=100, dy=100). Coordinates returned by the detector on the
    padded image are translated back to the original frame before being
    handed to the client.
    """

    def test_no_fallback_when_first_pass_hits(self) -> None:
        provider, mock_app = _create_provider_with_mock()
        provider.detect(_fake_image_bytes())
        assert mock_app.det_model.detect.call_count == 1

    def test_fallback_runs_when_first_pass_empty(self) -> None:
        provider, mock_app = _create_provider_with_mock()
        retry_hit = _make_det_output([{"bbox": [110.0, 120.0, 180.0, 190.0], "det_score": 0.95}])
        mock_app.det_model.detect.side_effect = [_empty_det_output(), retry_hit]

        faces = provider.detect(_fake_image_bytes())
        assert len(faces) == 1
        assert mock_app.det_model.detect.call_count == 2

    def test_both_passes_empty_returns_empty(self) -> None:
        provider, mock_app = _create_provider_with_mock()
        mock_app.det_model.detect.side_effect = [_empty_det_output(), _empty_det_output()]
        faces = provider.detect(_fake_image_bytes())
        assert faces == []
        assert mock_app.det_model.detect.call_count == 2

    def test_fallback_translates_bbox_to_original_coords(self) -> None:
        provider, mock_app = _create_provider_with_mock()
        # Padded bbox (110, 120, 180, 190) -> original (10, 20, 80, 90)
        retry_hit = _make_det_output([{"bbox": [110.0, 120.0, 180.0, 190.0], "det_score": 0.95}])
        mock_app.det_model.detect.side_effect = [_empty_det_output(), retry_hit]

        faces = provider.detect(_fake_image_bytes())
        assert len(faces) == 1
        f = faces[0]
        assert f.bbox.x == 10.0
        assert f.bbox.y == 20.0
        assert f.bbox.width == 70.0
        assert f.bbox.height == 70.0

    def test_fallback_clips_bbox_to_original_frame(self) -> None:
        # Padded bbox (90, 90, 250, 250) -> translated (-10, -10, 150, 150)
        # -> clipped to original 100x100 frame (0, 0, 100, 100).
        provider, mock_app = _create_provider_with_mock()
        retry_hit = _make_det_output([{"bbox": [90.0, 90.0, 250.0, 250.0], "det_score": 0.9}])
        mock_app.det_model.detect.side_effect = [_empty_det_output(), retry_hit]

        faces = provider.detect(_fake_image_bytes())
        assert faces[0].bbox.x == 0.0
        assert faces[0].bbox.y == 0.0
        assert faces[0].bbox.width == 100.0
        assert faces[0].bbox.height == 100.0

    def test_fallback_translates_landmarks(self) -> None:
        # _make_det_output places the third keypoint at the bbox center.
        # Padded bbox center (145, 155) -> original (45, 55).
        provider, mock_app = _create_provider_with_mock()
        retry_hit = _make_det_output([{"bbox": [110.0, 120.0, 180.0, 190.0], "det_score": 0.95}])
        mock_app.det_model.detect.side_effect = [_empty_det_output(), retry_hit]

        faces = provider.detect(_fake_image_bytes())
        landmarks = faces[0].landmarks
        assert landmarks is not None
        assert landmarks[2] == (45.0, 55.0)

    def test_embed_fallback_returns_translated_bbox_with_embedding(self) -> None:
        provider, mock_app = _create_provider_with_mock()
        retry_hit = _make_det_output([{"bbox": [110.0, 120.0, 180.0, 190.0], "det_score": 0.95}])
        mock_app.det_model.detect.side_effect = [_empty_det_output(), retry_hit]

        faces = provider.embed(_fake_image_bytes())
        assert len(faces) == 1
        assert faces[0].embedding is not None
        assert len(faces[0].embedding) == 512
        assert faces[0].bbox.x == 10.0
        assert faces[0].bbox.y == 20.0

    def test_analyze_fallback_returns_translated_bbox_with_demographics(self) -> None:
        provider, mock_app = _create_provider_with_mock()
        mock_ga = MagicMock()

        def fake_ga_get(img: object, face: object) -> tuple[int, int]:
            face["gender"] = 1  # type: ignore[index]
            face["age"] = 33  # type: ignore[index]
            return 1, 33

        mock_ga.get.side_effect = fake_ga_get
        mock_app.models["genderage"] = mock_ga

        retry_hit = _make_det_output([{"bbox": [110.0, 120.0, 180.0, 190.0], "det_score": 0.95}])
        mock_app.det_model.detect.side_effect = [_empty_det_output(), retry_hit]

        faces = provider.analyze(_fake_image_bytes())
        assert faces[0].bbox.x == 10.0
        assert faces[0].age == 33.0
        assert faces[0].gender == "male"

    def test_embed_batch_fallback_per_image(self) -> None:
        # Two images: first hits on the first pass, second misses and hits on
        # the padded retry. Verify each result lives in its own original frame.
        provider, mock_app = _create_provider_with_mock()
        first_hit = _make_det_output()  # bbox (10, 20, 110, 140) in original coords
        retry_hit = _make_det_output([{"bbox": [110.0, 120.0, 180.0, 190.0], "det_score": 0.9}])
        mock_app.det_model.detect.side_effect = [first_hit, _empty_det_output(), retry_hit]
        mock_app.models["recognition"].get_feat.return_value = np.random.randn(2, 512).astype(np.float32)

        results = provider.embed_batch([_fake_image_bytes(), _fake_image_bytes()])

        assert len(results) == 2
        assert len(results[0]) == 1
        assert results[0][0].bbox.x == 10.0  # first image: untranslated
        assert results[0][0].bbox.width == 100.0
        assert len(results[1]) == 1
        assert results[1][0].bbox.x == 10.0  # second image: translated from padded coords
        assert results[1][0].bbox.y == 20.0
        assert mock_app.det_model.detect.call_count == 3

    def test_embed_batch_no_fallback_when_all_hit(self) -> None:
        provider, mock_app = _create_provider_with_mock()
        mock_app.models["recognition"].get_feat.return_value = np.random.randn(2, 512).astype(np.float32)

        results = provider.embed_batch([_fake_image_bytes(), _fake_image_bytes()])
        assert len(results) == 2
        # No retries — one detect call per image.
        assert mock_app.det_model.detect.call_count == 2

    def test_pad_to_square_square_input(self) -> None:
        provider, _ = _create_provider_with_mock()
        img = np.full((100, 100, 3), 50, dtype=np.uint8)
        padded, dx, dy = provider._pad_to_square(img)
        assert padded.shape == (300, 300, 3)
        assert dx == 100
        assert dy == 100
        assert padded[0, 0].tolist() == [128, 128, 128]  # gray border
        assert padded[150, 150].tolist() == [50, 50, 50]  # original preserved at center

    def test_pad_to_square_non_square_input(self) -> None:
        provider, _ = _create_provider_with_mock()
        # h=80, w=120 -> side = max(80, 120) + 200 = 320
        img = np.zeros((80, 120, 3), dtype=np.uint8)
        padded, dx, dy = provider._pad_to_square(img)
        assert padded.shape == (320, 320, 3)
        assert dx == (320 - 120) // 2  # 100
        assert dy == (320 - 80) // 2  # 120

    def test_pad_to_square_honours_configured_border_and_fill(self) -> None:
        provider = InsightFaceProvider(use_gpu=False, pad_fallback_border_px=50, pad_fallback_fill=200)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        padded, dx, dy = provider._pad_to_square(img)
        assert padded.shape == (200, 200, 3)
        assert (dx, dy) == (50, 50)
        assert padded[0, 0].tolist() == [200, 200, 200]

    def test_alignment_uses_padded_image_not_original(self) -> None:
        # Regression guard: when the fallback fires, _align_and_embed must
        # receive the padded canvas (300x300) so warpAffine samples gray
        # padding past the original edges, not the (100x100) original.
        provider, mock_app = _create_provider_with_mock()
        retry_hit = _make_det_output([{"bbox": [110.0, 120.0, 180.0, 190.0], "det_score": 0.95}])
        mock_app.det_model.detect.side_effect = [_empty_det_output(), retry_hit]

        captured: list[np.ndarray] = []

        def fake_align(img: np.ndarray, kpss: np.ndarray) -> np.ndarray:
            captured.append(img)
            return np.zeros((kpss.shape[0], 512), dtype=np.float32)

        provider._align_and_embed = fake_align  # type: ignore[method-assign]

        provider.embed(_fake_image_bytes())

        assert len(captured) == 1
        assert captured[0].shape == (300, 300, 3)

    def test_detect_batch_fallback_per_image(self) -> None:
        # detect_batch on a non-batch-capable graph: sequential detects with
        # the same per-image pad retry as embed_batch/analyze_batch.
        provider, mock_app = _create_provider_with_mock()
        first_hit = _make_det_output()
        retry_hit = _make_det_output([{"bbox": [110.0, 120.0, 180.0, 190.0], "det_score": 0.9}])
        mock_app.det_model.detect.side_effect = [first_hit, _empty_det_output(), retry_hit]

        results = provider.detect_batch([_fake_image_bytes(), _fake_image_bytes()])

        assert len(results) == 2
        assert results[0][0].bbox.x == 10.0
        assert results[1][0].bbox.x == 10.0  # translated from padded coords
        assert results[1][0].bbox.y == 20.0
        assert results[0][0].embedding is None
        assert mock_app.det_model.detect.call_count == 3

    def test_analyze_batch_fallback_per_image(self) -> None:
        # Mirror of test_embed_batch_fallback_per_image for the analyze path.
        provider, mock_app = _create_provider_with_mock()
        first_hit = _make_det_output()
        retry_hit = _make_det_output([{"bbox": [110.0, 120.0, 180.0, 190.0], "det_score": 0.9}])
        mock_app.det_model.detect.side_effect = [first_hit, _empty_det_output(), retry_hit]
        mock_app.models["recognition"].get_feat.return_value = np.random.randn(2, 512).astype(np.float32)

        mock_ga = MagicMock()

        def fake_ga_get(img: object, face: object) -> tuple[int, int]:
            face["gender"] = 1  # type: ignore[index]
            face["age"] = 40
            return 1, 40

        mock_ga.get.side_effect = fake_ga_get
        mock_app.models["genderage"] = mock_ga

        results = provider.analyze_batch([_fake_image_bytes(), _fake_image_bytes()])

        assert len(results) == 2
        assert len(results[0]) == 1
        assert results[0][0].bbox.x == 10.0  # first image: untranslated
        assert len(results[1]) == 1
        assert results[1][0].bbox.x == 10.0  # second image: translated from padded coords
        assert results[1][0].bbox.y == 20.0
        assert results[1][0].age == 40.0
        assert results[1][0].gender == "male"
        assert mock_app.det_model.detect.call_count == 3


# --- Dynamic-batch SCRFD detection (issue #126) ---

_DET_STRIDES = [8, 16, 32]
_DET_ANCHORS_PER_CELL = 2
_DET_INPUT = 640


def _stride_anchor_count(stride: int) -> int:
    cells = (_DET_INPUT // stride) ** 2
    return cells * _DET_ANCHORS_PER_CELL


def _craft_scrfd_net_outs(
    faces_per_image: list[list[tuple[tuple[float, float, float, float], float]]],
) -> list[np.ndarray]:
    """Build raw batched SCRFD outputs (3 score maps, 3 bbox maps, 3 kps maps,
    each flat (N*K, C) — the converted graph keeps the stock 2-D layout, with
    image b occupying rows [b*K:(b+1)*K]) that decode to the given per-image
    (bbox, score) faces.

    Bboxes are in letterboxed 640x640 canvas space; every face is planted on
    the stride-8 anchor whose cell contains the bbox center, with all 5
    keypoints at the bbox center.
    """
    n = len(faces_per_image)
    outs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        s: (
            np.zeros((n, _stride_anchor_count(s), 1), dtype=np.float32),
            np.zeros((n, _stride_anchor_count(s), 4), dtype=np.float32),
            np.zeros((n, _stride_anchor_count(s), 10), dtype=np.float32),
        )
        for s in _DET_STRIDES
    }
    scores, bbox_preds, kps_preds = outs[8]
    width_cells = _DET_INPUT // 8
    for b, faces in enumerate(faces_per_image):
        for (x1, y1, x2, y2), score in faces:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            col, row = int(cx // 8), int(cy // 8)
            j = (row * width_cells + col) * _DET_ANCHORS_PER_CELL
            acx, acy = col * 8.0, row * 8.0
            scores[b, j, 0] = score
            bbox_preds[b, j] = [(acx - x1) / 8, (acy - y1) / 8, (x2 - acx) / 8, (y2 - acy) / 8]
            for p in range(5):
                kps_preds[b, j, 2 * p] = (cx - acx) / 8
                kps_preds[b, j, 2 * p + 1] = (cy - acy) / 8
    flat = [outs[s][0].reshape(-1, 1) for s in _DET_STRIDES]
    flat += [outs[s][1].reshape(-1, 4) for s in _DET_STRIDES]
    flat += [outs[s][2].reshape(-1, 10) for s in _DET_STRIDES]
    return flat


class _FakeNodeArg:
    name = "input.1"
    shape: list[object] = ["batch", 3, _DET_INPUT, _DET_INPUT]


class _FakeBatchedSession:
    """Stands in for an ORT session over the re-exported [N,3,640,640] graph."""

    def __init__(self, responses: list[list[np.ndarray]]) -> None:
        self._responses = responses
        self.run_batch_sizes: list[int] = []

    def get_inputs(self) -> list[_FakeNodeArg]:
        return [_FakeNodeArg()]

    def run(self, output_names: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        blob = next(iter(feed.values()))
        assert blob.shape[1:] == (3, _DET_INPUT, _DET_INPUT)
        self.run_batch_sizes.append(blob.shape[0])
        response = self._responses.pop(0)
        assert response[0].shape[0] == blob.shape[0] * _stride_anchor_count(8), (
            "crafted batch size must match the fed blob"
        )
        return response


def _make_batched_det_model(responses: list[list[np.ndarray]]) -> object:
    import types
    from types import SimpleNamespace

    from insightface.model_zoo.scrfd import SCRFD

    det_model = SimpleNamespace(
        session=_FakeBatchedSession(responses),
        input_size=(_DET_INPUT, _DET_INPUT),
        input_mean=127.5,
        input_std=128.0,
        input_name="input.1",
        output_names=[f"out_{i}" for i in range(9)],
        fmc=3,
        _feat_stride_fpn=_DET_STRIDES,
        _num_anchors=_DET_ANCHORS_PER_CELL,
        use_kps=True,
        det_thresh=0.5,
        nms_thresh=0.4,
    )
    det_model.nms = types.MethodType(SCRFD.nms, det_model)
    return det_model


def _create_batched_provider(
    responses: list[list[np.ndarray]], **provider_kwargs: object
) -> tuple[InsightFaceProvider, MagicMock]:
    provider = InsightFaceProvider(use_gpu=False, det_size=(_DET_INPUT, _DET_INPUT), **provider_kwargs)  # type: ignore[arg-type]
    mock_app = MagicMock()
    mock_app.det_model = _make_batched_det_model(responses)
    mock_rec = MagicMock()
    mock_rec.input_size = (112, 112)
    mock_rec.get_feat.return_value = np.random.randn(2, 512).astype(np.float32)
    mock_app.models = {"recognition": mock_rec}
    provider._app = mock_app
    provider._loaded = True
    return provider, mock_app


class TestBatchedDetection:
    """The 100x100 test image letterboxes onto the 640 canvas with
    det_scale = 6.4, so a face at canvas bbox (64, 128, 384, 448) decodes to
    (10, 20, 60, 70) in original image coordinates."""

    def test_detect_batch_single_session_run(self) -> None:
        responses = [
            _craft_scrfd_net_outs(
                [
                    [((64.0, 128.0, 384.0, 448.0), 0.9)],
                    [((128.0, 192.0, 256.0, 320.0), 0.8)],
                ]
            )
        ]
        provider, mock_app = _create_batched_provider(responses)

        results = provider.detect_batch([_fake_image_bytes(), _fake_image_bytes()])

        session = mock_app.det_model.session
        assert session.run_batch_sizes == [2]
        assert len(results) == 2
        f0, f1 = results[0][0], results[1][0]
        assert f0.bbox.x == pytest.approx(10.0, abs=0.05)
        assert f0.bbox.y == pytest.approx(20.0, abs=0.05)
        assert f0.bbox.width == pytest.approx(50.0, abs=0.1)
        assert f0.bbox.height == pytest.approx(50.0, abs=0.1)
        assert f0.det_score == pytest.approx(0.9, rel=1e-3)
        assert f0.landmarks is not None
        assert f0.landmarks[0][0] == pytest.approx(35.0, abs=0.05)
        assert f0.landmarks[0][1] == pytest.approx(45.0, abs=0.05)
        assert f1.bbox.x == pytest.approx(20.0, abs=0.05)
        assert f1.bbox.y == pytest.approx(30.0, abs=0.05)
        assert f1.det_score == pytest.approx(0.8, rel=1e-3)

    def test_detect_batch_chunked_by_det_trt_max_batch(self) -> None:
        face = [((64.0, 128.0, 384.0, 448.0), 0.9)]
        responses = [_craft_scrfd_net_outs([face]), _craft_scrfd_net_outs([face])]
        provider, mock_app = _create_batched_provider(responses, det_trt_max_batch=1)

        results = provider.detect_batch([_fake_image_bytes(), _fake_image_bytes()])

        assert mock_app.det_model.session.run_batch_sizes == [1, 1]
        assert len(results) == 2
        assert len(results[0]) == 1
        assert len(results[1]) == 1

    def test_pad_fallback_is_second_batched_pass_over_misses(self) -> None:
        # Image 1 hits on the first pass; image 2 misses and hits on the
        # padded (300x300 -> det_scale 640/300) retry. The retry pass must be
        # one batched run over just the misses, and its coordinates must come
        # back translated into the original 100x100 frame.
        pad_scale = 640.0 / 300.0
        padded_bbox = (110.0 * pad_scale, 120.0 * pad_scale, 180.0 * pad_scale, 190.0 * pad_scale)
        responses = [
            _craft_scrfd_net_outs([[((64.0, 128.0, 384.0, 448.0), 0.9)], []]),
            _craft_scrfd_net_outs([[(padded_bbox, 0.85)]]),
        ]
        provider, mock_app = _create_batched_provider(responses)

        results = provider.detect_batch([_fake_image_bytes(), _fake_image_bytes()])

        assert mock_app.det_model.session.run_batch_sizes == [2, 1]
        assert len(results[0]) == 1
        assert len(results[1]) == 1
        # Padded-space (110, 120, 180, 190) minus the (100, 100) pad offset.
        assert results[1][0].bbox.x == pytest.approx(10.0, abs=0.05)
        assert results[1][0].bbox.y == pytest.approx(20.0, abs=0.05)
        assert results[1][0].bbox.width == pytest.approx(70.0, abs=0.1)
        assert results[1][0].bbox.height == pytest.approx(70.0, abs=0.1)

    def test_embed_batch_uses_batched_detection(self) -> None:
        responses = [
            _craft_scrfd_net_outs(
                [
                    [((64.0, 128.0, 384.0, 448.0), 0.9)],
                    [((128.0, 192.0, 256.0, 320.0), 0.8)],
                ]
            )
        ]
        provider, mock_app = _create_batched_provider(responses)

        results = provider.embed_batch([_fake_image_bytes(), _fake_image_bytes()])

        assert mock_app.det_model.session.run_batch_sizes == [2]
        assert len(results) == 2
        assert results[0][0].embedding is not None
        assert len(results[0][0].embedding) == 512
        assert results[1][0].embedding is not None

    def test_undecodable_image_gets_empty_entry_without_detector_run(self) -> None:
        responses = [_craft_scrfd_net_outs([[((64.0, 128.0, 384.0, 448.0), 0.9)]])]
        provider, mock_app = _create_batched_provider(responses)

        results = provider.detect_batch([b"not-an-image", _fake_image_bytes()])

        # Only the decodable image reaches the detector.
        assert mock_app.det_model.session.run_batch_sizes == [1]
        assert results[0] == []
        assert len(results[1]) == 1

    def test_single_image_detect_routes_through_batched_graph(self) -> None:
        responses = [_craft_scrfd_net_outs([[((64.0, 128.0, 384.0, 448.0), 0.9)]])]
        provider, mock_app = _create_batched_provider(responses)

        faces = provider.detect(_fake_image_bytes())

        assert mock_app.det_model.session.run_batch_sizes == [1]
        assert len(faces) == 1
        assert faces[0].bbox.x == pytest.approx(10.0, abs=0.05)

    def test_batch_capability_probe_rejects_mocked_session(self) -> None:
        provider, _mock_app = _create_provider_with_mock()
        assert provider._det_batch_capable() is False
