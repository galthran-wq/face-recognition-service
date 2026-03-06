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

        provider = InsightFaceProvider(use_gpu=False, det_size=(320, 320), model_name="buffalo_l")
        provider.load_model()

        mock_fa_cls.assert_called_once_with(name="buffalo_l", root="~/.insightface", providers=["CPUExecutionProvider"])
        mock_instance.prepare.assert_called_once_with(ctx_id=0, det_size=(320, 320))
        assert provider.is_loaded is True

    @patch("insightface.app.FaceAnalysis", autospec=False)
    def test_load_model_gpu(self, mock_fa_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_fa_cls.return_value = mock_instance

        provider = InsightFaceProvider(use_gpu=True, ctx_id=1)
        provider.load_model()

        mock_fa_cls.assert_called_once_with(
            name="buffalo_l", root="~/.insightface", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        mock_instance.prepare.assert_called_once_with(ctx_id=1, det_size=(640, 640))

    def test_provider_name(self) -> None:
        provider = InsightFaceProvider()
        assert provider.provider_name == "insightface"

    def test_is_loaded_default_false(self) -> None:
        provider = InsightFaceProvider()
        assert provider.is_loaded is False
