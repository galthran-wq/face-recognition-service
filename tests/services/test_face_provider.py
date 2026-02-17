from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from src.services.face_provider.insightface import InsightFaceProvider


def _make_fake_face(
    *,
    bbox: list[float] | None = None,
    det_score: float = 0.95,
    normed_embedding: np.ndarray[..., np.dtype[np.float32]] | None = None,
    age: float | None = 30.0,
    sex: int | None = 1,
    race: str | None = "white",
) -> SimpleNamespace:
    return SimpleNamespace(
        bbox=np.array(bbox or [10.0, 20.0, 110.0, 140.0]),
        det_score=det_score,
        normed_embedding=normed_embedding if normed_embedding is not None else np.random.randn(512).astype(np.float32),
        age=age,
        sex=sex,
        race=race,
    )


def _create_provider_with_mock() -> tuple[InsightFaceProvider, MagicMock]:
    provider = InsightFaceProvider(use_gpu=False, det_size=(640, 640))
    mock_app = MagicMock()
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
        fake_face = _make_fake_face()
        mock_app.get.return_value = [fake_face]

        faces = provider.detect(_fake_image_bytes())

        assert len(faces) == 1
        assert faces[0].bbox.x == 10.0
        assert faces[0].bbox.y == 20.0
        assert faces[0].bbox.width == 100.0
        assert faces[0].bbox.height == 120.0
        assert faces[0].det_score == 0.95
        assert faces[0].embedding is None
        assert faces[0].age is None
        assert faces[0].gender is None
        assert faces[0].race is None

    def test_detect_empty_image(self) -> None:
        provider, mock_app = _create_provider_with_mock()
        faces = provider.detect(b"not-an-image")
        assert faces == []


class TestInsightFaceProviderEmbed:
    def test_embed_returns_embedding_no_demographics(self) -> None:
        provider, mock_app = _create_provider_with_mock()
        fake_face = _make_fake_face()
        mock_app.get.return_value = [fake_face]

        faces = provider.embed(_fake_image_bytes())

        assert len(faces) == 1
        assert faces[0].embedding is not None
        assert len(faces[0].embedding) == 512
        assert faces[0].age is None
        assert faces[0].gender is None
        assert faces[0].race is None

    def test_embed_multiple_faces(self) -> None:
        provider, mock_app = _create_provider_with_mock()
        mock_app.get.return_value = [_make_fake_face(), _make_fake_face(det_score=0.8)]

        faces = provider.embed(_fake_image_bytes())
        assert len(faces) == 2


class TestInsightFaceProviderAnalyze:
    def test_analyze_returns_all_fields(self) -> None:
        provider, mock_app = _create_provider_with_mock()
        fake_face = _make_fake_face(age=25.0, sex=0, race="asian")
        mock_app.get.return_value = [fake_face]

        faces = provider.analyze(_fake_image_bytes())

        assert len(faces) == 1
        f = faces[0]
        assert f.embedding is not None
        assert len(f.embedding) == 512
        assert f.age == 25.0
        assert f.gender == "female"
        assert f.race == "asian"
        assert f.bbox.x == 10.0

    def test_analyze_male_gender(self) -> None:
        provider, mock_app = _create_provider_with_mock()
        mock_app.get.return_value = [_make_fake_face(sex=1)]

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
