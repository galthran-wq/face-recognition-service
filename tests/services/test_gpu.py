import pytest
from src.services.face_provider.insightface import InsightFaceProvider

pytestmark = pytest.mark.gpu


def test_gpu_provider_loads() -> None:
    provider = InsightFaceProvider(use_gpu=True, ctx_id=0)
    provider.load_model()
    assert provider.is_loaded is True


def test_gpu_inference(test_image_bytes: bytes) -> None:
    provider = InsightFaceProvider(use_gpu=True, ctx_id=0)
    provider.load_model()
    faces = provider.analyze(test_image_bytes)
    assert len(faces) > 0
    for face in faces:
        assert face.embedding is not None
        assert len(face.embedding) == 512
        assert any(v != 0.0 for v in face.embedding)


def test_gpu_cpu_parity(test_image_bytes: bytes) -> None:
    gpu_provider = InsightFaceProvider(use_gpu=True, ctx_id=0)
    gpu_provider.load_model()
    cpu_provider = InsightFaceProvider(use_gpu=False)
    cpu_provider.load_model()

    gpu_faces = gpu_provider.analyze(test_image_bytes)
    cpu_faces = cpu_provider.analyze(test_image_bytes)

    assert len(gpu_faces) == len(cpu_faces)
    for gf, cf in zip(gpu_faces, cpu_faces, strict=True):
        assert gf.embedding is not None
        assert cf.embedding is not None
        import numpy as np

        g = np.array(gf.embedding)
        c = np.array(cf.embedding)
        cos_sim = float(np.dot(g, c) / (np.linalg.norm(g) * np.linalg.norm(c)))
        assert cos_sim > 0.99, f"GPU/CPU cosine similarity too low: {cos_sim}"
