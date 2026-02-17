import base64

from httpx import AsyncClient

# A valid 1x1 red PNG for testing
_TINY_PNG = base64.b64encode(
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
).decode()

_INVALID_B64 = "not-valid-base64!!!"


# --- /faces/detect ---


async def test_detect_single(client: AsyncClient) -> None:
    resp = await client.post("/faces/detect", json={"image_b64": _TINY_PNG})
    assert resp.status_code == 200
    data = resp.json()
    assert data["face_count"] == 1
    face = data["faces"][0]
    assert "bbox" in face
    assert "det_score" in face
    assert "embedding" not in face
    assert "age" not in face
    assert "gender" not in face


async def test_detect_invalid_base64(client: AsyncClient) -> None:
    resp = await client.post("/faces/detect", json={"image_b64": _INVALID_B64})
    assert resp.status_code == 400


async def test_detect_response_bbox_fields(client: AsyncClient) -> None:
    resp = await client.post("/faces/detect", json={"image_b64": _TINY_PNG})
    bbox = resp.json()["faces"][0]["bbox"]
    assert set(bbox.keys()) == {"x", "y", "width", "height"}


# --- /faces/embed ---


async def test_embed_single(client: AsyncClient) -> None:
    resp = await client.post("/faces/embed", json={"image_b64": _TINY_PNG})
    assert resp.status_code == 200
    data = resp.json()
    assert data["face_count"] == 1
    face = data["faces"][0]
    assert "bbox" in face
    assert "det_score" in face
    assert "embedding" in face
    assert isinstance(face["embedding"], list)
    assert len(face["embedding"]) == 512
    assert "age" not in face
    assert "gender" not in face


async def test_embed_invalid_base64(client: AsyncClient) -> None:
    resp = await client.post("/faces/embed", json={"image_b64": _INVALID_B64})
    assert resp.status_code == 400


# --- /faces/analyze ---


async def test_analyze_single(client: AsyncClient) -> None:
    resp = await client.post("/faces/analyze", json={"image_b64": _TINY_PNG})
    assert resp.status_code == 200
    data = resp.json()
    assert data["face_count"] == 1
    face = data["faces"][0]
    assert "bbox" in face
    assert "det_score" in face
    assert "embedding" in face
    assert "age" in face
    assert "gender" in face
    assert "race" in face
    assert "race_probs" in face


async def test_analyze_invalid_base64(client: AsyncClient) -> None:
    resp = await client.post("/faces/analyze", json={"image_b64": _INVALID_B64})
    assert resp.status_code == 400


# --- Batch endpoints ---


async def test_detect_batch(client: AsyncClient) -> None:
    resp = await client.post(
        "/faces/detect/batch",
        json={"images": [{"image_b64": _TINY_PNG}, {"image_b64": _TINY_PNG}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 2
    assert data["total_faces"] == 2
    for i, item in enumerate(data["results"]):
        assert item["index"] == i
        assert item["error"] is None
        assert item["face_count"] == 1


async def test_embed_batch(client: AsyncClient) -> None:
    resp = await client.post(
        "/faces/embed/batch",
        json={"images": [{"image_b64": _TINY_PNG}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_faces"] == 1
    assert data["results"][0]["faces"][0]["embedding"] is not None


async def test_analyze_batch(client: AsyncClient) -> None:
    resp = await client.post(
        "/faces/analyze/batch",
        json={"images": [{"image_b64": _TINY_PNG}]},
    )
    assert resp.status_code == 200
    face = resp.json()["results"][0]["faces"][0]
    assert "age" in face
    assert "gender" in face


async def test_batch_with_invalid_image(client: AsyncClient) -> None:
    resp = await client.post(
        "/faces/detect/batch",
        json={"images": [{"image_b64": _TINY_PNG}, {"image_b64": _INVALID_B64}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"][0]["error"] is None
    assert data["results"][0]["face_count"] == 1
    assert data["results"][1]["error"] is not None
    assert data["results"][1]["face_count"] == 0


async def test_batch_exceeds_max_size(client: AsyncClient) -> None:
    images = [{"image_b64": _TINY_PNG}] * 21  # default max is 20
    resp = await client.post("/faces/detect/batch", json={"images": images})
    assert resp.status_code == 400
