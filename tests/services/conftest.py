import base64
import os
from pathlib import Path

import pytest


@pytest.fixture
def test_image_bytes() -> bytes:
    image_path = os.environ.get("TEST_IMAGE_PATH")
    if image_path:
        return Path(image_path).read_bytes()

    image_b64 = os.environ.get("TEST_IMAGE_B64")
    if image_b64:
        return base64.b64decode(image_b64)

    pytest.skip("TEST_IMAGE_PATH or TEST_IMAGE_B64 not set")
