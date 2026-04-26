import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from gme.batch_builder import _build_request


def test_build_request_structure(tmp_path: Path) -> None:
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # fake JPEG bytes

    result = _build_request(
        article_id="0108775015",
        text="Strap top Vest top Black",
        image_path=image_path,
        embedding_dim=1536,
    )

    assert result["key"] == "0108775015"
    request = result["request"]
    assert "contents" in request
    parts = request["contents"][0]["parts"]
    assert parts[0]["text"] == "Strap top Vest top Black"
    assert parts[1]["inline_data"]["mime_type"] == "image/jpeg"
    assert len(parts[1]["inline_data"]["data"]) > 0  # has base64 content
    assert request["config"]["output_dimensionality"] == 1536
