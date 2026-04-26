import os
from pathlib import Path

import pytest
from pydantic import ValidationError


def test_config_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)

    # Clear cached singleton
    import gme.config as cfg_module
    cfg_module._settings = None

    from gme.config import Settings
    s = Settings()

    assert s.gemini_api_key == "test-key-123"
    assert s.qdrant_url == "http://localhost:6333"
    assert s.embedding_dim == 1536
    assert s.records_per_shard == 40
    assert s.max_concurrent_jobs == 9
    assert s.max_enqueued_tokens == 432_000

    cfg_module._settings = None  # cleanup


def test_config_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    import gme.config as cfg_module
    cfg_module._settings = None

    from gme.config import Settings
    with pytest.raises(ValidationError):
        Settings()

    cfg_module._settings = None
