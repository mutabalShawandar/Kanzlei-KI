from __future__ import annotations

import json

from ingest.sources import cache as cache_module
from ingest.sources.base import utcnow


def test_save_raw_text_writes_no_temp_files_left_behind(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache_module, "RAW_CACHE_ROOT", tmp_path)

    path = cache_module.save_raw_text("guidance", "key1", "https://example.test", utcnow(), "hello")

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["content"] == "hello"
    assert list(path.parent.glob("*.tmp")) == []


def test_save_raw_text_preserves_previous_entry_if_write_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache_module, "RAW_CACHE_ROOT", tmp_path)
    path = cache_module.save_raw_text("guidance", "key1", "https://example.test", utcnow(), "original")

    def _boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(cache_module.os, "replace", _boom)
    try:
        cache_module.save_raw_text("guidance", "key1", "https://example.test", utcnow(), "corrupted")
    except OSError:
        pass

    assert json.loads(path.read_text(encoding="utf-8"))["content"] == "original"
    assert list(path.parent.glob("*.tmp")) == []
