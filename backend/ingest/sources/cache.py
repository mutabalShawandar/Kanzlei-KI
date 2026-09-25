from __future__ import annotations

import base64
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

RAW_CACHE_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw_cache"


def cache_path(source_type: str, key: str) -> Path:
    return RAW_CACHE_ROOT / source_type / f"{key}.json"


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def save_raw_text(source_type: str, key: str, url: str, retrieved_at: datetime, content: str) -> Path:
    path = cache_path(source_type, key)
    payload = {
        "url": url,
        "retrieved_at": retrieved_at.isoformat(),
        "encoding": "text",
        "content": content,
    }
    _write_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2))
    return path


def save_raw_bytes(source_type: str, key: str, url: str, retrieved_at: datetime, content: bytes) -> Path:
    path = cache_path(source_type, key)
    payload = {
        "url": url,
        "retrieved_at": retrieved_at.isoformat(),
        "encoding": "base64",
        "content": base64.b64encode(content).decode("ascii"),
    }
    _write_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2))
    return path
