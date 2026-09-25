from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path

RAW_CACHE_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw_cache"


def cache_path(source_type: str, key: str) -> Path:
    return RAW_CACHE_ROOT / source_type / f"{key}.json"


def save_raw_text(source_type: str, key: str, url: str, retrieved_at: datetime, content: str) -> Path:
    path = cache_path(source_type, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "url": url,
        "retrieved_at": retrieved_at.isoformat(),
        "encoding": "text",
        "content": content,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_raw_bytes(source_type: str, key: str, url: str, retrieved_at: datetime, content: bytes) -> Path:
    path = cache_path(source_type, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "url": url,
        "retrieved_at": retrieved_at.isoformat(),
        "encoding": "base64",
        "content": base64.b64encode(content).decode("ascii"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
