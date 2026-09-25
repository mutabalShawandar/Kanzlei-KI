from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from pydantic import BaseModel


class SourceDocument(BaseModel):
    source_id: str
    source_type: str
    title: str
    url: str
    section: str | None
    text: str
    retrieved_at: datetime


def stable_id(*parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
