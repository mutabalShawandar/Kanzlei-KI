from __future__ import annotations

import httpx

from ingest.sources.base import SourceDocument
from ingest.sources.html_guidance import fetch_guidance_document

SOURCE_TYPE = "guidance"


def fetch_finanzamt_elster_page(url: str, client: httpx.Client | None = None) -> list[SourceDocument]:
    return [fetch_guidance_document(url, SOURCE_TYPE, client=client)]
