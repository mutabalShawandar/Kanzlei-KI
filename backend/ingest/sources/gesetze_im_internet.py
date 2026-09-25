from __future__ import annotations

import io
import zipfile
from datetime import datetime
from xml.etree import ElementTree as ET

import httpx

from ingest.sources.base import SourceDocument, stable_id, utcnow
from ingest.sources.cache import save_raw_bytes

SOURCE_TYPE = "statute"
ARCHIVE_URL_TEMPLATE = "https://www.gesetze-im-internet.de/{slug}/xml.zip"


class GesetzeImInternetError(Exception):
    pass


def _extract_xml_bytes(archive_bytes: bytes, slug: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        xml_names = [name for name in archive.namelist() if name.endswith(".xml")]
        if not xml_names:
            raise GesetzeImInternetError(f"No XML file found in archive for law '{slug}'")
        return archive.read(xml_names[0])


def _element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def _parse_norm(norm: ET.Element, slug: str, url: str, retrieved_at: datetime) -> SourceDocument | None:
    metadaten = norm.find("metadaten")
    if metadaten is None:
        return None

    enbez = _element_text(metadaten.find("enbez"))
    titel = _element_text(metadaten.find("titel"))

    textdaten = norm.find("textdaten")
    content = textdaten.find(".//Content") if textdaten is not None else None
    text = _element_text(content)
    if not text:
        return None

    title = titel or enbez or slug.upper()
    section = enbez or None
    source_id = f"{slug}-{stable_id(slug, enbez or titel or text[:50])}"
    return SourceDocument(
        source_id=source_id,
        source_type=SOURCE_TYPE,
        title=title,
        url=url,
        section=section,
        text=text,
        retrieved_at=retrieved_at,
    )


def parse_gesetz_xml(xml_bytes: bytes, slug: str, url: str, retrieved_at: datetime) -> list[SourceDocument]:
    root = ET.fromstring(xml_bytes)
    documents = [
        document
        for norm in root.findall("norm")
        if (document := _parse_norm(norm, slug, url, retrieved_at)) is not None
    ]
    if not documents:
        raise GesetzeImInternetError(f"No sections parsed from XML for law '{slug}'")
    return documents


def fetch_gesetz(slug: str, client: httpx.Client | None = None) -> list[SourceDocument]:
    url = ARCHIVE_URL_TEMPLATE.format(slug=slug)
    owns_client = client is None
    http_client = client if client is not None else httpx.Client(timeout=30.0)
    try:
        response = http_client.get(url)
        response.raise_for_status()
    finally:
        if owns_client:
            http_client.close()

    retrieved_at = utcnow()
    save_raw_bytes(SOURCE_TYPE, slug, url, retrieved_at, response.content)
    xml_bytes = _extract_xml_bytes(response.content, slug)
    return parse_gesetz_xml(xml_bytes, slug, url, retrieved_at)
