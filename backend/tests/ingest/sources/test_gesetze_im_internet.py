from __future__ import annotations

import io
import zipfile

import httpx
import pytest
import respx

from ingest.sources import cache as cache_module
from ingest.sources.gesetze_im_internet import (
    GesetzeImInternetError,
    fetch_gesetz,
    parse_gesetz_xml,
)
from ingest.sources.base import utcnow

ADJACENT_PARAGRAPHS_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<dokumente builddate="20250101000000">
<norm doknr="1">
<metadaten>
<jurabk>EStG</jurabk>
<enbez>&#167; 1</enbez>
<titel format="text">Steuerpflicht</titel>
</metadaten>
<textdaten>
<text format="XML"><Content><P>Erster Absatz endet hier.</P><P>Zweiter Absatz beginnt hier.</P></Content></text>
</textdaten>
</norm>
</dokumente>
"""

SAMPLE_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<dokumente builddate="20250101000000">
<norm doknr="1">
<metadaten>
<jurabk>EStG</jurabk>
<enbez>&#167; 1</enbez>
<titel format="text">Steuerpflicht</titel>
</metadaten>
<textdaten>
<text format="XML"><Content><P>(1) Natuerliche Personen mit Wohnsitz im Inland sind unbeschraenkt einkommensteuerpflichtig.</P></Content></text>
</textdaten>
</norm>
<norm doknr="2">
<metadaten>
<jurabk>EStG</jurabk>
<enbez>&#167; 2</enbez>
<titel format="text">Umfang der Besteuerung</titel>
</metadaten>
<textdaten>
<text format="XML"><Content><P>Der Einkommensteuer unterliegen die im Gesetz genannten Einkuenfte.</P></Content></text>
</textdaten>
</norm>
<norm doknr="3">
<metadaten>
<jurabk>EStG</jurabk>
<enbez>Inhaltsuebersicht</enbez>
</metadaten>
</norm>
</dokumente>
"""


def _zip_bytes(xml_bytes: bytes, filename: str = "BJNR010050934.xml") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(filename, xml_bytes)
    return buffer.getvalue()


def test_parse_gesetz_xml_extracts_sections_and_skips_empty_norms() -> None:
    documents = parse_gesetz_xml(SAMPLE_XML, slug="estg", url="https://example.test/estg", retrieved_at=utcnow())

    assert len(documents) == 2
    assert documents[0].section == "§ 1"
    assert documents[0].title == "Steuerpflicht"
    assert documents[0].source_type == "statute"
    assert "einkommensteuerpflichtig" in documents[0].text
    assert documents[1].section == "§ 2"


def test_parse_gesetz_xml_keeps_adjacent_paragraphs_separate() -> None:
    documents = parse_gesetz_xml(
        ADJACENT_PARAGRAPHS_XML, slug="estg", url="https://example.test/estg", retrieved_at=utcnow()
    )

    assert len(documents) == 1
    assert "endet hier.\nZweiter" in documents[0].text
    assert "endet hier.Zweiter" not in documents[0].text


def test_parse_gesetz_xml_raises_when_no_sections_found() -> None:
    empty_xml = b"<dokumente></dokumente>"
    with pytest.raises(GesetzeImInternetError):
        parse_gesetz_xml(empty_xml, slug="estg", url="https://example.test/estg", retrieved_at=utcnow())


@respx.mock
def test_fetch_gesetz_downloads_and_parses_and_caches_raw_zip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache_module, "RAW_CACHE_ROOT", tmp_path)

    zip_content = _zip_bytes(SAMPLE_XML)
    route = respx.get("https://www.gesetze-im-internet.de/estg/xml.zip").mock(
        return_value=httpx.Response(200, content=zip_content)
    )

    documents = fetch_gesetz("estg")

    assert route.called
    assert len(documents) == 2
    assert all(doc.source_id.startswith("estg-") for doc in documents)

    cached_files = list((tmp_path / "statute").glob("*.json"))
    assert len(cached_files) == 1


@respx.mock
def test_fetch_gesetz_follows_redirects_on_caller_supplied_client(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache_module, "RAW_CACHE_ROOT", tmp_path)

    zip_content = _zip_bytes(SAMPLE_XML)
    respx.get("https://www.gesetze-im-internet.de/estg/xml.zip").mock(
        return_value=httpx.Response(
            302, headers={"Location": "https://www.gesetze-im-internet.de/estg/xml-final.zip"}
        )
    )
    respx.get("https://www.gesetze-im-internet.de/estg/xml-final.zip").mock(
        return_value=httpx.Response(200, content=zip_content)
    )

    with httpx.Client(follow_redirects=False) as caller_client:
        documents = fetch_gesetz("estg", client=caller_client)

    assert len(documents) == 2
