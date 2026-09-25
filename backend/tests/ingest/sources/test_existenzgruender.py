from __future__ import annotations

import httpx
import pytest
import respx

from ingest.sources import cache as cache_module
from ingest.sources.existenzgruender import fetch_existenzgruender_page
from ingest.sources.html_guidance import GuidancePageError

SAMPLE_HTML = """
<html>
<head><title>Fallback Title</title></head>
<body>
<h1>Freiberuflichkeit anmelden</h1>
<main>
<p>Freiberufler melden ihre Taetigkeit beim Finanzamt an, nicht beim Gewerbeamt.</p>
<h2>Voraussetzungen</h2>
<p>Eine kaufmaennische Eintragung ist fuer Freiberufler nicht erforderlich.</p>
<li>Steuerliche Erfassung ueber das Formular "Fragebogen zur steuerlichen Erfassung"</li>
</main>
<footer><p>Nicht relevanter Footer-Text</p></footer>
</body>
</html>
"""


@respx.mock
def test_fetch_existenzgruender_page_extracts_title_and_main_text(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache_module, "RAW_CACHE_ROOT", tmp_path)
    url = "https://www.existenzgruender.de/DE/Gruendung-vorbereiten/Beispiel.html"
    respx.get(url).mock(return_value=httpx.Response(200, text=SAMPLE_HTML))

    documents = fetch_existenzgruender_page(url)

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Freiberuflichkeit anmelden"
    assert document.source_type == "guidance"
    assert document.url == url
    assert document.section is None
    assert "Fragebogen zur steuerlichen Erfassung" in document.text
    assert "Nicht relevanter Footer-Text" not in document.text

    cached_files = list((tmp_path / "guidance").glob("*.json"))
    assert len(cached_files) == 1


@respx.mock
def test_fetch_existenzgruender_page_raises_when_no_main_content(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache_module, "RAW_CACHE_ROOT", tmp_path)
    url = "https://www.existenzgruender.de/DE/Gruendung-vorbereiten/Leer.html"
    respx.get(url).mock(return_value=httpx.Response(200, text="<html><body><h1>Titel</h1></body></html>"))

    with pytest.raises(GuidancePageError):
        fetch_existenzgruender_page(url)
