from __future__ import annotations

import httpx
import respx

from ingest.sources import cache as cache_module
from ingest.sources.finanzamt_elster import fetch_finanzamt_elster_page

SAMPLE_HTML = """
<html>
<head><title>Fallback Title</title></head>
<body>
<article>
<h1>Fragebogen zur steuerlichen Erfassung</h1>
<p>Nach Aufnahme einer selbststaendigen Taetigkeit ist der Fragebogen elektronisch zu uebermitteln.</p>
<p>Die Uebermittlung erfolgt ueber Mein ELSTER.</p>
</article>
</body>
</html>
"""


@respx.mock
def test_fetch_finanzamt_elster_page_extracts_title_and_article_text(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache_module, "RAW_CACHE_ROOT", tmp_path)
    url = "https://www.elster.de/eportal/infoseite/fragebogen-steuerliche-erfassung"
    respx.get(url).mock(return_value=httpx.Response(200, text=SAMPLE_HTML))

    documents = fetch_finanzamt_elster_page(url)

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Fragebogen zur steuerlichen Erfassung"
    assert document.source_type == "guidance"
    assert "Mein ELSTER" in document.text

    cached_files = list((tmp_path / "guidance").glob("*.json"))
    assert len(cached_files) == 1
