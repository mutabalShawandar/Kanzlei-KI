from __future__ import annotations

from ingest.sources.html_guidance import extract_title_and_text


def test_extract_title_and_text_does_not_duplicate_nested_list_paragraphs() -> None:
    html = """
    <html><body>
      <h1>Gründung</h1>
      <main>
        <ul>
          <li><p>Erster Schritt: Anmeldung beim Finanzamt.</p></li>
        </ul>
      </main>
    </body></html>
    """

    _, text = extract_title_and_text(html, url="https://example.test/guide")

    assert text.count("Erster Schritt: Anmeldung beim Finanzamt.") == 1
