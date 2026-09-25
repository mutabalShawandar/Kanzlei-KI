from __future__ import annotations

import httpx
from bs4 import BeautifulSoup

from ingest.sources.base import SourceDocument, stable_id, utcnow
from ingest.sources.cache import save_raw_text


class GuidancePageError(Exception):
    pass


def extract_title_and_text(html: str, url: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    title_element = soup.find("h1") or soup.find("title")
    if title_element is None or not title_element.get_text(strip=True):
        raise GuidancePageError(f"No title found for page '{url}'")
    title = title_element.get_text(strip=True)

    main = soup.find("main") or soup.find("article") or soup.find(attrs={"role": "main"})
    if main is None:
        raise GuidancePageError(f"No main content region found for page '{url}'")

    paragraphs = [
        paragraph_text
        for element in main.find_all(["p", "li", "h2", "h3"])
        if (paragraph_text := element.get_text(" ", strip=True))
    ]
    text = "\n".join(paragraphs)
    if not text:
        raise GuidancePageError(f"No article text extracted for page '{url}'")

    return title, text


def fetch_guidance_document(
    url: str, source_type: str, client: httpx.Client | None = None
) -> SourceDocument:
    owns_client = client is None
    http_client = client if client is not None else httpx.Client(timeout=30.0, follow_redirects=True)
    try:
        response = http_client.get(url)
        response.raise_for_status()
    finally:
        if owns_client:
            http_client.close()

    retrieved_at = utcnow()
    source_id = stable_id(source_type, url)
    save_raw_text(source_type, source_id, url, retrieved_at, response.text)
    title, text = extract_title_and_text(response.text, url)

    return SourceDocument(
        source_id=source_id,
        source_type=source_type,
        title=title,
        url=url,
        section=None,
        text=text,
        retrieved_at=retrieved_at,
    )
