"""HTTP adapter for HTML documents."""

import httpx

DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml",
    "User-Agent": "AnimateAgent/0.1 (+document parser)",
}


async def fetch_html(url: str, *, client: httpx.AsyncClient | None = None) -> str:
    """Download an HTML page and fail explicitly for HTTP or content-type errors."""

    owns_client = client is None
    http_client = client or httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(20.0),
        headers=DEFAULT_HEADERS,
    )
    try:
        response = await http_client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if content_type and "html" not in content_type and "xhtml" not in content_type:
            raise ValueError(f"URL did not return HTML (content-type: {content_type})")
        return response.text
    finally:
        if owns_client:
            await http_client.aclose()
