"""Find competitors via Serper and scrape their cleaned homepage content."""

from pathlib import Path
from urllib.parse import urlparse

import requests

from app.config import get_env
from app.services.llm_service import generate_search_queries
from app.utils.cleaner import Cleaner
from app.utils.html_store import (
    ROOT_DIR,
    fetch_page,
    save_clean_html,
    save_raw_html,
)
from app.utils.validators import Validator

SERPER_URL = "https://google.serper.dev/search"
TIMEOUT = 30.0
DEFAULT_GL = "in"
NON_COMPETITOR_HOSTS = (
    "youtube.com",
    "youtu.be",
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "twitter.com",
    "x.com",
    "reddit.com",
    "quora.com",
    "wikipedia.org",
)


def _site_host(host: str) -> str:
    """Reduce a hostname to its likely registrable domain for comparisons."""
    labels = str(host or "").strip().rstrip(".").lower().split(".")
    return ".".join(labels[-2:]) if len(labels) >= 2 else ""


def _is_non_competitor_result(url: str) -> bool:
    """Skip social/video/reference pages that are not competitor websites."""
    host = urlparse(url).hostname or ""
    site = _site_host(host)
    return site in NON_COMPETITOR_HOSTS


def find_competitors(
    clean_html_path: str | Path,
    count: int = 3,
    gl: str = DEFAULT_GL,
    max_competitors: int = 3,
    main_url: str = "",
    model: str | None = None,
) -> dict:
    """Generate queries, search them and scrape the resulting competitor sites."""
    queries = generate_search_queries(clean_html_path, count=count, model=model)
    search_results = search_queries(queries, gl=gl)
    urls = collect_competitor_urls(search_results, main_url=main_url)
    competitors = scrape_competitors(urls, max_competitors=max_competitors)
    if not competitors:
        raise RuntimeError(
            "Serper returned no accessible competitor websites for this analysis."
        )

    return {
        "queries": queries,
        "results": search_results,
        "competitors": competitors,
    }


def search_queries(queries: list[str], gl: str = DEFAULT_GL, limit: int = 5) -> list[dict]:
    """Run each search query one by one and return its organic results."""
    api_key = get_env("SERPER_API")
    if not api_key or api_key.startswith("<"):
        raise RuntimeError("SERPER_API is missing or still a placeholder in .env")

    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    results = []

    for query in queries:
        try:
            response = requests.post(
                SERPER_URL,
                headers=headers,
                json={"q": query, "gl": gl},
                timeout=TIMEOUT,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Serper search failed for {query!r}.") from exc

        if response.status_code >= 400:
            raise RuntimeError(
                f"Serper search failed for {query!r} (HTTP {response.status_code})."
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"Serper returned invalid JSON for {query!r}.") from exc
        organic = [
            {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            }
            for item in (data.get("organic") or [])[:limit]
        ]
        results.append(
            {
                "query": query,
                "organic": organic,
                "answer_box": data.get("answerBox") or data.get("answer_box"),
                "people_also_ask": (data.get("peopleAlsoAsk") or [])[:limit],
            }
        )

    return results


def collect_competitor_urls(search_results: list[dict], main_url: str = "") -> list[str]:
    """Pull unique, valid competitor URLs out of Serper's organic results."""
    main_site = _site_host(urlparse(main_url).hostname or "") if main_url else ""
    urls = []

    for result in search_results:
        for item in result.get("organic") or []:
            url = str(item.get("link") or "").strip()
            if not url or not Validator(url).has_valid_format():
                continue
            if _is_non_competitor_result(url):
                continue
            site = _site_host(urlparse(url).hostname or "")
            if main_site and site == main_site:
                continue
            urls.append(url)

    return list(dict.fromkeys(urls))


def scrape_competitors(urls: list[str], max_competitors: int = 3) -> list[dict]:
    """Fetch, clean and save the homepage of each unique competitor."""
    competitors = []
    seen_hosts = set()

    for url in urls:
        if len(competitors) >= max_competitors:
            break

        if not Validator(url).has_valid_format():
            continue

        host = urlparse(url).netloc.lower()
        if not host or host in seen_hosts:
            continue
        seen_hosts.add(host)

        try:
            html, final_url = fetch_page(url)
        except Exception:
            continue

        cleaner = Cleaner(html)
        content = cleaner.clean()
        raw_path = save_raw_html(final_url, html)
        clean_path = save_clean_html(final_url, cleaner.cleaned_html)

        competitors.append(
            {
                "url": final_url,
                "title": content["title"],
                "raw_html": str(raw_path.relative_to(ROOT_DIR)),
                "clean_html": str(clean_path.relative_to(ROOT_DIR)),
            }
        )

    return competitors
