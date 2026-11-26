"""Helpers for validating, fetching, cleaning and saving a homepage."""

from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from fastapi import HTTPException, status

from app.utils.cleaner import Cleaner
from app.utils.html_store import (
    IMAGES_DIR,
    ROOT_DIR,
    TIMEOUT,
    USER_AGENT,
    fetch_page,
    file_stem,
    save_clean_html,
    save_raw_html,
)
from app.utils.validators import (
    BlockedURLError,
    Validator,
    ValidationResult,
    request_public_url,
)

FAVICON_EXTENSIONS = {
    "image/x-icon": ".ico",
    "image/vnd.microsoft.icon": ".ico",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/svg+xml": ".svg",
    "image/webp": ".webp",
}


def validate_link(link: str) -> tuple[Validator, ValidationResult]:
    """Validate a submitted link and raise a clean HTTP error when unusable."""
    validator = Validator(link)
    result = validator.validate()

    if not result.is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": result.message, "link": link},
        )

    if not result.is_online:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": result.message,
                "link": validator.link,
                "status_code": result.status_code,
            },
        )

    return validator, result


def extract_homepage(
    link: str,
    validated: tuple[Validator, ValidationResult] | None = None,
) -> dict:
    """Fetch, clean and save one homepage. Validation may be reused."""
    if validated is None:
        validated = validate_link(link)
    validator, result = validated

    html, final_url = fetch_page(validator.link)
    raw_html_path = save_raw_html(final_url, html)

    cleaner = Cleaner(html)
    title = cleaner.get_title()
    favicon = get_favicon(cleaner.soup, final_url)
    content = cleaner.clean()
    clean_html_path = save_clean_html(final_url, cleaner.cleaned_html)

    return {
        "link": link,
        "final_url": final_url,
        "online": True,
        "status_code": result.status_code,
        "message": result.message,
        "title": title,
        "favicon": favicon,
        "raw_html": str(raw_html_path.relative_to(ROOT_DIR)),
        "clean_html": str(clean_html_path.relative_to(ROOT_DIR)),
        "content": content,
    }


def get_favicon(soup: BeautifulSoup, page_url: str) -> dict:
    """Find, download and store the site's favicon; always return its URL."""
    icon_url = urljoin(page_url, "/favicon.ico")

    for link in soup.find_all("link", href=True):
        rel = link.get("rel", "")
        rels = " ".join(rel).lower() if isinstance(rel, (list, tuple)) else str(rel).lower()
        if "icon" in rels:
            icon_url = urljoin(page_url, str(link["href"]).strip())
            break

    result = {"url": icon_url, "path": None}
    try:
        response = request_public_url(
            icon_url,
            timeout=TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
    except (BlockedURLError, requests.RequestException):
        return result

    content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
    if "image" not in content_type:
        return result

    suffix = Path(urlparse(icon_url).path).suffix.lower()
    if suffix not in {".ico", ".png", ".jpg", ".jpeg", ".svg", ".webp"}:
        suffix = FAVICON_EXTENSIONS.get(content_type, ".ico")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    path = IMAGES_DIR / f"{file_stem(icon_url)}{suffix}"
    path.write_bytes(response.content[:1_000_000])
    result["path"] = str(path.relative_to(ROOT_DIR))
    return result
