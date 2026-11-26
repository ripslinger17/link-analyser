"""Shared helpers for fetching and persisting page HTML."""

import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse

import requests
from fastapi import HTTPException, status

from app.utils.validators import BlockedURLError, request_public_url

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_HTML_DIR = ROOT_DIR / "app" / "data" / "raw_html"
CLEAN_HTML_DIR = ROOT_DIR / "app" / "data" / "clean_html"
IMAGES_DIR = ROOT_DIR / "app" / "data" / "images"
USER_AGENT = "LinkAnalyser/1.0"
TIMEOUT = 10.0


def fetch_page(url: str, timeout: float = TIMEOUT) -> tuple[str, str]:
    """Download a page and return its HTML plus the final redirected URL."""
    try:
        response = request_public_url(
            url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
    except BlockedURLError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "The site redirected to a blocked or invalid address.",
                "link": url,
            },
        ) from exc
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"error": "The website could not be downloaded.", "link": url},
        ) from exc

    if "html" not in response.headers.get("Content-Type", "").lower():
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"error": "The link does not point to an HTML page.", "link": url},
        )

    return response.text, response.url


def save_raw_html(url: str, html: str) -> Path:
    """Persist untouched HTML under app/data/raw_html."""
    return _save_html(url, html, RAW_HTML_DIR)


def save_clean_html(url: str, html: str) -> Path:
    """Persist whitelisted HTML under app/data/clean_html."""
    return _save_html(url, html, CLEAN_HTML_DIR)


def file_stem(url: str) -> str:
    """Build a unique, filesystem-safe name for a URL."""
    host = urlparse(url).netloc or "site"
    safe_host = re.sub(r"[^A-Za-z0-9]+", "_", host).strip("_") or "site"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"{safe_host}_{digest}"


def _save_html(url: str, html: str, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{file_stem(url)}.html"
    path.write_text(html, encoding="utf-8")
    return path
