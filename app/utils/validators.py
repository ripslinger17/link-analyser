"""Validation helpers for user supplied links, including SSRF protection."""

import ipaddress
import re
import socket
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests

ALLOWED_SCHEMES = {"http", "https"}
DEFAULT_TIMEOUT = 5.0
MAX_REDIRECTS = 5
REDIRECT_STATUSES = {301, 302, 303, 307, 308}
USER_AGENT = "LinkAnalyser/1.0"
_BLOCKED_HOSTNAMES = {"localhost"}
_HOST_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?")


class BlockedURLError(ValueError):
    """Raised when a URL or one of its redirects points at a blocked host."""


def _is_global_ip(value: str) -> bool:
    """Return True only for publicly routable IPv4/IPv6 addresses."""
    try:
        return ipaddress.ip_address(value).is_global
    except ValueError:
        return False


@dataclass(frozen=True)
class ValidationResult:
    """Result of checking a link's format, safety and availability."""

    is_valid: bool
    is_online: bool
    status_code: Optional[int] = None
    message: str = ""


class Validator:
    """Validate a link and check whether the target website is online."""

    def __init__(self, link: str, timeout: float = DEFAULT_TIMEOUT):
        self.link = (link or "").strip()
        self.timeout = timeout
        self.redirect_blocked = False

    def has_valid_format(self) -> bool:
        """Accept only http/https URLs pointing at public hosts."""
        if not self.link or any(char.isspace() for char in self.link):
            return False

        parsed = urlparse(self.link)
        if parsed.scheme not in ALLOWED_SCHEMES or not parsed.netloc:
            return False

        return self.is_public_host(parsed.hostname)

    def has_valid_host(self, host: Optional[str]) -> bool:
        """Backward-compatible alias for the public-host check."""
        return self.is_public_host(host)

    def is_public_host(self, host: Optional[str]) -> bool:
        """Block localhost, private ranges and other non-public addresses."""
        if not host:
            return False

        host = str(host).strip().rstrip(".")
        if not host or host.lower() in _BLOCKED_HOSTNAMES:
            return False
        if host.lower().endswith(".localhost"):
            return False

        try:
            return ipaddress.ip_address(host).is_global
        except ValueError:
            pass

        labels = host.split(".")
        if not (
            len(labels) >= 2
            and len(labels[-1]) >= 2
            and all(_HOST_LABEL.fullmatch(label) for label in labels)
        ):
            return False

        try:
            resolved = socket.getaddrinfo(host, None)
        except (socket.gaierror, UnicodeError, OSError):
            return False

        return bool(resolved) and all(_is_global_ip(info[4][0]) for info in resolved)

    def get_website_status(self) -> Optional[int]:
        """Return the site's HTTP status code, or None if it cannot be reached."""
        if not self.has_valid_format():
            return None

        try:
            response = request_public_url(
                self.link,
                timeout=self.timeout,
                stream=True,
                headers={"User-Agent": USER_AGENT},
            )
        except BlockedURLError:
            self.redirect_blocked = True
            return None
        except requests.RequestException:
            return None

        try:
            return response.status_code
        finally:
            response.close()

    def is_website_online(self) -> bool:
        """A site is online when it answers with anything other than a 5xx."""
        status = self.get_website_status()
        return status is not None and status < 500

    def validate(self) -> ValidationResult:
        """Run all checks and describe the outcome in a single result."""
        if not self.has_valid_format():
            return ValidationResult(
                is_valid=False,
                is_online=False,
                message=(
                    "Invalid link. Use a public http:// or https:// URL; "
                    "private and local addresses are not allowed."
                ),
            )

        status = self.get_website_status()
        if self.redirect_blocked:
            return ValidationResult(
                is_valid=False,
                is_online=False,
                message="The site redirected to a blocked or invalid address.",
            )
        if status is None:
            return ValidationResult(
                is_valid=True,
                is_online=False,
                message="The website is currently down or unreachable.",
            )
        if status >= 500:
            return ValidationResult(
                is_valid=True,
                is_online=False,
                status_code=status,
                message=f"The website is currently down (server returned {status}).",
            )

        return ValidationResult(
            is_valid=True,
            is_online=True,
            status_code=status,
            message="The website is online and reachable.",
        )


def request_public_url(
    url: str,
    timeout: float = DEFAULT_TIMEOUT,
    *,
    stream: bool = False,
    headers: Optional[dict] = None,
) -> requests.Response:
    """GET a URL while checking every redirect before it is followed.

    ``requests`` normally follows redirects internally, which would allow a
    public URL to reach a private/internal address. Following redirects here
    lets the SSRF check run on each hop first.
    """
    current = url
    visited = set()
    request_headers = headers or {"User-Agent": USER_AGENT}

    for _ in range(MAX_REDIRECTS + 1):
        if not Validator(current).has_valid_format():
            raise BlockedURLError(f"Blocked or invalid URL: {current}")
        if current in visited:
            raise BlockedURLError("The URL redirected in a loop.")
        visited.add(current)

        response = requests.get(
            current,
            timeout=timeout,
            headers=request_headers,
            allow_redirects=False,
            stream=stream,
        )

        location = response.headers.get("Location")
        if response.status_code in REDIRECT_STATUSES and location:
            response.close()
            current = urljoin(current, location)
            continue

        return response

    raise BlockedURLError("The URL redirected too many times.")
