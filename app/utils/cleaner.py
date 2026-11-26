"""Keep only SEO-relevant HTML tags and expose useful page content."""

import re
from typing import Optional

from bs4 import BeautifulSoup, Tag


def _attr(tag: Tag, key: str, default=""):
    """Read an attribute even from a tag BeautifulSoup has partially detached."""
    attrs = getattr(tag, "attrs", None)
    if not isinstance(attrs, dict):
        return default
    return attrs.get(key, default)


def _is_active_tag(tag: Tag) -> bool:
    """True when a tag is still usable after earlier decompose/unwrap passes."""
    return (
        bool(getattr(tag, "name", None))
        and not getattr(tag, "decomposed", False)
        and isinstance(getattr(tag, "attrs", None), dict)
    )


# Tags that are kept in the cleaned document.
PRESERVED_TAGS = {
    "title",
    "meta",
    "link",
    "h1",
    "h2",
    "h3",
    "p",
    "li",
    "a",
    "main",
    "article",
    "section",
    "address",
}

# Head tags are kept only when they match these exact attributes.
PRESERVED_META = (
    {"name": "description"},
    {"name": "keywords"},
    {"property": "og:title"},
    {"property": "og:description"},
)

# Elements whose text is never useful, even if their tag would be unwrapped.
REMOVE_TAGS = {
    "script",
    "style",
    "noscript",
    "template",
    "iframe",
    "svg",
    "img",
    "picture",
    "source",
    "video",
    "audio",
    "canvas",
    "form",
    "button",
    "aside",
    "ins",
}

# Common id/class names used by ads, trackers and page chrome.
NOISE_PATTERN = re.compile(
    r"(?:^|[-_\s])(?:"
    r"advertisements?|advertising|advert|ads?|sponsored?|sponsor|"
    r"promo(?:tion)?|banners?|sidebars?|popups?|modals?|"
    r"cookies?|consent|newsletter|social|shares?|comments?|"
    r"related|recommended|trending|widgets?"
    r")(?:[-_\s]|$)",
    re.IGNORECASE,
)


class Cleaner:
    """Clean a HTML document and expose its SEO-relevant pieces."""

    def __init__(self, html: str):
        self.soup = BeautifulSoup(html or "", "html.parser")
        self.cleaned_html = ""

    def clean(self) -> dict:
        """Run every cleanup pass and return the extracted content as JSON."""
        self.remove_noise()
        self.remove_hidden_elements()
        self.filter_preserved_tags()
        self.remove_empty_tags()
        self.cleaned_html = str(self.soup)

        return {
            "title": self.get_title(),
            "description": self.get_description(),
            "headings": self.get_headings(),
            "text": self.get_text(),
            "links": self.get_links(),
        }

    def clean_html(self) -> str:
        """Return the whitelisted HTML document."""
        if not self.cleaned_html:
            self.clean()
        return self.cleaned_html

    def remove_noise(self) -> None:
        """Discard scripts, media, forms and ad/tracking blocks entirely."""
        for tag in list(self.soup.find_all(REMOVE_TAGS)):
            if getattr(tag, "decomposed", False):
                continue
            tag.decompose()

        for tag in list(self.soup.find_all(True)):
            if not _is_active_tag(tag):
                continue
            values = self._attribute_values(tag)
            if any(NOISE_PATTERN.search(value) for value in values):
                tag.decompose()

    def remove_hidden_elements(self) -> None:
        for tag in list(self.soup.find_all(True)):
            if not _is_active_tag(tag):
                continue
            if tag.has_attr("hidden"):
                tag.decompose()
                continue
            if str(_attr(tag, "aria-hidden")).lower() == "true":
                tag.decompose()
                continue

            style = str(_attr(tag, "style")).lower()
            if re.search(r"display\s*:\s*none|visibility\s*:\s*hidden", style):
                tag.decompose()

    def filter_preserved_tags(self) -> None:
        """Keep only the requested tags; unwrap everything else."""
        kept_anchors = {
            id(tag) for tag in self.soup.find_all("a") if tag.find_parent(["header", "nav"])
        }
        kept_addresses = {
            id(tag) for tag in self.soup.find_all("address") if tag.find_parent("footer")
        }

        for tag in list(self.soup.find_all(True)):
            if not _is_active_tag(tag):
                continue
            if tag.name in {"meta", "link"}:
                if not self._is_preserved_head_tag(tag):
                    tag.decompose()
                continue
            if tag.name == "a":
                if id(tag) not in kept_anchors:
                    tag.unwrap()
                continue
            if tag.name == "address":
                if id(tag) not in kept_addresses:
                    tag.unwrap()
                continue
            if tag.name in PRESERVED_TAGS:
                continue
            if tag.name in REMOVE_TAGS:
                tag.decompose()
            else:
                tag.unwrap()

    def remove_empty_tags(self) -> None:
        for tag in list(
            self.soup.find_all(["p", "li", "section", "article", "main", "address"])
        ):
            if not _is_active_tag(tag):
                continue
            if not tag.get_text(strip=True):
                tag.decompose()

    def _is_preserved_head_tag(self, tag: Tag) -> bool:
        if tag.name == "meta":
            return any(
                all(
                    str(_attr(tag, key)).strip().lower() == value
                    for key, value in attributes.items()
                )
                for attributes in PRESERVED_META
            )
        if tag.name == "link":
            rel = _attr(tag, "rel")
            rels = " ".join(rel).lower() if isinstance(rel, (list, tuple)) else str(rel).lower()
            return "canonical" in rels.split()
        return False

    @staticmethod
    def _attribute_values(tag: Tag) -> list[str]:
        values = []
        for attribute in (
            _attr(tag, "id"),
            _attr(tag, "class"),
            _attr(tag, "data-testid"),
        ):
            if isinstance(attribute, (list, tuple)):
                values.extend(str(item) for item in attribute)
            elif attribute:
                values.append(str(attribute))
        return values

    def get_title(self) -> str:
        return self.soup.title.get_text(strip=True) if self.soup.title else ""

    def get_description(self) -> str:
        meta = self.soup.find("meta", attrs={"name": "description"})
        if not meta:
            meta = self.soup.find("meta", attrs={"property": "og:description"})
        return str(_attr(meta, "content")).strip() if meta else ""

    def get_headings(self) -> list[dict]:
        headings = []
        for tag in self.soup.find_all(["h1", "h2", "h3"]):
            text = tag.get_text(" ", strip=True)
            if text:
                headings.append({"level": int(tag.name[1]), "text": text})
        return headings

    def get_text(self) -> str:
        root: Optional[Tag] = (
            self.soup.find("main")
            or self.soup.find("article")
            or self.soup.body
            or self.soup
        )
        lines = [line.strip() for line in root.get_text("\n", strip=True).splitlines()]
        return "\n".join(line for line in lines if line)

    def get_links(self, limit: int = 100) -> list[dict]:
        links = []
        seen = set()
        for anchor in self.soup.find_all("a", href=True):
            text = anchor.get_text(" ", strip=True)
            url = str(anchor["href"]).strip()
            if not url or url in seen:
                continue
            seen.add(url)
            links.append({"text": text, "url": url})
            if len(links) >= limit:
                break
        return links
