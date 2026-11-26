"""Gemini helpers for query generation and structured comparison reports."""

import json
import logging
import re
from functools import wraps
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from app.config import get_env

logger = logging.getLogger(__name__)

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
API_REVISION = "2026-05-20"
GEMINI_MODELS = {
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
    "gemini-3.6-flash": "Gemini 3.6 Flash",
    "gemini-3.8-flash": "Gemini 3.8 Flash",
}
DEFAULT_MODEL = "gemini-3.5-flash-lite"
TIMEOUT = 60.0
MAX_HTML_CHARS = 20_000

TABLE_ROWS = (
    "Business overview",
    "Business category",
    "Products & Services",
    "Strength",
    "Weakness and Gaps",
    "Brand's competitive advantage",
    "Opportunities to improve market positioning",
    "Target Market",
)

SEARCH_PLATFORMS = ("Google", "Facebook", "Instagram", "LinkedIn", "X")
SEARCH_COLUMNS = (
    "Search Query",
    "Relevant Platform",
    "Search Intent",
    "Business Relevance",
    "Target audience for our website",
)

SERVICE_COLUMNS = (
    "Service Name",
    "Description",
    "Target Industry",
    "Target Customer",
    "Main Benefit",
)
SERVICE_SOURCES = ("website", "inferred")

AUDIENCE_LABELS = (
    "Primary Target Audience",
    "Secondary Target Audience",
    "Industry Segments",
    "Business Types",
    "Customer Profiles",
    "Location or Geographical Focus",
    "Customer Needs and Challenges",
    "Potential Customer Interests",
    "Recommended Audience Segments for Marketing",
)

SEO_GROUPS = {
    "Keyword Recommendations": (
        "Primary Keyword",
        "Secondary Keyword",
        "Long-tail keywords",
        "Relevant business keywords",
    ),
    "SEO Title": (
        "Recommended SEO Title",
        "Relevant primary keyword",
        "Improved title structure",
    ),
    "Metadata Description": (
        "Recommended meta description",
        "Short human-friendly business summary",
        "Relevant keywords",
    ),
    "Additional Support Content": (
        "Service landing page",
        "Industry-specific page",
        "FAQ",
        "Blog topics",
        "Location-based landing page",
        "Supporting business articles",
    ),
}


class LLMOutputValidationError(ValueError):
    """Raised when Gemini does not honour one of the structured-output contracts."""


def retry_invalid_structured_output(function):
    """Retry one malformed Gemini response with the same Gemini model and prompt."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except LLMOutputValidationError as first_error:
            logger.warning(
                "Gemini structured output failed validation in %s; retrying once: %s",
                function.__name__,
                first_error,
            )
            try:
                return function(*args, **kwargs)
            except LLMOutputValidationError as second_error:
                logger.error(
                    "Gemini structured output failed validation again in %s: %s",
                    function.__name__,
                    second_error,
                )
                raise LLMOutputValidationError(
                    "Gemini returned invalid structured output after one retry in "
                    f"{function.__name__}: {second_error}"
                ) from second_error

    return wrapped

QUERY_PROMPT_TEMPLATE = """You are an SEO competitor research assistant.
Based only on the website content below, write exactly {count} Google search queries
that would find competing websites, products or services in the same niche.

Rules:
- Return ONLY a JSON array of {count} strings, for example: ["query one", "query two", "query three"]
- Each query must be 3 to 8 words long.
- Use different angles: the core offering, the target audience or location, and the main problem solved.
- Do not include explanations, numbering or markdown.

WEBSITE CONTENT:
{content}
"""

TABLE_PROMPT_TEMPLATE = """You are an SEO competitive research analyst.

Compare the MAIN website with the competitor websites using the cleaned content below.

Return ONLY a JSON object in this exact shape:
{{
  "rows": [
    {{"label": "{row_example}", "values": {example_values}}}
  ]
}}

Rules:
- Include exactly these {row_count} row labels, in this order:
{rows}
- Each "values" array must contain exactly {column_count} strings, one per column in COLUMNS order.
- For "Strength" and "Weakness and Gaps", analyze SEO only: content depth, keyword and topic coverage, title and meta tags, heading structure, internal links, and unique relevant text.
- For all other rows, use plain business language.
- Keep every cell under 20 words.
- Use "—" when the content does not support an answer.
- Do not use markdown, numbering, explanations, or extra keys.

COLUMNS:
{columns}

SITES:
{sites}
"""

SEARCH_TABLE_PROMPT_TEMPLATE = """You are a search opportunity analyst.

Using only the website content below, identify {count} relevant search queries.

Return ONLY a JSON object in this exact shape:
{{
  "rows": [
    {{"values": ["query", "platform", "intent", "relevance", "audience"]}}
  ]
}}

Rules:
- Return exactly {count} rows.
- Every row must have exactly 5 values in this column order:
{columns}
- "Relevant Platform" must be exactly one of: {platforms}
- Use different platforms where possible, but never invent a platform.
- "Search Intent" must be one of: Informational, Navigational, Transactional, Commercial.
- "Business Relevance" must be high, medium, or low with a short reason.
- "Target audience for our website" must name a specific audience for this business.
- Keep every cell under 20 words.
- Do not use markdown, numbering, explanations, or extra keys.

WEBSITE CONTENT:
{content}
"""

SEO_SUGGESTIONS_PROMPT_TEMPLATE = """You are an SEO content strategist.

Using only the website content below, give practical, specific suggestions.

Return ONLY a JSON object in this exact shape:
{{
  "groups": [
    {{"title": "Group name", "items": [{{"label": "Item label", "text": "Suggestion"}}]}}
  ]
}}

Rules:
- Include exactly these groups and labels:
{groups}
- Every label must appear once under its group.
- Keep each suggestion under 30 words and specific to this business.
- For list-style labels such as long-tail keywords, relevant business keywords, blog topics, and supporting business articles, use comma-separated items.
- Do not use markdown, numbering, explanations, or extra keys.

WEBSITE CONTENT:
{content}
"""

SERVICE_TABLE_PROMPT_TEMPLATE = """You are a business analyst.

Read the website below and identify its {count} most important services.

Return ONLY a JSON object in this exact shape:
{{
  "rows": [
    {{"values": ["service name", "description", "industry", "customer", "benefit", "website or inferred"]}}
  ]
}}

Rules:
- Return exactly {count} rows, ordered from most to least important.
- Every row must have exactly 6 values: the 5 columns below, then the source.
{columns}
- The source must be exactly "website" or "inferred".
- Use "website" only when the service is clearly supported by the website content.
- Use "inferred" when the service is a reasonable recommendation rather than a stated offering.
- Keep each cell under 25 words and avoid generic filler.
- Do not use markdown, numbering, explanations, or extra keys.

WEBSITE CONTENT:
{content}
"""

AUDIENCE_PROMPT_TEMPLATE = """You are an audience research analyst.

Using the website's content, products, services and business category, identify the most relevant target audience.

Return ONLY a JSON object in this exact shape:
{{
  "items": [
    {{"label": "Item label", "text": "Answer"}}
  ]
}}

Rules:
- Include exactly these labels:
{labels}
- Every label must appear once.
- For location focus, use "Not specified" when the website gives no geographical clue.
- Keep each answer under 30 words, specific to this business, and avoid generic filler.
- Do not use markdown, numbering, explanations, or extra keys.

WEBSITE CONTENT:
{content}
"""


@retry_invalid_structured_output
def generate_search_queries(
    clean_html_path: str | Path,
    count: int = 3,
    model: str | None = None,
) -> list[str]:
    """Read cleaned homepage HTML and return `count` relevant search queries."""
    html = Path(clean_html_path).read_text(encoding="utf-8", errors="replace")
    response_text = ask_gemini(
        QUERY_PROMPT_TEMPLATE.format(count=count, content=html_to_text(html)),
        system_instruction="You answer only with valid JSON arrays of search queries.",
        model=model,
    )
    queries = parse_queries(response_text, count)

    if len(queries) < count:
        raise LLMOutputValidationError(
            f"Gemini returned only {len(queries)} of {count} search queries."
        )

    return queries[:count]


@retry_invalid_structured_output
def generate_comparison_table(
    sites: list[dict],
    column_count: int = 4,
    model: str | None = None,
) -> dict:
    """Build a dynamically sized company comparison table from cleaned content."""
    columns = []
    for site in sites[:column_count]:
        name = site.get("name") or hostname(site.get("url", ""))
        columns.append(
            {
                "name": name,
                "url": site.get("url", ""),
                "role": site.get("role", "competitor"),
            }
        )

    while len(columns) < column_count:
        number = len(columns)
        columns.append(
            {
                "name": f"Competitor {number}" if number > 1 else "Competitor",
                "url": "",
                "role": "competitor",
            }
        )

    columns = columns[:column_count]
    columns_text = "\n".join(
        f"{index + 1}. {column['name']} — {column['url'] or 'no scraped URL'}"
        for index, column in enumerate(columns)
    )
    rows_text = "\n".join(f"- {label}" for label in TABLE_ROWS)

    site_sections = []
    for index, site in enumerate(sites[:column_count]):
        clean_path = site.get("clean_html")
        if not clean_path or not Path(clean_path).exists():
            continue
        html = Path(clean_path).read_text(encoding="utf-8", errors="replace")
        role = "MAIN" if site.get("role") == "main" else "COMPETITOR"
        site_sections.append(
            f"{role} [{columns[index]['name']}] {site.get('url', '')}:\n{html_to_text(html)}"
        )

    response_text = ask_gemini(
        TABLE_PROMPT_TEMPLATE.format(
            row_example=TABLE_ROWS[0],
            example_values=json.dumps(["cell"] * column_count),
            row_count=len(TABLE_ROWS),
            rows=rows_text,
            column_count=column_count,
            columns=columns_text,
            sites="\n\n".join(site_sections) or "No usable site content.",
        ),
        system_instruction="You answer only with one valid JSON object.",
        max_output_tokens=max(1800, column_count * 600),
        model=model,
    )

    raw_rows = parse_table_rows(response_text, column_count)
    required_labels = {normalize_label(label) for label in TABLE_ROWS}
    if set(raw_rows) != required_labels:
        raise LLMOutputValidationError(
            "The comparison response did not include every required row."
        )
    rows = []
    for label in TABLE_ROWS:
        values = raw_rows[normalize_label(label)]
        rows.append({"label": label, "values": values})

    return {"rowHeader": True, "columns": columns, "rows": rows}


@retry_invalid_structured_output
def generate_search_query_table(
    clean_html_path: str | Path,
    count: int = 5,
    model: str | None = None,
) -> dict:
    """Build section 2: five queries with platform, intent, and audience."""
    html = Path(clean_html_path).read_text(encoding="utf-8", errors="replace")
    columns_text = "\n".join(f"- {column}" for column in SEARCH_COLUMNS)

    response_text = ask_gemini(
        SEARCH_TABLE_PROMPT_TEMPLATE.format(
            count=count,
            columns=columns_text,
            platforms=", ".join(SEARCH_PLATFORMS),
            content=html_to_text(html),
        ),
        system_instruction="You answer only with one valid JSON object.",
        max_output_tokens=1200,
        model=model,
    )

    rows = parse_value_rows(response_text, count=count, column_count=len(SEARCH_COLUMNS))
    normalized = []
    for values in rows:
        if values[1] not in SEARCH_PLATFORMS:
            raise LLMOutputValidationError(
                "A search-query row used an unsupported platform."
            )
        if values[2] not in {"Informational", "Navigational", "Transactional", "Commercial"}:
            raise LLMOutputValidationError(
                "A search-query row used an unsupported search intent."
            )
        normalized.append({"label": values[0], "values": values})

    return {
        "rowHeader": False,
        "columns": [{"name": column} for column in SEARCH_COLUMNS],
        "rows": normalized,
    }


@retry_invalid_structured_output
def generate_seo_suggestions(clean_html_path: str | Path, model: str | None = None) -> dict:
    """Build section 3: grouped SEO suggestions as bullet-friendly data."""
    html = Path(clean_html_path).read_text(encoding="utf-8", errors="replace")
    groups_text = "\n".join(
        f"- {title}: {', '.join(labels)}" for title, labels in SEO_GROUPS.items()
    )

    response_text = ask_gemini(
        SEO_SUGGESTIONS_PROMPT_TEMPLATE.format(
            groups=groups_text,
            content=html_to_text(html),
        ),
        system_instruction="You answer only with one valid JSON object.",
        max_output_tokens=1800,
        model=model,
    )

    parsed = load_json_object(response_text) or {}
    raw_groups = parsed.get("groups") if isinstance(parsed, dict) else None
    if not isinstance(raw_groups, list):
        raw_groups = []

    suggestions = {}
    for group in raw_groups:
        if not isinstance(group, dict):
            continue
        title_key = normalize_label(str(group.get("title") or ""))
        items = group.get("items")
        if not title_key or not isinstance(items, list):
            continue

        label_map = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            label_key = normalize_label(str(item.get("label") or ""))
            text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
            if label_key:
                label_map[label_key] = text
        suggestions[title_key] = label_map

    groups = []
    for title, labels in SEO_GROUPS.items():
        label_map = suggestions.get(normalize_label(title), {})
        missing = [
            label
            for label in labels
            if not is_meaningful_value(label_map.get(normalize_label(label), ""))
        ]
        if missing:
            raise LLMOutputValidationError(
                f"The SEO response is missing usable values for: {', '.join(missing)}."
            )
        groups.append(
            {
                "title": title,
                "items": [
                    {
                        "label": label,
                        "text": label_map[normalize_label(label)],
                    }
                    for label in labels
                ],
            }
        )

    return {"groups": groups}


@retry_invalid_structured_output
def generate_services_table(
    clean_html_path: str | Path,
    count: int = 5,
    model: str | None = None,
) -> dict:
    """Build section 4: the website's five most important services."""
    html = Path(clean_html_path).read_text(encoding="utf-8", errors="replace")
    columns_text = "\n".join(f"- {column}" for column in SERVICE_COLUMNS)

    response_text = ask_gemini(
        SERVICE_TABLE_PROMPT_TEMPLATE.format(
            count=count,
            columns=columns_text,
            content=html_to_text(html),
        ),
        system_instruction="You answer only with one valid JSON object.",
        max_output_tokens=1200,
        model=model,
    )

    rows = parse_value_rows(
        response_text,
        count=count,
        column_count=len(SERVICE_COLUMNS) + 1,
    )
    for values in rows:
        if values[-1].lower() not in SERVICE_SOURCES:
            raise LLMOutputValidationError(
                "A service row did not declare whether it came from the website or inference."
            )
        values[-1] = normalize_source(values[-1])

    return {
        "rowHeader": False,
        "columns": [{"name": column} for column in SERVICE_COLUMNS],
        "rows": [{"label": values[0], "values": values} for values in rows],
    }


@retry_invalid_structured_output
def generate_audience_profile(
    clean_html_path: str | Path,
    model: str | None = None,
) -> dict:
    """Build section 5: the most relevant target audience profile."""
    html = Path(clean_html_path).read_text(encoding="utf-8", errors="replace")
    labels_text = "\n".join(f"- {label}" for label in AUDIENCE_LABELS)

    response_text = ask_gemini(
        AUDIENCE_PROMPT_TEMPLATE.format(
            labels=labels_text,
            content=html_to_text(html),
        ),
        system_instruction="You answer only with one valid JSON object.",
        max_output_tokens=1400,
        model=model,
    )

    parsed = load_json_object(response_text) or {}
    raw_items = parsed.get("items") if isinstance(parsed, dict) else None
    if not isinstance(raw_items, list):
        raw_items = []

    answers = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        label_key = normalize_label(str(item.get("label") or ""))
        text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
        if label_key:
            answers[label_key] = text

    missing = [
        label
        for label in AUDIENCE_LABELS
        if not is_meaningful_value(answers.get(normalize_label(label), ""))
    ]
    if missing:
        raise LLMOutputValidationError(
            f"The audience response is missing usable values for: {', '.join(missing)}."
        )

    return {
        "groups": [
            {
                "title": "Target Audience",
                "items": [
                    {
                        "label": label,
                        "text": answers[normalize_label(label)],
                    }
                    for label in AUDIENCE_LABELS
                ],
            }
        ]
    }


def resolve_model(model: str | None = None) -> str:
    """Choose the UI model first, then the env override, then the default."""
    requested = str(model or "").strip()
    if requested in GEMINI_MODELS:
        return requested
    if requested:
        raise ValueError(f"Unsupported Gemini model: {requested}")

    configured = get_env("GEMINI_MODEL").strip()
    return configured or DEFAULT_MODEL


def ask_gemini(
    prompt: str,
    system_instruction: str | None = None,
    max_output_tokens: int = 600,
    model: str | None = None,
) -> str:
    """Send a prompt to Gemini and return its text output."""
    api_key = get_env("GEMINI_API")
    if not api_key or api_key.startswith("<"):
        raise RuntimeError("GEMINI_API is missing or still a placeholder in .env")

    payload = {
        "model": resolve_model(model),
        "input": prompt,
        "generation_config": {"max_output_tokens": max_output_tokens},
    }
    if system_instruction:
        payload["system_instruction"] = system_instruction

    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
        "Api-Revision": API_REVISION,
    }

    response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=TIMEOUT)
    if response.status_code >= 400:
        raise RuntimeError(
            f"Gemini request failed ({response.status_code}): {response.text[:300]}"
        )

    text = extract_text(response.json())
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text


def html_to_text(html: str) -> str:
    """Convert HTML to compact text suitable for a prompt."""
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()

    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
    return text[:MAX_HTML_CHARS]


def hostname(url: str) -> str:
    """Return a readable host name for a URL."""
    return urlparse(url).netloc.replace("www.", "") or "Website"


def extract_text(data: dict) -> str:
    """Read text from both the Interactions and legacy generateContent shapes."""
    if not isinstance(data, dict):
        return ""

    for key in ("output_text", "outputText"):
        if isinstance(data.get(key), str):
            return data[key].strip()

    texts = []
    for step in data.get("steps") or []:
        if not isinstance(step, dict):
            continue
        content = step.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    texts.append(item["text"])
                elif isinstance(item, str):
                    texts.append(item)
    if texts:
        return "".join(texts).strip()

    for candidate in data.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        for part in content.get("parts") or []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
    return "".join(texts).strip()


def parse_queries(text: str, count: int) -> list[str]:
    """Parse a JSON array of queries, tolerating code fences and stray text."""
    parsed = load_json_array(text)

    if not isinstance(parsed, list):
        parsed = re.findall(r'"([^"]+)"', text)

    queries = []
    seen = set()
    for item in parsed:
        query = re.sub(r"^\s*\d+[.)]\s*", "", str(item)).strip().strip('"')
        if query and query.lower() not in seen:
            seen.add(query.lower())
            queries.append(query)
        if len(queries) >= count:
            break
    return queries


def parse_table_rows(text: str, column_count: int) -> dict[str, list[str]]:
    """Parse row labels and cell values from Gemini's JSON object."""
    parsed = load_json_object(text) or {}
    items = parsed.get("rows") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        raise LLMOutputValidationError("The comparison response is not a JSON rows object.")

    rows = {}
    for item in items:
        if not isinstance(item, dict):
            raise LLMOutputValidationError("The comparison response contains an invalid row.")
        label = str(item.get("label") or "").strip()
        values = item.get("values")
        if not label or not isinstance(values, list) or len(values) != column_count:
            raise LLMOutputValidationError("A comparison row has the wrong shape.")
        label_key = normalize_label(label)
        if label_key in rows:
            raise LLMOutputValidationError("The comparison response contains duplicate rows.")

        cells = []
        for value in values:
            cell = re.sub(r"\s+", " ", str(value or "")).strip()
            if not cell:
                raise LLMOutputValidationError("A comparison row contains an empty cell.")
            cells.append(cell)

        rows[label_key] = cells

    return rows


def parse_value_rows(text: str, count: int, column_count: int) -> list[list[str]]:
    """Parse plain value rows from Gemini's JSON object."""
    parsed = load_json_object(text) or {}
    items = parsed.get("rows") if isinstance(parsed, dict) else None
    if not isinstance(items, list) or len(items) != count:
        raise LLMOutputValidationError(
            f"The model response must contain exactly {count} rows."
        )

    rows = []
    for item in items:
        if isinstance(item, dict):
            values = item.get("values")
        elif isinstance(item, list):
            # Gemini sometimes returns the documented row values directly
            # instead of wrapping them in a {"values": [...]} object.
            values = item
        else:
            raise LLMOutputValidationError("The model response contains an invalid row.")
        if not isinstance(values, list) or len(values) != column_count:
            raise LLMOutputValidationError("A model row has the wrong number of values.")

        cells = []
        for value in values:
            cell = re.sub(r"\s+", " ", str(value or "")).strip()
            if not is_meaningful_value(cell):
                raise LLMOutputValidationError("A model row contains an empty placeholder value.")
            cells.append(cell)
        rows.append(cells)

    return rows


def is_meaningful_value(value: object) -> bool:
    """Reject missing values and placeholder tokens before building a report."""
    return str(value or "").strip() not in {"", "—", "-"}


def normalize_source(value: str) -> str:
    """Only trust an explicit website citation; everything else is inferred."""
    text = str(value or "").strip().lower()
    return "website" if text == "website" else "inferred"


def normalize_platform(value: str) -> str:
    """Force a platform cell into the supported platform list."""
    text = str(value or "").strip().lower()
    aliases = {
        "google": "Google",
        "facebook": "Facebook",
        "meta": "Facebook",
        "instagram": "Instagram",
        "linkedin": "LinkedIn",
        "linked in": "LinkedIn",
        "x": "X",
        "twitter": "X",
    }
    if text in aliases:
        return aliases[text]
    for platform in SEARCH_PLATFORMS:
        if platform.lower() in text:
            return platform
    return "Google"


def load_json_array(text: str):
    """Extract a JSON array from a model reply that may contain extra text."""
    cleaned = strip_code_fences(text)
    match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def load_json_object(text: str):
    """Extract a JSON object from a model reply that may contain extra text."""
    cleaned = strip_code_fences(text)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def strip_code_fences(text: str) -> str:
    """Remove markdown fences from a model reply."""
    cleaned = (text or "").strip()
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)


def normalize_label(label: str) -> str:
    """Make row labels comparable without worrying about punctuation."""
    return re.sub(r"[^a-z0-9]+", " ", str(label).lower()).strip()
