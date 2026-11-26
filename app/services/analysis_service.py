"""The single streaming endpoint that runs the full link-analysis pipeline."""

import json
import logging
import re
from collections.abc import Iterator
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import AliasChoices, BaseModel, Field

from app.services.competitor_service import find_competitors
from app.services.llm_service import (
    LLMOutputValidationError,
    SEO_GROUPS,
    generate_audience_profile,
    generate_comparison_table,
    generate_search_query_table,
    generate_seo_suggestions,
    generate_services_table,
    normalize_label,
    normalize_platform,
    normalize_source,
    resolve_model,
)
from app.services.website_extractor import extract_homepage, validate_link
from app.utils.validators import ValidationResult, Validator

router = APIRouter(tags=["link-analysis"])
logger = logging.getLogger(__name__)

SECTION_ONE_TITLE = "Competitive overview"
SECTION_TWO_TITLE = "Search opportunity map"
SECTION_THREE_TITLE = "SEO recommendations"
SECTION_FOUR_TITLE = "Core services"
SECTION_FIVE_TITLE = "Target audience"


class LinkRequest(BaseModel):
    link: str = Field(
        default="",
        validation_alias=AliasChoices("url", "link"),
        description="The website to analyse",
    )
    max_competitors: int = Field(
        default=3,
        ge=3,
        le=10,
        description="Number of researched competitors to include (3 to 10)",
    )
    model: str | None = Field(
        default=None,
        description="Gemini model id selected in the UI",
    )


@router.post("/link-analysis", response_model=None)
def link_analysis(payload: LinkRequest, stream: bool = False) -> StreamingResponse | dict:
    """Return JSON by default; the UI opts into SSE progress with ?stream=true."""
    # Validate before opening a stream so bad links return a real 400/502.
    validated = validate_link(payload.link)
    if stream:
        return StreamingResponse(
            run_pipeline(payload, validated=validated),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    return collect_result(payload, validated=validated)


def collect_result(
    payload: LinkRequest,
    validated: tuple[Validator, ValidationResult] | None = None,
) -> dict:
    """Run the pipeline without emitting progress and return the final JSON."""
    result = None
    for frame in run_pipeline(payload, validated=validated):
        event, data = parse_sse_frame(frame)
        if event == "result":
            result = data
        elif event == "error":
            raise HTTPException(
                status_code=int(data.get("status_code") or 500),
                detail=data,
            )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Analysis finished without a result."},
        )
    return result


def parse_sse_frame(frame: str) -> tuple[str, dict]:
    """Read one server-sent event into an event name and JSON payload."""
    event = "message"
    data_lines = []
    for line in frame.splitlines():
        if line.startswith("event: "):
            event = line[7:].strip()
        elif line.startswith("data: "):
            data_lines.append(line[6:])

    data = json.loads("\n".join(data_lines)) if data_lines else {}
    return event, data


def run_pipeline(
    payload: LinkRequest,
    validated: tuple[Validator, ValidationResult] | None = None,
) -> Iterator[str]:
    try:
        model = resolve_model(payload.model)
        if validated is None:
            yield step_event(0, "active", "Checking the link format")
            validated = validate_link(payload.link)
        yield step_event(0, "complete", "Valid HTTP or HTTPS URL")

        yield step_event(1, "active", "Fetching and cleaning the homepage")
        homepage = extract_homepage(payload.link, validated=validated)
        yield step_event(1, "complete", "Homepage saved and cleaned")

        yield step_event(2, "active", "Generating queries and searching competitors")
        competitors = find_competitors(
            homepage["clean_html"],
            count=3,
            max_competitors=payload.max_competitors,
            main_url=homepage["final_url"],
            model=model,
        )
        yield step_event(
            2,
            "complete",
            f"Found {len(competitors['competitors'])} competitor(s)",
        )

        sites = [
            {
                "url": homepage["final_url"],
                "name": homepage["title"] or hostname(homepage["final_url"]),
                "clean_html": homepage["clean_html"],
                "role": "main",
            }
        ]
        sites.extend(
            {
                "url": competitor["url"],
                "name": competitor["title"] or hostname(competitor["url"]),
                "clean_html": competitor["clean_html"],
                "role": "competitor",
            }
            for competitor in competitors["competitors"]
        )

        yield section_start_event(1, SECTION_ONE_TITLE)
        comparison_table = generate_comparison_table(
            sites,
            column_count=len(sites),
            model=model,
        )
        yield section_done_event(1, SECTION_ONE_TITLE)

        yield section_start_event(2, SECTION_TWO_TITLE)
        search_table = generate_search_query_table(homepage["clean_html"], model=model)
        yield section_done_event(2, SECTION_TWO_TITLE)

        yield section_start_event(3, SECTION_THREE_TITLE)
        seo_suggestions = generate_seo_suggestions(homepage["clean_html"], model=model)
        yield section_done_event(3, SECTION_THREE_TITLE)

        yield section_start_event(4, SECTION_FOUR_TITLE)
        services_table = generate_services_table(homepage["clean_html"], model=model)
        yield section_done_event(4, SECTION_FOUR_TITLE)

        yield section_start_event(5, SECTION_FIVE_TITLE)
        audience_suggestions = generate_audience_profile(homepage["clean_html"], model=model)
        yield section_done_event(5, SECTION_FIVE_TITLE)

        yield sse(
            "result",
            build_structured_response(
                homepage,
                comparison_table,
                search_table,
                seo_suggestions,
                services_table,
                audience_suggestions,
            ),
        )
    except LLMOutputValidationError as exc:
        logger.error("Link analysis stopped after Gemini validation failure: %s", exc)
        yield sse(
            "error",
            {
                "message": (
                    "The language model returned incomplete structured data. "
                    "Please retry the analysis."
                ),
                "status_code": 502,
            },
        )
    except HTTPException as exc:
        detail = exc.detail
        message = detail.get("error") if isinstance(detail, dict) else str(detail)
        yield sse("error", {"message": message, "status_code": exc.status_code})
    except Exception as exc:
        yield sse(
            "error",
            {"message": str(exc) or "Analysis failed.", "status_code": 500},
        )


def build_structured_response(
    homepage: dict,
    comparison_table: dict,
    search_table: dict,
    seo_suggestions: dict,
    services_table: dict,
    audience_suggestions: dict,
) -> dict:
    """Shape all generated analysis into the requested API response."""
    columns = comparison_table.get("columns") or []
    brand_name = homepage.get("title") or hostname(homepage.get("final_url", ""))
    summary = (
        row_value(comparison_table, "Business overview", 0)
        or clean_text((homepage.get("content") or {}).get("description"))
        or ""
    )
    category = row_value(comparison_table, "Business category", 0)

    competitor_rows = []
    for index, column in enumerate(columns[1:], start=1):
        competitor_rows.append(
            {
                "name": column.get("name") or f"Competitor {index}",
                "url": column.get("url") or "",
                "overview": row_value(comparison_table, "Business overview", index),
                "category": row_value(comparison_table, "Business category", index),
                "products_services": row_value(
                    comparison_table, "Products & Services", index
                ),
                "strength": row_value(comparison_table, "Strength", index),
                "weakness_and_gaps": row_value(
                    comparison_table, "Weakness and Gaps", index
                ),
                "competitive_advantage": row_value(
                    comparison_table, "Brand's competitive advantage", index
                ),
                "opportunities": row_value(
                    comparison_table, "Opportunities to improve market positioning", index
                ),
                "target_market": row_value(comparison_table, "Target Market", index),
                "source": "Serper + Gemini" if column.get("url") else "Gemini suggestion",
                "research_backed": bool(column.get("url")),
            }
        )

    search_queries = []
    for row in search_table.get("rows") or []:
        values = row.get("values") or []
        if not values or not clean_text(values[0]):
            continue
        search_queries.append(
            {
                "query": clean_text(values[0]),
                "platforms": split_platforms(values[1] if len(values) > 1 else ""),
                "intent": clean_text(values[2] if len(values) > 2 else ""),
                "business_relevance": clean_text(values[3] if len(values) > 3 else ""),
                "audience": clean_text(values[4] if len(values) > 4 else ""),
            }
        )

    keyword_items = suggestion_items(seo_suggestions, "Keyword Recommendations")
    title_items = suggestion_items(seo_suggestions, "SEO Title")
    meta_items = suggestion_items(seo_suggestions, "Metadata Description")
    content_items = suggestion_items(seo_suggestions, "Additional Support Content")

    supporting_content = []
    support_details = {}
    for label in SEO_GROUPS["Additional Support Content"]:
        text = content_items.get(normalize_label(label), "")
        support_details[label] = text
        supporting_content.extend(as_list(text))

    audience_items = suggestion_items(audience_suggestions, "Target Audience")

    top_services = []
    for row in services_table.get("rows") or []:
        values = row.get("values") or []
        if not values or not clean_text(values[0]):
            continue
        service_name = clean_text(values[0])
        source = normalize_source(values[5] if len(values) > 5 else "inferred")
        top_services.append(
            {
                "name": (
                    service_name
                    if source == "website"
                    else f"{service_name} (AI-generated suggestion)"
                ),
                "service_name": service_name,
                "source": source,
                "description": clean_text(values[1] if len(values) > 1 else ""),
                "target_industry": clean_text(values[2] if len(values) > 2 else ""),
                "target_customer": clean_text(values[3] if len(values) > 3 else ""),
                "main_benefit": clean_text(values[4] if len(values) > 4 else ""),
            }
        )

    return {
        "success": True,
        "website": homepage.get("final_url", ""),
        "brand": {
            "name": brand_name,
            "category": category,
            "summary": summary,
        },
        "analysis": {
            "competitor_analysis": {
                "brand_overview": summary,
                "source_note": (
                    "Competitor URLs were discovered with Serper. Comparison text "
                    "was generated by Gemini from scraped page content and should "
                    "be treated as AI-generated analysis, not audited facts."
                ),
                "competitors": competitor_rows,
                "strengths": prefixed_values(comparison_table, "Strength", columns),
                "opportunities": prefixed_values(
                    comparison_table, "Opportunities to improve market positioning", columns
                ),
                "gaps": prefixed_values(comparison_table, "Weakness and Gaps", columns),
                "competitive_advantages": prefixed_values(
                    comparison_table, "Brand's competitive advantage", columns
                ),
                "target_markets": prefixed_values(
                    comparison_table, "Target Market", columns
                ),
                "table": comparison_table,
            },
            "target_search_queries": search_queries,
            "seo_improvements": {
                "primary_keyword": keyword_items.get(
                    normalize_label("Primary Keyword"), ""
                ),
                "secondary_keywords": as_list(
                    keyword_items.get(normalize_label("Secondary Keyword"), "")
                ),
                "long_tail_keywords": as_list(
                    keyword_items.get(normalize_label("Long-tail keywords"), "")
                ),
                "business_keywords": as_list(
                    keyword_items.get(normalize_label("Relevant business keywords"), "")
                ),
                "recommended_title": title_items.get(
                    normalize_label("Recommended SEO Title"), ""
                ),
                "relevant_primary_keyword": title_items.get(
                    normalize_label("Relevant primary keyword"), ""
                ),
                "title_structure": title_items.get(
                    normalize_label("Improved title structure"), ""
                ),
                "meta_description": meta_items.get(
                    normalize_label("Recommended meta description"), ""
                ),
                "short_summary": meta_items.get(
                    normalize_label("Short human-friendly business summary"), ""
                ),
                "relevant_keywords": as_list(
                    meta_items.get(normalize_label("Relevant keywords"), "")
                ),
                "supporting_content": supporting_content,
                "support_content_details": support_details,
            },
            "top_services": top_services,
            "top_services_note": (
                "Services marked '(AI-generated suggestion)' were inferred by "
                "Gemini and are not stated offerings on the website."
            ),
            "target_audience": {
                "primary_audience": as_list(
                    audience_items.get(normalize_label("Primary Target Audience"), "")
                ),
                "secondary_audience": as_list(
                    audience_items.get(normalize_label("Secondary Target Audience"), "")
                ),
                "industries": as_list(
                    audience_items.get(normalize_label("Industry Segments"), "")
                ),
                "business_types": as_list(
                    audience_items.get(normalize_label("Business Types"), "")
                ),
                "customer_profiles": as_list(
                    audience_items.get(normalize_label("Customer Profiles"), "")
                ),
                "location_focus": as_list(
                    audience_items.get(
                        normalize_label("Location or Geographical Focus"), ""
                    )
                ),
                "customer_needs": as_list(
                    audience_items.get(
                        normalize_label("Customer Needs and Challenges"), ""
                    )
                ),
                "customer_interests": as_list(
                    audience_items.get(
                        normalize_label("Potential Customer Interests"), ""
                    )
                ),
                "recommended_segments": as_list(
                    audience_items.get(
                        normalize_label("Recommended Audience Segments for Marketing"), ""
                    )
                ),
            },
        },
    }


def row_value(table: dict, label: str, index: int) -> str:
    """Read one cell from a labelled table row."""
    for row in table.get("rows") or []:
        if normalize_label(row.get("label", "")) != normalize_label(label):
            continue
        values = row.get("values") or []
        return clean_text(values[index]) if index < len(values) else ""
    return ""


def prefixed_values(table: dict, label: str, columns: list[dict]) -> list[str]:
    """Return a labelled row as brand-prefixed strings."""
    values = []
    for index, column in enumerate(columns):
        value = row_value(table, label, index)
        if value:
            values.append(f"{column.get('name') or 'Website'}: {value}")
    return values


def suggestion_items(data: dict, group_title: str) -> dict[str, str]:
    """Flatten one suggestion group into normalized labels."""
    for group in data.get("groups") or []:
        if normalize_label(group.get("title", "")) != normalize_label(group_title):
            continue
        items = {}
        for item in group.get("items") or []:
            label = normalize_label(item.get("label", ""))
            if label:
                items[label] = clean_text(item.get("text", ""))
        return items
    return {}


def as_list(value) -> list[str]:
    """Turn a string or list into a clean list of strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = clean_text(value)
    if not text:
        return []
    return [
        part.strip()
        for part in re.split(r"[,;\n]+", text)
        if part.strip()
    ]


def split_platforms(value) -> list[str]:
    """Normalize platform names and remove duplicates."""
    return list(dict.fromkeys(normalize_platform(item) for item in as_list(value)))


def clean_text(value) -> str:
    """Normalize a model value into display-ready text."""
    text = str(value or "").strip()
    return "" if text in {"", "—", "-"} else text


def hostname(url: str) -> str:
    """Readable fallback name for a site without a title."""
    return urlparse(url).netloc.replace("www.", "") or "Website"


def step_event(step: int, state: str, description: str) -> str:
    return sse("step", {"step": step, "state": state, "description": description})


def section_start_event(index: int, title: str) -> str:
    return sse("section_start", {"index": index, "title": title})


def section_done_event(index: int, title: str) -> str:
    return sse("section_done", {"index": index, "title": title})


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
