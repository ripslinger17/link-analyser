"""Offline regression tests for the Link Analyser API contract."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests
from fastapi import HTTPException
from pydantic import ValidationError

from app.services.analysis_service import (
    LinkRequest,
    build_structured_response,
    collect_result,
    link_analysis,
    parse_sse_frame,
    run_pipeline,
    sse,
)
from app.services.competitor_service import collect_competitor_urls, search_queries
from app.services.llm_service import (
    LLMOutputValidationError,
    generate_audience_profile,
    generate_comparison_table,
    generate_seo_suggestions,
    generate_services_table,
    parse_value_rows,
)
from app.utils.validators import BlockedURLError, Validator, request_public_url

PUBLIC_ADDRESS = [(2, 1, 6, "", ("93.184.216.34", 443))]


def make_group(title, values):
    return {
        "title": title,
        "items": [
            {"label": label, "text": text} for label, text in values.items()
        ],
    }


def sample_tables():
    comparison_columns = [
        {"name": "Brand", "url": "https://brand.example", "role": "main"},
        {"name": "Rival A", "url": "https://rival-a.example", "role": "competitor"},
        {"name": "Rival B", "url": "https://rival-b.example", "role": "competitor"},
        {"name": "Rival C", "url": "https://rival-c.example", "role": "competitor"},
    ]
    comparison_rows = [
        {
            "label": label,
            "values": [f"{label} — brand", f"{label} — A", f"{label} — B", f"{label} — C"],
        }
        for label in (
            "Business overview",
            "Business category",
            "Products & Services",
            "Strength",
            "Weakness and Gaps",
            "Brand's competitive advantage",
            "Opportunities to improve market positioning",
            "Target Market",
        )
    ]

    search_rows = [
        {
            "values": [
                f"query {index}",
                platform,
                "Informational",
                "high — relevant",
                f"audience {index}",
            ]
        }
        for index, platform in enumerate(
            ["Google", "Facebook", "Instagram", "LinkedIn", "X"], start=1
        )
    ]

    seo_groups = [
        make_group(
            "Keyword Recommendations",
            {
                "Primary Keyword": "seo audit",
                "Secondary Keyword": "technical seo, content audit",
                "Long-tail keywords": "how to audit a website",
                "Relevant business keywords": "seo tools, ranking report",
            },
        ),
        make_group(
            "SEO Title",
            {
                "Recommended SEO Title": "SEO Audit Services",
                "Relevant primary keyword": "seo audit",
                "Improved title structure": "Service | Brand | Location",
            },
        ),
        make_group(
            "Metadata Description",
            {
                "Recommended meta description": "Practical SEO audits.",
                "Short human-friendly business summary": "We improve rankings.",
                "Relevant keywords": "seo audit, rankings",
            },
        ),
        make_group(
            "Additional Support Content",
            {
                "Service landing page": "Create an audit landing page",
                "Industry-specific page": "Create a SaaS page",
                "FAQ": "Answer common audit questions",
                "Blog topics": "Technical SEO checklist",
                "Location-based landing page": "Create a city page",
                "Supporting business articles": "Link-building guide",
            },
        ),
    ]

    service_rows = [
        {
            "values": [
                f"Service {index}",
                f"Description {index}",
                f"Industry {index}",
                f"Customer {index}",
                f"Benefit {index}",
                "website" if index < 5 else "inferred",
            ]
        }
        for index in range(1, 6)
    ]

    audience_labels = (
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
    audience = {
        "groups": [
            make_group(
                "Target Audience",
                {label: f"answer for {label.lower()}" for label in audience_labels},
            )
        ]
    }

    return {
        "comparison": {"rowHeader": True, "columns": comparison_columns, "rows": comparison_rows},
        "search": {"rowHeader": False, "rows": search_rows},
        "seo": {"groups": seo_groups},
        "services": {"rowHeader": False, "rows": service_rows},
        "audience": audience,
    }


class RequestContractTests(unittest.TestCase):
    def test_accepts_spec_url_key(self):
        payload = LinkRequest(url="https://example.com")
        self.assertEqual(payload.link, "https://example.com")

    def test_accepts_legacy_link_key(self):
        payload = LinkRequest(link="https://example.com")
        self.assertEqual(payload.link, "https://example.com")

    def test_competitor_count_is_limited_to_three_through_ten(self):
        self.assertEqual(
            LinkRequest(url="https://example.com", max_competitors=10).max_competitors,
            10,
        )
        for count in (2, 11):
            with self.subTest(count=count), self.assertRaises(ValidationError):
                LinkRequest(url="https://example.com", max_competitors=count)

    def test_default_json_mode_returns_final_object(self):
        payload = LinkRequest(url="https://example.com")
        frames = [
            sse("step", {"step": 0, "state": "complete", "description": "done"}),
            sse("result", {"success": True, "answer": "ready"}),
        ]
        with patch(
            "app.services.analysis_service.run_pipeline", return_value=iter(frames)
        ) as pipeline:
            result = collect_result(payload, validated=(MagicMock(), MagicMock()))

        self.assertEqual(result, {"success": True, "answer": "ready"})
        pipeline.assert_called_once()

    def test_validation_failure_is_a_real_http_error(self):
        payload = LinkRequest(url="http://127.0.0.1")
        error = HTTPException(status_code=400, detail={"error": "blocked"})
        with patch(
            "app.services.analysis_service.validate_link", side_effect=error
        ):
            with self.assertRaises(HTTPException) as caught:
                link_analysis(payload)

        self.assertEqual(caught.exception.status_code, 400)

    def test_only_one_api_path_is_exposed(self):
        from app.main import app

        self.assertEqual(list(app.openapi()["paths"]), ["/link-analysis"])

    def test_docs_endpoint_is_exposed(self):
        from app.main import app

        self.assertIn("/docs", {getattr(route, "path", None) for route in app.routes})


class ModelSelectionTests(unittest.TestCase):
    def test_selected_model_is_used_by_every_gemini_call(self):
        tables = sample_tables()
        seen_models = []

        def capture(name, value):
            def call(*args, **kwargs):
                seen_models.append(kwargs.get("model"))
                return value

            return call

        with tempfile.TemporaryDirectory() as directory:
            page = Path(directory) / "clean.html"
            page.write_text("<title>Brand</title>", encoding="utf-8")
            homepage = {
                "final_url": "https://brand.example",
                "title": "Brand",
                "content": {"description": "A sample brand."},
                "clean_html": str(page),
            }
            competitors = {
                "competitors": [
                    {
                        "url": f"https://rival-{index}.example",
                        "title": f"Rival {index}",
                        "clean_html": str(page),
                    }
                    for index in range(1, 4)
                ]
            }
            payload = LinkRequest(
                url="https://brand.example", model="gemini-3.8-flash"
            )

            with patch(
                "app.services.analysis_service.extract_homepage", return_value=homepage
            ), patch(
                "app.services.analysis_service.find_competitors",
                side_effect=capture("competitors", competitors),
            ), patch(
                "app.services.analysis_service.generate_comparison_table",
                side_effect=capture("comparison", tables["comparison"]),
            ), patch(
                "app.services.analysis_service.generate_search_query_table",
                side_effect=capture("search", tables["search"]),
            ), patch(
                "app.services.analysis_service.generate_seo_suggestions",
                side_effect=capture("seo", tables["seo"]),
            ), patch(
                "app.services.analysis_service.generate_services_table",
                side_effect=capture("services", tables["services"]),
            ), patch(
                "app.services.analysis_service.generate_audience_profile",
                side_effect=capture("audience", tables["audience"]),
            ):
                for _ in run_pipeline(payload, validated=(MagicMock(), MagicMock())):
                    pass

        self.assertEqual(len(seen_models), 6)
        self.assertEqual(set(seen_models), {"gemini-3.8-flash"})

    def test_comparison_expands_to_every_requested_competitor(self):
        tables = sample_tables()
        captured = {}

        def comparison(*args, **kwargs):
            captured["sites"] = args[0]
            captured["column_count"] = kwargs["column_count"]
            return tables["comparison"]

        with tempfile.TemporaryDirectory() as directory:
            page = Path(directory) / "clean.html"
            page.write_text("<title>Brand</title>", encoding="utf-8")
            homepage = {
                "final_url": "https://brand.example",
                "title": "Brand",
                "content": {"description": "A sample brand."},
                "clean_html": str(page),
            }
            competitors = {
                "competitors": [
                    {
                        "url": f"https://rival-{index}.example",
                        "title": f"Rival {index}",
                        "clean_html": str(page),
                    }
                    for index in range(1, 11)
                ]
            }
            payload = LinkRequest(url="https://brand.example", max_competitors=10)

            with patch(
                "app.services.analysis_service.extract_homepage", return_value=homepage
            ), patch(
                "app.services.analysis_service.find_competitors", return_value=competitors
            ), patch(
                "app.services.analysis_service.generate_comparison_table",
                side_effect=comparison,
            ), patch(
                "app.services.analysis_service.generate_search_query_table",
                return_value=tables["search"],
            ), patch(
                "app.services.analysis_service.generate_seo_suggestions",
                return_value=tables["seo"],
            ), patch(
                "app.services.analysis_service.generate_services_table",
                return_value=tables["services"],
            ), patch(
                "app.services.analysis_service.generate_audience_profile",
                return_value=tables["audience"],
            ):
                for _ in run_pipeline(payload, validated=(MagicMock(), MagicMock())):
                    pass

        self.assertEqual(len(captured["sites"]), 11)
        self.assertEqual(captured["column_count"], 11)


class LLMStructuredOutputValidationTests(unittest.TestCase):
    def test_retries_once_with_gemini_when_structured_output_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            page = Path(directory) / "clean.html"
            page.write_text("<h1>Example</h1>", encoding="utf-8")
            valid = {
                "rows": [
                    {
                        "values": [
                            f"Service {index}",
                            "Description",
                            "Industry",
                            "Customer",
                            "Benefit",
                            "website",
                        ]
                    }
                    for index in range(1, 6)
                ]
            }
            with patch(
                "app.services.llm_service.ask_gemini",
                side_effect=['{"rows": []}', json.dumps(valid)],
            ) as ask:
                services = generate_services_table(page)

        self.assertEqual(ask.call_count, 2)
        self.assertEqual(len(services["rows"]), 5)

    def test_accepts_valid_array_rows_from_gemini(self):
        rows = parse_value_rows(
            json.dumps(
                {
                    "rows": [
                        ["query", "Google", "Commercial", "high", "buyers"]
                        for _ in range(5)
                    ]
                }
            ),
            count=5,
            column_count=5,
        )

        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0][0], "query")

    def test_rejects_incomplete_comparison_and_value_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            page = Path(directory) / "clean.html"
            page.write_text("<h1>Example</h1>", encoding="utf-8")
            with patch(
                "app.services.llm_service.ask_gemini",
                return_value='{"rows": []}',
            ):
                with self.assertRaises(LLMOutputValidationError):
                    generate_comparison_table(
                        [{"url": "https://example.com", "clean_html": str(page)}]
                    )

        with self.assertRaises(LLMOutputValidationError):
            parse_value_rows('{"rows": []}', count=5, column_count=5)

        with self.assertRaises(LLMOutputValidationError):
            parse_value_rows(
                json.dumps({"rows": [{"values": ["query", "Google", "—", "high", "buyers"]}] * 5}),
                count=5,
                column_count=5,
            )

    def test_rejects_missing_seo_and_audience_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            page = Path(directory) / "clean.html"
            page.write_text("<h1>Example</h1>", encoding="utf-8")

            with patch(
                "app.services.llm_service.ask_gemini",
                return_value='{"groups": []}',
            ):
                with self.assertRaises(LLMOutputValidationError):
                    generate_seo_suggestions(page)

            with patch(
                "app.services.llm_service.ask_gemini",
                return_value='{"items": []}',
            ):
                with self.assertRaises(LLMOutputValidationError):
                    generate_audience_profile(page)

    def test_pipeline_returns_a_clean_error_for_invalid_llm_structure(self):
        payload = LinkRequest(url="https://example.com")
        with patch(
            "app.services.analysis_service.resolve_model",
            side_effect=LLMOutputValidationError("bad response"),
        ):
            event, data = parse_sse_frame(
                next(run_pipeline(payload, validated=(MagicMock(), MagicMock())))
            )

        self.assertEqual(event, "error")
        self.assertEqual(data["status_code"], 502)
        self.assertIn("incomplete structured data", data["message"])


class ValidatorTests(unittest.TestCase):
    def test_blocks_private_local_and_non_http_urls(self):
        blocked = (
            "http://127.0.0.1",
            "http://localhost",
            "http://192.168.1.10",
            "http://[::1]",
            "ftp://example.com",
            "not-a-url",
            "",
        )
        for value in blocked:
            with self.subTest(value=value):
                self.assertFalse(Validator(value).has_valid_format())

    def test_accepts_public_domain(self):
        with patch(
            "app.utils.validators.socket.getaddrinfo", return_value=PUBLIC_ADDRESS
        ):
            self.assertTrue(Validator("https://example.com").has_valid_format())

    def test_redirect_to_private_host_is_blocked_before_it_is_followed(self):
        redirect = MagicMock(status_code=302)
        redirect.headers = {"Location": "http://127.0.0.1/admin"}

        with patch(
            "app.utils.validators.socket.getaddrinfo", return_value=PUBLIC_ADDRESS
        ), patch(
            "app.utils.validators.requests.get", return_value=redirect
        ) as get:
            with self.assertRaises(BlockedURLError):
                request_public_url("https://example.com", timeout=0.1)

        get.assert_called_once()


class CompetitorFilterTests(unittest.TestCase):
    def test_serper_failure_stops_analysis_instead_of_falling_back(self):
        with patch(
            "app.services.competitor_service.get_env", return_value="serper-key"
        ), patch(
            "app.services.competitor_service.requests.post",
            side_effect=requests.RequestException("offline"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Serper search failed"):
                search_queries(["example competitor search"])

    def test_social_video_and_same_site_results_are_excluded(self):
        results = [
            {
                "organic": [
                    {"link": "https://www.youtube.com/watch?v=python"},
                    {"link": "https://www.python.org/about/"},
                    {"link": "https://docs.python.org/3/"},
                    {"link": "https://rival.example/"},
                    {"link": "https://another-rival.example/"},
                ]
            }
        ]

        with patch.object(Validator, "has_valid_format", return_value=True):
            urls = collect_competitor_urls(results, main_url="https://www.python.org/")

        self.assertEqual(
            urls,
            ["https://rival.example/", "https://another-rival.example/"],
        )


class StructuredResponseTests(unittest.TestCase):
    def setUp(self):
        self.tables = sample_tables()
        self.homepage = {
            "final_url": "https://brand.example",
            "title": "Brand",
            "content": {"description": "A sample brand."},
        }

    def build(self):
        return build_structured_response(
            self.homepage,
            self.tables["comparison"],
            self.tables["search"],
            self.tables["seo"],
            self.tables["services"],
            self.tables["audience"],
        )

    def test_has_three_competitors_five_queries_and_five_services(self):
        analysis = self.build()["analysis"]
        self.assertEqual(len(analysis["competitor_analysis"]["competitors"]), 3)
        self.assertEqual(len(analysis["target_search_queries"]), 5)
        self.assertEqual(len(analysis["top_services"]), 5)

    def test_inferred_service_is_clearly_labelled(self):
        services = self.build()["analysis"]["top_services"]
        inferred = [service for service in services if service["source"] == "inferred"]
        self.assertEqual(len(inferred), 1)
        self.assertIn("AI-generated suggestion", inferred[0]["name"])

    def test_search_platforms_are_from_the_allowed_list(self):
        allowed = {"Google", "Facebook", "Instagram", "LinkedIn", "X"}
        queries = self.build()["analysis"]["target_search_queries"]
        for query in queries:
            with self.subTest(query=query["query"]):
                self.assertTrue(set(query["platforms"]).issubset(allowed))

    def test_audience_has_all_nine_requested_labels(self):
        expected = {
            "primary_audience",
            "secondary_audience",
            "industries",
            "business_types",
            "customer_profiles",
            "location_focus",
            "customer_needs",
            "customer_interests",
            "recommended_segments",
        }
        audience = self.build()["analysis"]["target_audience"]
        self.assertTrue(expected.issubset(audience))

    def test_competitor_analysis_declares_its_sources(self):
        competitor_analysis = self.build()["analysis"]["competitor_analysis"]
        self.assertIn("Serper", competitor_analysis["source_note"])
        for competitor in competitor_analysis["competitors"]:
            self.assertTrue(competitor["research_backed"])


if __name__ == "__main__":
    unittest.main()
