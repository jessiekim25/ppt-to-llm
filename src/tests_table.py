"""Test-section slides: extract structured test metadata and upload to MySQL.

A slide whose (propagated) `section` names a "test" bucket — e.g. "Live tests",
"Flagship tests", "Upcoming tests", "Blocked tests" — carries one experiment's
spec. Instead of leaving that spec buried in free-form `detail` blocks, we
ask the LLM to project it onto the fixed `llm_monitor.historical_tests` table
schema, INSERT the resulting row, and replace the slide's `detail` in
slides.jsonl with a single-block marker pointing at the MySQL table so the
downstream RAG pipeline doesn't re-embed the same content in two places.

Table schema (column order matches the target table):

  issueKey         : slide_id + section        (primary key)
  test_name        : sub_section
  test_group       : one of TEST_GROUPS
  hypothesis       : slide body under 'Hypothesis' subheader
  primary_kpi      : one of APPROVED_KPIS
  secondary_kpis   : list of APPROVED_KPIS (JSON-encoded in MySQL)
  target_audience  : free text
  notes            : caveats / exclusions / watch-outs
  importDate       : ISO date of the extraction run
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoids an import at runtime just for type hints
    from openai import OpenAI

TEST_TABLE_NAME = "cro.historical_test"

TEST_COLUMNS: tuple[str, ...] = (
    "issueKey",
    "test_name",
    "test_group",
    "hypothesis",
    "primary_kpi",
    "secondary_kpis",
    "target_audience",
    "notes",
    "importDate",
)

APPROVED_KPIS: tuple[str, ...] = (
    "CVR",
    "AOV",
    "Engagement Rate",
    "Add to Cart Rate",
    "Revenue per Visitor",
)

TEST_GROUPS: tuple[str, ...] = (
    "A/B Testing and Optimization",
    "Abandoned Cart",
    "Abandoned Cart Optimization",
    "Abandoned Cart Recovery",
    "Abandoned Carts & Shopping Journey",
    "Bundle & Add-Ons Optimization",
    "Campaign and Promotion Messaging",
    "Cart Abandonment and Recovery",
    "Checkout and Payment Enhancements",
    "Checkout Process Improvements",
    "Configurator and Customization Improvements",
    "Content and Visual Elements Enhancement",
    "Content Enhancement and Visibility Improvements",
    "Content Personalization",
    "Countdowns and Urgency Tactics",
    "Device and Model Personalization",
    "Finance and Pricing Options",
    "Gallery Content Enhancement",
    "Gift and Savings Optimizations",
    "Incentives and Reward Signposting",
    "Launch & Pre-Order Enhancements",
    "Messaging & Signposting",
    "Personalisation and Personalised Recommendations",
    "Personalisation and Targeting",
    "Personalization",
    "Personalization & Customization",
    "Personalization & Recommendations",
    "Personalization & User Affinity",
    "Personalization and Targeted Messaging",
    "Pre-order and Promotional Strategies",
    "Price Optimization & Finance",
    "Pricing & Finance",
    "Pricing and Finance Options",
    "Pricing and Financing Options",
    "Privacy & Security Enhancements",
    "Product Configuration and Content Enhancements",
    "Product Detail Page Optimizations",
    "Product Discovery & Selection Assistance",
    "Product Information and Configurator Enhancements",
    "Product Recommendations and Add-ons",
    "Product Selection Guidance",
    "Promo & Campaign Visibility",
    "Promo Banners & Discounts",
    "Promotion & Visibility Optimization",
    "Promotional Banners & Messaging",
    "Promotional Messaging and Banners",
    "Promotional Placements & Visibility",
    "Purchase Funnel Optimization",
    "Rewards & Loyalty Programs",
    "Rewards & Loyalty Signposting",
    "Rewards & Signposting",
    "Rewards and Offer Signposting",
    "Rewards and Signposting Enhancements",
    "Rewards Signposting",
    "Smart Switch & Device Detection Enhancements",
    "Social Proof & User Reviews",
    "Upsell and Cross-sell",
    "Upselling & Cross-selling",
    "Upselling and Cross-selling Strategies",
    "Upselling Strategies",
    "User Experience Enhancements",
    "User Journey and Engagement Tools",
    "User Journey and Navigation Adjustments",
    "UX/UI Improvements",
    "Video and Visual Content Enhancements",
    "Visibility & Messaging Impact",
    "Visual Enhancements",
)

_TEST_SECTION_RE = re.compile(r"\btests?\b", re.IGNORECASE)


def is_test_section(section: str) -> bool:
    """True when a slide's propagated section names a test bucket.

    Matches any section whose name contains the whole word "test" or "tests"
    — covers "Live tests", "Flagship tests", "Upcoming tests", "Blocked
    tests", "A/B tests", etc. A section like "Contest" or "Latest news"
    does NOT match because \\b enforces a word boundary.
    """
    return bool(_TEST_SECTION_RE.search(section or ""))


def _walk_blocks_flatten(blocks: list, out: list[str]) -> None:
    for b in blocks or []:
        if not isinstance(b, dict):
            continue
        sh = (b.get("subheader") or "").strip()
        body = (b.get("body") or "").strip()
        table = (b.get("table") or "").strip()
        if sh:
            out.append(f"[{sh}]")
        if body:
            out.append(body)
        if table:
            out.append(table)
        _walk_blocks_flatten(b.get("children") or [], out)


def _flatten_record(record: dict) -> str:
    """Flatten a slide's detail blocks into a plain-text digest for the LLM."""
    parts: list[str] = []
    _walk_blocks_flatten(record.get("detail") or [], parts)
    return "\n".join(parts)


def _extract_slide_note(record: dict) -> str:
    """Pull the 'slide note' block (appended by _build_detail_from_pptx_groups)."""
    for b in record.get("detail") or []:
        if not isinstance(b, dict):
            continue
        if (b.get("subheader") or "").strip().lower() == "slide note":
            return (b.get("body") or "").strip()
    return ""


def _coerce_kpi(value: object) -> str | None:
    """Return `value` when it matches an APPROVED_KPI verbatim, else None."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    return v if v in APPROVED_KPIS else None


def _coerce_group(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip()
    return v if v in TEST_GROUPS else None


def _coerce_str_or_none(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        return v or None
    return str(value).strip() or None


def build_test_row(record: dict, extracted: dict, import_date: str) -> dict:
    """Assemble a fixed-schema row from a slide record + LLM extraction.

    Approved-list validation happens HERE — the LLM is instructed to pick
    from the lists, but any drift is coerced to None rather than shipped
    to MySQL.
    """
    slide_id = str(record.get("slide_id", "") or "").strip()
    section = str(record.get("section", "") or "").strip()
    sub_section = str(record.get("sub_section", "") or "").strip()

    primary = _coerce_kpi(extracted.get("primary_kpi"))
    seen: set[str] = set()
    secondary: list[str] = []
    for k in extracted.get("secondary_kpis") or []:
        norm = _coerce_kpi(k)
        if not norm or norm == primary or norm in seen:
            continue
        seen.add(norm)
        secondary.append(norm)

    return {
        "issueKey": f"{slide_id}_{section}" if section else slide_id,
        "test_name": sub_section or None,
        "test_group": _coerce_group(extracted.get("test_group")),
        "hypothesis": _coerce_str_or_none(extracted.get("hypothesis")),
        "primary_kpi": primary,
        "secondary_kpis": secondary,
        "target_audience": _coerce_str_or_none(extracted.get("target_audience")),
        "notes": _coerce_str_or_none(extracted.get("notes")),
        "importDate": import_date,
    }


def extract_test_row(client: "OpenAI", model: str, record: dict, import_date: str) -> dict:
    """LLM-extract a test slide's fields and return a validated row dict."""
    from .llm import extract_test_metadata

    payload = {
        "section": record.get("section", ""),
        "sub_section": record.get("sub_section", ""),
        "content": _flatten_record(record),
        "notes": _extract_slide_note(record),
    }
    extracted = extract_test_metadata(client, model, payload, list(TEST_GROUPS))
    return build_test_row(record, extracted, import_date)


def make_detail_marker(row: dict) -> list[dict]:
    """Return the placeholder `detail` block that replaces a test slide's body.

    Anyone reading slides.jsonl learns the row is authoritative in MySQL and
    is handed the issueKey to look it up.
    """
    return [
        {
            "body": (
                f"Test metadata for this slide is stored in MySQL table "
                f"`{TEST_TABLE_NAME}` under issueKey={row['issueKey']!r}."
            )
        }
    ]


def upload_rows_to_mysql(rows: list[dict], settings) -> None:
    """Upsert every row into cro.historical_test keyed by issueKey.

    Uses INSERT ... ON DUPLICATE KEY UPDATE so a re-extract of the same
    file overwrites the earlier row for the same issueKey instead of
    inserting a duplicate. This requires `issueKey` to be the PRIMARY
    KEY or carry a UNIQUE index; without one, the ON DUPLICATE branch
    never fires and rows accumulate.

    No-ops (with a printed warning) when pymysql isn't installed or the
    MySQL credentials aren't configured — the extractor still writes
    slides.jsonl and tests.jsonl to disk, so a missing sink degrades
    gracefully.
    """
    if not rows:
        return
    try:
        import pymysql  # type: ignore
    except ImportError:
        print("[tests] pymysql not installed; skipping MySQL upload.")
        return
    if not (settings.mysql_host and settings.mysql_user):
        missing = [
            k for k, v in (("RDS_HOSTNAME", settings.mysql_host), ("RDS_USERNAME_TESTDB", settings.mysql_user))
            if not v
        ]
        print(
            f"[tests] MySQL credentials not configured (missing {', '.join(missing)} "
            f"in AWS Secrets Manager secret 'MySQL' — override the secret name "
            f"with MYSQL_SECRET_NAME). Skipping upload."
        )
        return

    conn = pymysql.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_database,
        autocommit=False,
        charset="utf8mb4",
    )
    try:
        cols = ", ".join(f"`{c}`" for c in TEST_COLUMNS)
        placeholders = ", ".join(["%s"] * len(TEST_COLUMNS))
        # Every non-key column is refreshed from the incoming row on conflict.
        updates = ", ".join(
            f"`{c}`=VALUES(`{c}`)" for c in TEST_COLUMNS if c != "issueKey"
        )
        sql = (
            f"INSERT INTO {TEST_TABLE_NAME} ({cols}) VALUES ({placeholders}) "
            f"ON DUPLICATE KEY UPDATE {updates}"
        )
        values = [
            tuple(
                json.dumps(r[c], ensure_ascii=False) if c == "secondary_kpis" else r[c]
                for c in TEST_COLUMNS
            )
            for r in rows
        ]
        with conn.cursor() as cur:
            cur.executemany(sql, values)
        conn.commit()
        print(f"[tests] upserted {len(rows)} row(s) into {TEST_TABLE_NAME}")
    finally:
        conn.close()


def write_tests_jsonl(rows: list[dict], out_path) -> None:
    """Persist test rows locally as JSONL — the same shape as MySQL columns.

    Local copy so a run whose MySQL upload was skipped still leaves the
    structured extraction on disk for review or a later push.
    """
    if not rows:
        return
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            ordered = {c: r.get(c) for c in TEST_COLUMNS}
            f.write(json.dumps(ordered, ensure_ascii=False))
            f.write("\n")
    print(f"[tests] wrote {len(rows)} test row(s) to {out_path}")


def today_iso() -> str:
    return date.today().isoformat()
