"""Test-section slides: extract structured test metadata and upload to MySQL.

A slide whose (propagated) `section` names a "test" bucket — e.g. "Live tests",
"Flagship tests", "Upcoming tests", "Blocked tests" — carries one experiment's
spec. Instead of leaving that spec buried in free-form `detail` blocks, we
ask the LLM to project it onto the fixed `cro.historical_test` table
schema, INSERT the resulting row, and replace the slide's `detail` in
slides.jsonl with a single-block marker pointing at the MySQL table so the
downstream RAG pipeline doesn't re-embed the same content in two places.

Table schema (column order matches the target table):

  source           : slide_id + section        (primary key)
  target_activity  : always None for now — reserved for later hand-tagging
  test_name        : sub_section
  concept          : one of CONCEPTS
  component        : list of APPROVED_COMPONENTS (JSON-encoded in MySQL)
  product          : list of APPROVED_PRODUCTS (JSON-encoded in MySQL)
  hypothesis       : slide body under 'Hypothesis' subheader
  kpi              : list of APPROVED_KPIS (JSON-encoded in MySQL)
  target_audience  : free text
  notes            : caveats / exclusions / watch-outs
  image_path       : list of saved right-side image paths (JSON-encoded in MySQL)
  importDate       : ISO date of the extraction run
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoids an import at runtime just for type hints
    from openai import OpenAI

TEST_TABLE_NAME = "cro.historical_test"

TEST_COLUMNS: tuple[str, ...] = (
    "source",
    "target_activity",
    "test_name",
    "concept",
    "component",
    "product",
    "hypothesis",
    "kpi",
    "target_audience",
    "notes",
    "image_path",
    "importDate",
)

# Columns whose Python value is a list — they're JSON-encoded on the way to MySQL.
_LIST_COLUMNS: frozenset[str] = frozenset({"component", "product", "kpi", "image_path"})

APPROVED_KPIS: tuple[str, ...] = (
    "CVR",
    "AOV",
    "Engagement Rate",
    "Add to Cart Rate",
    "Revenue per Visitor",
)

APPROVED_COMPONENTS: tuple[str, ...] = (
    "Buy Page",
    "Handraisers",
    "Home Page",
    "Home PCD",
    "Landing Page",
    "Multiple",
    "Offer Page",
    "PCD",
    "PD-MMP",
    "PDP",
    "PF",
    "PFP",
)

APPROVED_PRODUCTS: tuple[str, ...] = (
    "TV",
    "DA",
    "Tablet",
    "Paradigm",
    "Flip7/Fold7",
    "B7Q7",
    "Total",
)

CONCEPTS: tuple[str, ...] = (
    "Abandoned Cart & Journey Recovery",
    "Above-the-Fold & Hero Optimisation",
    "App Promotion",
    "Bundles & Add-ons Optimisation",
    "Checkout & Payment Enhancement",
    "Configurator & Product Enhancement",
    "Content & Visibility Enhancement",
    "Countdowns & Urgency",
    "Birthday Campaign",
    "Device & Model Personalisation",
    "EPP Experience & Messaging",
    "Finance, Pricing & Contracts",
    "Gifts, Savings & Vouchers",
    "Gallery, Video & Visual Content Enhancement",
    "Launch & Pre-Order Optimisation",
    "Reward, Loyalty & Incentive Signposting",
    "User Journey & Navigation Adjustments",
    "Promotion Optimisation",
    "VIP Personalisation",
    "Personalisation & Targeting",
    "Personalised Content & Messaging",
    "Personalised Recommendations & Affinity",
    "Personalised Search",
    "PDP Optimisation",
    "Product Discovery & Selection Guidance",
    "Promotional Banners & Messaging",
    "Promotional Placement & Visibility",
    "Purchase Funnel Optimisation",
    "Recently Viewed LP",
    "Returning User Enhancement",
    "Security & Privacy Enhancement",
    "Samsung Care+ Protection",
    "OOS Recommendation",
    "Smart Switch",
    "Social Proof & User Reviews",
    "Trade-In Optimisation",
    "Upsell & Cross-Sell",
    "SMB User Login & Registration",
    "UX/UI Enhancement",
    "KV Personalisation",
    "Paradigm EUG",
    "Post-Purchase Experience",
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


def _coerce_concept(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip()
    return v if v in CONCEPTS else None


def _coerce_from_list(value: object, approved: tuple[str, ...]) -> list[str]:
    """Coerce an LLM value into a de-duplicated list restricted to `approved`.

    Accepts either a single string or a list of strings — LLMs sometimes
    emit a scalar when only one value applies even though the schema asks
    for a list. Anything not exactly in `approved` is silently dropped.
    """
    approved_set = set(approved)
    if value is None:
        return []
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, list):
        raw = value
    else:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for v in raw:
        if not isinstance(v, str):
            continue
        s = v.strip()
        if s in approved_set and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _coerce_str_or_none(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        return v or None
    return str(value).strip() or None


def build_test_row(
    record: dict,
    extracted: dict,
    import_date: str,
    image_paths: list[str] | None = None,
) -> dict:
    """Assemble a fixed-schema row from a slide record + LLM extraction.

    Approved-list validation happens HERE — the LLM is instructed to pick
    from the lists, but any drift is coerced away rather than shipped to
    MySQL. `image_paths` is the list of right-side image paths the caller
    already saved to disk for this test slide (may be empty).
    """
    slide_id = str(record.get("slide_id", "") or "").strip()
    section = str(record.get("section", "") or "").strip()
    sub_section = str(record.get("sub_section", "") or "").strip()

    # kpi is now one combined list — prefer the LLM's `kpi` field, but fall
    # back to the legacy primary_kpi/secondary_kpis shape if the model still
    # emits it, so a stale prompt won't silently drop KPI values.
    kpi_raw = extracted.get("kpi")
    if kpi_raw is None:
        kpi_raw = []
        primary = extracted.get("primary_kpi")
        if primary is not None:
            kpi_raw.append(primary)
        for k in extracted.get("secondary_kpis") or []:
            kpi_raw.append(k)
    kpi = _coerce_from_list(kpi_raw, APPROVED_KPIS)

    component = _coerce_from_list(extracted.get("component"), APPROVED_COMPONENTS)
    product = _coerce_from_list(extracted.get("product"), APPROVED_PRODUCTS)

    return {
        "source": f"{slide_id}_{section}" if section else slide_id,
        "target_activity": None,
        "test_name": sub_section or None,
        "concept": _coerce_concept(extracted.get("concept") or extracted.get("test_group")),
        "component": component,
        "product": product,
        "hypothesis": _coerce_str_or_none(extracted.get("hypothesis")),
        "kpi": kpi,
        "target_audience": _coerce_str_or_none(extracted.get("target_audience")),
        "notes": _coerce_str_or_none(extracted.get("notes")),
        "image_path": list(image_paths or []),
        "importDate": import_date,
    }


def extract_test_row(
    client: "OpenAI",
    model: str,
    record: dict,
    import_date: str,
    image_paths: list[str] | None = None,
) -> dict:
    """LLM-extract a test slide's fields and return a validated row dict."""
    from .llm import extract_test_metadata

    payload = {
        "section": record.get("section", ""),
        "sub_section": record.get("sub_section", ""),
        "content": _flatten_record(record),
        "notes": _extract_slide_note(record),
    }
    extracted = extract_test_metadata(
        client,
        model,
        payload,
        concepts=list(CONCEPTS),
        components=list(APPROVED_COMPONENTS),
        products=list(APPROVED_PRODUCTS),
    )
    return build_test_row(record, extracted, import_date, image_paths=image_paths)


def make_detail_marker(row: dict) -> list[dict]:
    """Return the placeholder `detail` block that replaces a test slide's body.

    Anyone reading slides.jsonl learns the row is authoritative in MySQL and
    is handed the source key to look it up.
    """
    return [
        {
            "body": (
                f"Test metadata for this slide is stored in MySQL table "
                f"`{TEST_TABLE_NAME}` under source={row['source']!r}."
            )
        }
    ]


def upload_rows_to_mysql(rows: list[dict], settings) -> None:
    """Upsert every row into cro.historical_test keyed by source.

    Uses INSERT ... ON DUPLICATE KEY UPDATE so a re-extract of the same
    file overwrites the earlier row for the same source instead of
    inserting a duplicate. This requires `source` to be the PRIMARY
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
            f"`{c}`=VALUES(`{c}`)" for c in TEST_COLUMNS if c != "source"
        )
        sql = (
            f"INSERT INTO {TEST_TABLE_NAME} ({cols}) VALUES ({placeholders}) "
            f"ON DUPLICATE KEY UPDATE {updates}"
        )
        values = [
            tuple(
                json.dumps(r[c], ensure_ascii=False) if c in _LIST_COLUMNS else r[c]
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
    """Timestamp of the extraction run — 'YYYY-MM-DD HH:MM:SS' in local time.

    MySQL DATETIME accepts this format directly, so a re-extract's
    importDate carries the actual wall-clock time of the run rather than
    midnight-of-that-day.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
