"""Text-only LLM classifier for the deterministic-first pipeline.

Consumes what pdfplumber and Camelot already extracted (text blocks + candidate tables)
and returns the same row schema `extract.build_row` expects — product, codename, section,
sub_section, model, detail, tables, subheaders — so the deterministic stack + this call
can slot into the existing DB writer without a vision pass.
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

SYSTEM_PROMPT = """You classify content that was already extracted from a Samsung campaign visual identity slide by deterministic tools.

You will NOT see the slide image. You receive JSON with:
  - text_blocks: ordered list of text blocks (each has `text`, `x0`, `top`, `x1`, `bottom`; positions are PDF points, top-left origin).
  - camelot_tables.lattice / camelot_tables.stream: candidate tables from Camelot's two flavors. Pick whichever looks cleaner per table; drop obvious duplicates. Ignore font metadata — you don't get any.

Return a JSON object with these fields, using ONLY text that appears in the input blocks (never invent). Assign role from position and content:

  - product: general phone series if visible on the slide (e.g. "Galaxy S"). "" if generic/not visible.
  - codename: campaign code name if literally on the slide. "" if not visible.
  - section: top-left section label near the top of the slide (e.g. "01 Brand Basics"). "" if none.
  - sub_section: the slide's main title/heading.
  - model: specific phone model shown (e.g. "Galaxy S26 Ultra"). "" if not on slide.
  - detail: slide-level body text not tied to any subheader — introductory paragraphs, footnotes, do's/don'ts. Preserve specifics (hex codes, pixel values, ratios). "" if none. Numbered legends ("N: <label> — <description>", one per line) go here in full when present.
  - tables: consolidated list of text-only tables — schema { "title": "", "columns": [...], "rows": [[...], ...] }. Reconcile lattice vs stream. Return [] if none.
  - subheaders: array describing every heading + descriptive-text pair, other than the main slide title. Each entry:
      { "title": "<heading text only>",
        "detail": "<body text under this heading, verbatim, excluding child headings>",
        "tables": [ ... same schema as top-level tables, if any belong to this subheader ],
        "children": [ ... nested subheaders, same schema ] }
    Strict rules:
      (a) every distinct heading gets its OWN entry — never merge multiple headings into one entry's `detail` as a bulleted list.
      (b) `title` is heading text only — no leading dashes/bullets, no description.
      (c) nested sub-blocks (e.g. "How to build layout:" under "4:1 proportion") go in the parent's `children`, not flattened at the top level.
      (d) don't invent headings; don't put the main slide title in subheaders.

Return ONLY the JSON object. No prose, no code fences."""


def classify_from_deterministic(
    client: OpenAI,
    model: str,
    plumber: dict[str, Any],
    camelot_result: dict[str, Any],
) -> dict[str, Any]:
    """Send deterministic output as text to a cheap LLM; get back a row-shaped dict."""
    payload = {
        "text_blocks": plumber.get("blocks", []),
        "camelot_tables": {
            "lattice": camelot_result.get("lattice") if isinstance(camelot_result.get("lattice"), list) else [],
            "stream": camelot_result.get("stream") if isinstance(camelot_result.get("stream"), list) else [],
        },
    }
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)
