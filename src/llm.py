import json

from openai import OpenAI

SYSTEM_PROMPT = """You extract structured data from a single slide of a Samsung campaign visual identity guideline deck.

The slide's text and figures have already been extracted from the PDF geometrically. You will receive a JSON payload describing the slide layout — no image.

INPUT SHAPE:
{
  "page_size": [width, height],                          // in PDF points; ignore units, coordinates are already normalized
  "text_lines": [
    {"bbox": [x0, y0, x1, y1], "text": "...", "size": 12.0, "bold": false}
  ],
  "figures": [
    {"idx": 1, "bbox": [x0, y0, x1, y1], "label": "..."} // pre-detected figure regions; their nearest short caption is in `label`
  ]
}
All bboxes are TOP-LEFT origin, fractions of page (0-1). text_lines arrive roughly in top-to-bottom order but you must decide layout structure from bbox geometry, not input order.

Use text_lines' geometry and typography to reconstruct layout:
- Larger `size` or `bold: true` marks a heading (section label, slide title, subheader).
- Text lines whose bboxes share the same x0/x1 across multiple rows are a single column.
- Tables have already been extracted geometrically. You do NOT emit tables in your output — there is no `tables` field. If you see text lines that visually resemble a table (grid-aligned rows/columns of image labels or short captions), treat each column as a separate subheader.
- `figures[i].bbox` marks where an image sits — use it only to understand layout. A text line's position near/inside/across a figure does NOT make it a caption to drop.

COMPLETENESS RULE — highest priority:
Every text line in the payload MUST appear somewhere in your output — as one of the slide-level string fields, in slide-level `detail`, inside a subheader's `title`/`detail`, as a table cell, or nested in `children`. NEVER drop a text line as "noise", "figure caption", "already visible on the slide", or "duplicate". If you're unsure where a line belongs, put it in slide-level `detail` rather than dropping it. The only exemptions are purely decorative fragments with no words (stray dashes, arrows) and the PAGE CHROME described below.

PAGE CHROME — always drop, never emit anywhere in the output:
These are templated running headers/footers repeated on every slide and are not slide content.

TOP EDGE:
- Corner stamps in the top-RIGHT (e.g. "Confidential", "Draft", "Internal Only", "Do Not Distribute", "Proprietary"). NOTE: the top-LEFT section indicator (e.g. "01 Brand Basics", "Guidance usage") is NOT chrome — it belongs in the `section` field.

BOTTOM EDGE:
- Navigation strips listing section names separated by " | " (e.g. "TOC | Strategy | Campaign assets | Guidance usage | Resources").
- Copyright / legal footers (e.g. "Copyright © 2013-2025 Samsung Electronics Co., Ltd. All Rights Reserved.").
- Repeated deck-title footers naming the whole deck (e.g. "Galaxy S26 Campaign visual guidelines").
- Page-number-only lines (a small isolated integer, or "N of M").

A text line is page chrome when it sits within ~5% of the top or bottom page edge AND matches one of the patterns above. Do NOT drop legitimate slide content that just happens to be near an edge (the slide's sub_section title near the top is NOT chrome; small footnotes/disclaimers that are unique to the slide are NOT chrome).

MULTI-LINE TITLES: if two or more consecutive text lines near the top of the slide share the same (or very close) `size` and sit at consecutive y-positions with matching x0, they are ONE title that wrapped to multiple lines. Concatenate them with a single space and put the joined string in `sub_section`. Never emit only the first line.

COLUMN STRUCTURE — read this before assigning any text to a subheader or detail:
1. Scan every bold/large heading. If two or more headings share a similar y0 (within ~5% of page height) at clearly different x0 positions, the slide has PARALLEL COLUMNS at that y-band. Each such heading is a separate column-anchor subheader.
2. A COLUMN CAN SPAN A FIGURE. A column's heading may sit at the top of a section and its descriptive body may sit at the bottom of the same section, with a figure (or blank space) between them. Group by x-range: any text in the same x-band as a column heading — above OR below any intermediate figure — belongs to that column's subheader (as `detail` or as a nested child). Do NOT let an intervening figure orphan the body text.
3. For each column-anchor subheader, its column extends across the full height of that section. Every text line whose x-center falls within (or near) that heading's x-range belongs to that column.
4. Column body content NEVER lands in the slide-level `detail`. If a line clearly sits within a column's x-band, it belongs to that column's subheader.
5. REPEATED CHILD HEADINGS: when the SAME label (e.g. "How to build layout:") appears once under each column, EACH occurrence is a distinct child of the column subheader directly above it — do NOT merge them into one entry, and do NOT hoist them to the slide level.
6. OUTPUT ORDER FOR COLUMNS: within a parallel-column band, emit the LEFT column's subheader FULLY (title + detail + all children recursively) before starting the RIGHT column's subheader. Do NOT interleave content from parallel columns.

EMPTY SUBHEADER RULE: this should almost never fire under the COMPLETENESS RULE. But if a candidate heading truly has no body and no children after you've tried COLUMN STRUCTURE step 2 (checking above/below figures in the same x-band), it may be a decorative label — put it in slide-level `detail` verbatim rather than emitting a subheader with no content.

Return a JSON object with these fields.

Slide-level string fields ("" if not visible):
- product: general phone series (e.g. "Galaxy S"). "" if generic.
- section: the top-left corner header — the deck's section/chapter indicator (e.g. "01 Brand Basics", "Campaign Assets", "Guidance usage", "Resources"). This is nearly ALWAYS present as small text near coordinates (x0 ≤ 0.15, y0 ≤ 0.06). Read that text verbatim and put it here — do NOT drop it as chrome. Only return "" if the slide genuinely has no top-left corner text.
- sub_section: the slide's main title/heading (typically the largest text near the top of the slide, not counting the section label).
- model: specific phone model shown (e.g. "Galaxy S26 Ultra"). "" if none.

Content fields:
- detail: general body text on the slide that is NOT tied to any subheader (see below) — introductory paragraphs, footnotes, do's & don'ts.
  IMPORTANT: numbered legends must be captured here in full. A numbered legend is a vertical or side-by-side list of items where each item begins with a small isolated single digit (1, 2, 3, ...) followed by a short label and (optionally) a description. Capture every legend item verbatim, one per line, formatted as "N: <label> — <description>" (drop "—" if there's no description). A short isolated digit text line adjacent to a descriptive text line is almost certainly a numbered legend entry, even though the circle around the digit doesn't appear in the payload.
  Preserve specifics (hex codes, pixel values, ratios). "" if there is truly no slide-level body text at all.

  Slide-wide text that must always land in this slide-level `detail` field, regardless of layout:
    * running text above a horizontal rule that introduces the slide (e.g. "Type family and weight distribution.");
    * footnotes, disclaimers, or fine print at the very bottom of the slide (small `size`, near the bottom of the page).

- subheaders: array describing every distinct heading + descriptive-text pair on the slide, other than the main slide title itself. A subheader is a SHORT LABEL that introduces a block of descriptive body text OR a nested sub-block. Includes:
    * sub-titles that horizontally divide the slide into sections;
    * bold column headings at the top of side-by-side text blocks in a multi-column layout — these ARE column anchors (see COLUMN STRUCTURE);
    * labels marking each cell of a grid layout, with a paragraph next to or below;
    * captions under figures that name each panel type — text sitting BETWEEN two figures with descriptive text below IS a subheader.
  Capture ALL such headings in reading order (top-to-bottom then left-to-right, respecting COLUMN STRUCTURE above). For each one, put the full descriptive body text next to/below that heading into the subheader's `detail` field, verbatim and complete.

  WHAT COUNTS AS A SUBHEADER — all three must hold:
    (α) At most 10 words, and it does NOT end with a period. A phrase that ends with "." is a sentence, not a subheader — it belongs in `detail`. (A trailing ":" is fine and common on subheaders like "How to build layout:".)
    (β) A clear typographic distinction from the text that follows: strictly larger `size`, OR `bold: true` when the body below is not bold.
    (γ) At least one text line of descriptive body directly under/beside it that would become its `detail` or a nested `children` entry. A short bold phrase with nothing beneath it is decorative — put it in slide-level `detail` verbatim, do not emit as a subheader.
  If a candidate fails ANY of (α)/(β)/(γ), it is body text, not a subheader — merge it back into the surrounding paragraph in `detail` (or its parent subheader's `detail`).

  STRICT RULES:
    (a) EVERY qualifying heading must be its OWN entry with its heading text in the `title` field. Do NOT collapse multiple headings into one subheader's `detail` as a bulleted list.
    (b) `title` contains ONLY the heading text — never the description, never a leading dash or bullet.
    (c) `detail` contains the full descriptive body text for THAT subheader only — never other subheaders' titles as bullets, never content that belongs to a different column.
    (d) Do not invent headings, and do not use the main slide title as a subheader.
    (e) NESTING: if a subheader's visual area contains another labeled sub-block below it (e.g. "4:1 proportion" column contains a "How to build layout:" heading with a numbered list beneath), put that inner sub-block in the parent's `children` array — do NOT flatten it, do NOT stuff the child's content into the parent's `detail`, and do NOT lift the child to the slide-level detail. A child subheader has the same schema as its parent and can itself have `children`.

  Each entry:
  {
    "title": "<the heading text exactly as printed, no trailing period, ≤10 words>",
    "detail": "<all descriptive body text under/next to this heading (excluding any child subheaders' content), verbatim; \"\" if none>",
    "children": [ <nested subheader entries; [] if none> ]
  }

Return ONLY the JSON object. No prose, no code fences."""


def _round_bbox(b):
    return [round(float(x), 4) for x in b]


def build_payload(layout, page_num: int) -> dict:
    """Serialize a PageLayout into the compact JSON payload the LLM prompt expects.

    text_lines are sorted top-to-bottom, then left-to-right, with y-values banded
    to 1% of the page so adjacent lines in the same paragraph stay clustered.
    """
    def sort_key(tl):
        y0, x0 = tl.bbox_pct[1], tl.bbox_pct[0]
        return (round(y0, 2), x0)

    text_lines = []
    for tl in sorted(layout.text_lines, key=sort_key):
        entry = {
            "bbox": _round_bbox(tl.bbox_pct),
            "text": tl.text,
        }
        if tl.size is not None:
            entry["size"] = round(float(tl.size), 2)
        if tl.bold:
            entry["bold"] = True
        if tl.group_id is not None:
            entry["group"] = tl.group_id
        text_lines.append(entry)

    figures = [
        {
            "idx": i,
            "bbox": _round_bbox(fig.bbox_pct),
            **({"label": fig.label} if fig.label else {}),
        }
        for i, fig in enumerate(layout.figures, start=1)
    ]

    return {
        "page": page_num,
        "page_size": [round(layout.width, 2), round(layout.height, 2)],
        "text_lines": text_lines,
        "figures": figures,
    }


SYSTEM_PROMPT_PPTX = """You extract structured data from a single slide of a PowerPoint deck.

The slide's text, tables, and figure regions have already been extracted from the pptx geometrically. You will receive a JSON payload describing the slide layout — no image.

INPUT SHAPE:
{
  "page_size": [width, height],
  "text_lines": [
    {"bbox": [x0, y0, x1, y1], "text": "...", "size": 12.0, "bold": false, "group": 3}
  ],
  "figures": [
    {"idx": 1, "bbox": [x0, y0, x1, y1], "label": "..."}
  ]
}
All bboxes are TOP-LEFT origin, fractions of page (0-1). text_lines arrive roughly in top-to-bottom order but you must decide layout structure from bbox geometry, not input order.

Use text_lines' geometry and typography to reconstruct layout:
- Larger `size` or `bold: true` marks a heading (slide title, subheader).
- Text lines whose bboxes share the same x0/x1 across multiple rows are a single column.
- `group` is a per-text-frame integer. Every text_line that came from the SAME text box in the source deck has the same `group` value. Use it as the strongest grouping signal — lines with the same `group` are guaranteed to be one visual block that must be interpreted together (see SPLITTING WITHIN ONE TEXT BLOCK below). Lines with different `group` values may still be adjacent on the slide, but they came from separate boxes; do not merge them without a geometric reason.
- Tables have already been extracted. Do NOT emit tables in your output — there is no `tables` field. If you see text lines that visually resemble a table (grid-aligned rows/columns of short captions), treat each column as a separate subheader.
- `figures[i].bbox` marks where an image sits — use it only to understand layout.

SPLITTING WITHIN ONE TEXT BLOCK — critical for pptx:
A single text box (all lines with the same `group`) often contains multiple subheader+body pairs stacked vertically. When the styling alternates — bold short line, then non-bold body line(s), then another bold short line, then more body — SPLIT the block into separate subheader entries. Do NOT concatenate the whole block into one flat body paragraph just because the lines came from one box or arrived adjacent in the input.

  Example block (one group, four paragraphs):
    - "KEY INITIATIVES"           (bold, ≤4 words)
    - "- SIA rework"              (regular, bullet)
    - "- Nav simplification"      (regular, bullet)
    - "- Promo journey optimisation" (regular)
  becomes ONE subheader:
    { "title": "KEY INITIATIVES",
      "detail": "- SIA rework\\n- Nav simplification\\n- Promo journey optimisation",
      "children": [] }

  Example nested block (one group, six paragraphs) — a time-frame column with a nested KEY INITIATIVES list:
    - "Week 33 - 35, August"       (large, bold)
    - "KEY INITIATIVES"            (bold, smaller)
    - "- SIA rework"               (regular)
    - "- Nav simplification"       (regular)
    - "- Cart optimisation"        (regular)
    - "- Promo journey optimisation" (regular)
  becomes:
    { "title": "Week 33 - 35, August",
      "detail": "",
      "children": [
        { "title": "KEY INITIATIVES",
          "detail": "- SIA rework\\n- Nav simplification\\n- Cart optimisation\\n- Promo journey optimisation",
          "children": [] }
      ] }

WORKED EXAMPLE — three-column roadmap slide with nested subheaders:
Suppose the slide's main title is "COP - Optimisation Road Map" at the top, and BELOW it there are three parallel columns at similar y0, each in its own text group:
   left column   (group A): "Week 33 - 35, August"    + "KEY INITIATIVES" + bullets
   middle column (group B): "Week 36 - 40, September" + "KEY INITIATIVES" + bullets
   right column  (group C): "Week 40 - 44, October"   + "KEY INITIATIVES" + bullets
Plus a small legend in the bottom-left (BACKLOG / UX/BUILD/QA / LIVE/DONE).

Expected output:
   sub_section = "COP - Optimisation Road Map"
   detail      = "BACKLOG\\nUX/BUILD/QA\\nLIVE/DONE"   (or whatever the legend text is, verbatim)
   subheaders  = [
     { "title": "Week 33 - 35, August",
       "detail": "",
       "children": [ { "title": "KEY INITIATIVES", "detail": "<bullets from left column verbatim>", "children": [] } ] },
     { "title": "Week 36 - 40, September",
       "detail": "",
       "children": [ { "title": "KEY INITIATIVES", "detail": "<bullets from middle column verbatim>", "children": [] } ] },
     { "title": "Week 40 - 44, October",
       "detail": "",
       "children": [ { "title": "KEY INITIATIVES", "detail": "<bullets from right column verbatim>", "children": [] } ] }
   ]
Notice the three "KEY INITIATIVES" occurrences are each their own child under their own time-frame parent — do NOT merge them into one, and do NOT hoist their bullets into the slide-level detail.

COMPLETENESS RULE — highest priority:
Every text line in the payload MUST appear somewhere in your output — as one of the slide-level string fields, in slide-level `detail`, inside a subheader's `title`/`detail`, or nested in `children`. NEVER drop a text line as "noise" or "already visible on the slide". The only exemptions are purely decorative fragments with no words and PAGE CHROME (below).

PAGE CHROME — always drop:
- Slide number stamps (a small isolated integer near the bottom-right corner).
- Repeated brand/deck footers that appear on nearly every slide (e.g. company name logotype in the top-right corner like "SAMSUNG").
- Copyright / legal footers hugging the bottom edge.

SECTION INTRO DETECTION — the primary difference from a guideline deck:
This deck is divided into content sections. Each section starts with a "section intro" slide whose entire purpose is to introduce that section. Use this checklist — a slide IS a section intro when ALL are true:
  1. Very sparse content — usually 1-3 text lines total, at most 4.
  2. No tables, no columns of parallel headings, no bullet lists, no descriptive paragraphs of body text.
  3. One dominant heading text whose `size` is dramatically larger than every other text on the slide — typically ≥2x. This heading is short: 1-4 words is typical (e.g. "Roadmap", "March review", "Live tests", "Next steps", "Q&A"). It sits large in the slide's main visual area (not tucked in a corner).
  4. At most ONE other short line (a subtitle or one-sentence description).
  A regular content slide has multiple text lines of body content, subheaders, tables, or columns — even if it also has a big title. If any of (1)-(4) fail, it is NOT an intro.

When the slide IS a section intro:
  - Set `is_section_intro: true`.
  - Put the dominant heading text (e.g. "Roadmap", "March review", "Live tests") in the `section` field VERBATIM.
  - If a short subtitle exists, put it in `sub_section`.
  - Leave `detail` as "" and `subheaders` as []. Do NOT invent subheaders — the intro slide's content IS the title/subtitle themselves.

When the slide is NOT a section intro (regular content slide):
  - Set `is_section_intro: false`.
  - Set `section` to "" ALWAYS. The section for content slides is filled in by post-processing from the most recent intro slide. Any value you put here is discarded, so setting it wrong wastes the field — leave it empty.
  - Extract `sub_section` (the slide's main title) and everything else normally.

MULTI-LINE TITLES: consecutive text lines near the top with the same (or very close) `size` and matching x0 are ONE title that wrapped. Concatenate with a single space.

COLUMN STRUCTURE — read this before assigning any text to a subheader or detail:
1. Scan every bold/large heading. If two or more headings share a similar y0 (within ~5% of page height) at clearly different x0 positions, the slide has PARALLEL COLUMNS at that y-band. Each such heading is a separate column-anchor subheader.
2. A COLUMN CAN SPAN A FIGURE. Group by x-range: any text in the same x-band as a column heading — above OR below any intermediate figure — belongs to that column's subheader.
3. For each column-anchor subheader, its column extends across the full height of that section. Every text line whose x-center falls within (or near) that heading's x-range belongs to that column.
4. Column body content NEVER lands in the slide-level `detail`.
5. OUTPUT ORDER FOR COLUMNS: emit the LEFT column's subheader FULLY (title + detail + all children recursively) before the RIGHT column's subheader. Do NOT interleave.

Return a JSON object with these fields.

Slide-level string fields ("" if not visible):
- product: general phone series (e.g. "Galaxy S"). "" if generic or not applicable.
- section: ONLY set on section-intro slides (see SECTION INTRO DETECTION). Otherwise "".
- sub_section: the slide's main title/heading — the largest text near the top, not counting any product/brand mark. On a section-intro slide, this is the optional subtitle.
- model: specific phone model shown (e.g. "Galaxy Watch 8"). "" if none.
- is_section_intro: true if this slide's sole purpose is to introduce a new section (see rules above), otherwise false.

Content fields:
- detail: general body text on the slide that is NOT tied to any subheader — introductory paragraphs, bullets, footnotes. Preserve specifics (dates, numbers, links). "" if there is truly no slide-level body text at all. On a section-intro slide this is almost always "".

- subheaders: array describing every distinct heading + body pair on the slide, other than the main slide title itself.
  WHAT COUNTS AS A SUBHEADER — all three must hold:
    (α) At most 10 words, and does NOT end with a period. A trailing ":" is fine.
    (β) A clear typographic distinction from the text that follows. Either of these qualifies:
        - strictly larger `size` than the text that follows, OR
        - `bold: true` while the text that follows is not bold.
      IMPORTANT: bold-vs-non-bold is the STRONGEST signal in this deck — most subheaders and even table column labels are set in bold with the body text below them in regular weight. Whenever you see a short bold line immediately followed by longer non-bold text that reads as its explanation, treat the bold line as a subheader by default. Do NOT swallow it into the body paragraph.
    (γ) At least one text line beneath/beside it that will become its content — EITHER a descriptive body line OR another qualifying subheader that becomes a child. A bold "Week 33 - 35, August" heading whose only content below is another bold "KEY INITIATIVES" subheader (which in turn owns the bullets) still satisfies (γ) because the child block becomes its `children`. Only a short bold phrase with NOTHING beneath it — no body, no child subheader — is decorative; put it in slide-level `detail` verbatim.
  If a candidate fails ANY of (α)/(β)/(γ), it is body text — merge it back into the surrounding paragraph in `detail`. But don't discard weak-typography short lines just because you're unsure — err on the side of promoting a plausibly-bold short line to a subheader when the line after it looks like body copy.

  STRICT RULES:
    (a) EVERY qualifying heading is its OWN entry with its heading text in `title`.
    (b) `title` contains ONLY the heading text.
    (c) `detail` contains the full descriptive body text for THAT subheader only.
    (d) Do not invent headings, and do not use the main slide title as a subheader.
    (e) NESTING: if a subheader's visual area contains another labeled sub-block below it, put it in the parent's `children` array. A child subheader has the same schema.

  Each entry:
  {
    "title": "<heading text, no trailing period, ≤10 words>",
    "detail": "<all descriptive body text under/next to this heading, verbatim; \\"\\" if none>",
    "children": [ <nested subheader entries; [] if none> ]
  }

Return ONLY the JSON object. No prose, no code fences."""


VISION_PROMPT = """You are reading a single pptx slide rendered as an image. The slide has almost no extractable text via the file's structure, so any content you can convey has to come from the image itself.

Return a JSON object with these fields (all strings, empty string if not present):
  {
    "sub_section": "<the slide's main heading / title as printed on the slide>",
    "body": "<every other piece of text visible on the slide, in reading order, joined with newlines>",
    "figure_description": "<one short factual sentence describing what the picture depicts — ONLY when the slide is dominated by a photo/illustration with little or no text; otherwise leave empty>"
  }

Rules:
- Read every piece of text visible in the image. Preserve line breaks between visually separate items in `body` — one item per line.
- `sub_section` is the largest / topmost heading on the slide. If the slide has no obvious title, leave it "".
- `figure_description` is a fallback for photo-only slides so the record isn't empty. If the slide already carries text you captured in `sub_section` or `body`, leave `figure_description` empty.
- Return ONLY the JSON object. No prose, no code fences."""


def _b64_image(image_path) -> str:
    import base64
    from pathlib import Path
    data = Path(image_path).read_bytes()
    return base64.b64encode(data).decode("ascii")


def extract_slide_from_image(client: OpenAI, model: str, image_path) -> dict:
    """Vision fallback for image-only slides: send the rendered slide PNG to
    the LLM and get back sub_section + body text scraped from the image."""
    b64 = _b64_image(image_path)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": VISION_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract sub_section, body, and figure_description from this slide."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


def extract_slide(client: OpenAI, model: str, payload: dict, kind: str = "pdf") -> dict:
    """Call the LLM with the prompt appropriate for the input kind ('pdf' or 'pptx')."""
    system_prompt = SYSTEM_PROMPT_PPTX if kind == "pptx" else SYSTEM_PROMPT
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


TEST_EXTRACTION_PROMPT = """You extract structured test-experiment metadata from a single PowerPoint slide describing one website A/B test.

You will receive a JSON payload with the slide's flattened content and its speaker notes:
{
  "section": "<the section the slide belongs to, e.g. Live tests>",
  "sub_section": "<the slide's main title, i.e. the test name>",
  "content": "<flattened detail blocks — subheaders in [brackets], body/table text on following lines>",
  "notes": "<slide speaker notes / memo — may contain audience, KPIs, caveats>"
}

Return a JSON object with EXACTLY these fields (no others):
{
  "test_name": "<the test's name — usually the slide's sub_section verbatim; if the slide clearly names the test differently in its content/notes, use that>",
  "concept": "<one value from the APPROVED CONCEPTS list below, or null>",
  "component": ["<ONE OR MORE values from APPROVED COMPONENTS — the page type(s) or area(s) of the website where the experiment runs; NEVER []>"],
  "product": ["<ONE OR MORE product groups being tested — see PRODUCT EXAMPLES below; NEVER []>"],
  "hypothesis": "<the slide's body text under the 'Hypothesis' subheader, verbatim; null if absent>",
  "kpi": ["<ONE OR MORE values from APPROVED KPIS — every KPI the test tracks (primary AND secondary), de-duplicated; NEVER []>"],
  "target_audience": "<who the test runs on or is aimed at — see RULE 10; null ONLY when the slide gives no signal whatsoever>",
  "notes": "<see RULE 11 — captures test results (concluded tests), background/goals/objectives/opportunities, and caveats; null only when the slide truly has none of these beyond what the other columns cover>"
}

APPROVED CONCEPTS — `concept` MUST be an exact match from this list (case, punctuation, ampersand vs 'and' — all matter):
{CONCEPTS_LIST}

APPROVED COMPONENTS — each entry of `component` MUST be an exact match from this list. When the slide is genuinely ambiguous about where the test runs, use ["Multiple"] rather than an empty list:
{COMPONENTS_LIST}

PRODUCT EXAMPLES — these are common product groups but the list is NOT exhaustive. If the slide mentions a specific product (e.g. "Galaxy S26", "QLED 8K", "Bespoke Fridge") that isn't listed, include the product's name AS PRINTED on the slide — do NOT force-map it to something on the list:
{PRODUCTS_LIST}
When the slide does not mention any specific product group, use ["Total"] (the catch-all).

APPROVED KPIS — each entry of `kpi` MUST be an exact match from this list:
- CVR
- AOV
- Engagement Rate
- Add to Cart Rate
- Revenue per Visitor

RULES:
1. Read BOTH the `content` and the `notes` when populating kpi, component, product, and target_audience — a slide often puts these only in the memo.
2. `test_name` — default to the slide's `sub_section` verbatim. Override only if the slide's content or notes clearly names the test differently (e.g. an "Experiment ID" or a "Test: X" label whose text differs from the slide title). Never leave it null.
3. `kpi` is one combined, de-duplicated list — put every KPI the slide names (primary and secondary alike) in the same array. Must contain AT LEAST ONE value: if the slide names a metric not in the approved list (e.g. "click-through rate", "session engagement"), map it to the CLOSEST approved KPI. Never emit `[]`.
4. `component` — infer from where the test runs / where the change is shown on the site. Multiple pages qualifying → include all of them (e.g. `["Home Page", "PDP"]`). Must contain AT LEAST ONE value: when the location is unclear, use `["Multiple"]`. Never emit `[]`.
5. `product` — infer from the product group being tested. Multiple product groups → include all of them. Must contain AT LEAST ONE value: use `["Total"]` when no specific product is mentioned. Products outside the PRODUCT EXAMPLES list are welcome — include them as printed. Never emit `[]`.
6. `concept` — if the slide's test-type description is not on the APPROVED CONCEPTS list, choose the CLOSEST approved concept. Do not invent new concept names.
7. If a scalar field (`target_audience`, `notes`) truly cannot be inferred from either content or notes, use `null`. But do NOT default to null just because there is no explicit label — RULES 10 and 11 below require inference from the surrounding language. `test_name`, `component`, `product`, and `kpi` are never null / never `[]`.
8. hypothesis is the body text directly under a subheader named "Hypothesis" (or a very close synonym). If no such subheader exists, use null — do NOT paraphrase the slide.

9. TEST RESULTS — concluded tests share a repeating three-slot layout (an uplift/lift figure, a revenue/monetary figure, and a short learnings paragraph — each typically next to an icon). Capture EVERY slot you can find, in the slide OR its notes:
   a) Uplift / lift / significance figures — "+18% CVR Uplift", "+3.2% ATC", "flat", "no impact", "stat sig at 95%".
   b) Revenue / monetary impact — "£109K so far", "$1.2M projected", "+£40 AOV".
   c) Learnings / analysis paragraph — verbatim (or tightened to essentials): the "whilst / however / because" sentence that explains what happened (e.g. "Whilst this test produced an orders uplift, there was no improvement in the % of users taking out the finance proposition").
   d) Verdicts / next steps — "winner", "rolled out to 100%", "iterate", "kill", "hold".
   Prepend to `notes` as a `Result:` clause with items separated by `; ` (semicolons). Include the numbers VERBATIM (percent signs, currency symbols, magnitudes). Aim for one bullet per slot when present — do not collapse three findings into one. Example for the example slide above:
     "Result: +18% CVR Uplift; £109K revenue so far; Orders uplift observed, but no improvement in % of users taking out finance proposition — finance-option volumes minimal (<50) with no orders in either experience"
   If the slide ALSO has caveats/exclusions AND background (RULE 11), append them before the Result clause with " | " between segments:
     "Goal: ... | Caveats: ... | Result: ..."
   If there is no result-shaped content on the slide, skip the Result clause and just apply RULE 11. Sections that trigger result extraction: 'Concluded tests', 'Completed tests', 'Wrapped tests', or any subheader like 'Results', 'Outcome', 'Learnings', 'Impact'.

10. TARGET AUDIENCE — populate this whenever the slide gives ANY signal about who the test aims at, not just when a "Target audience" label appears. Sources to mine, in order of precedence:
    a) An explicit audience/segmentation line ("mobile users", "logged-in customers", "returning US visitors").
    b) The Background or Objective paragraph: language like "we aim to encourage more customers to ...", "for shoppers who ...", "users considering finance", "prospects browsing the configurator" — infer the implied audience from what the test is trying to influence.
    c) Product/scheme context: a trade-in scheme test implies "users with an eligible old device to trade in"; a finance-configurator test implies "shoppers considering finance on <product line>"; a mobile-only banner test implies "mobile visitors".
    d) The slide's speaker notes.
    Keep it CONCISE (a short noun phrase, e.g. "Samsung TV shoppers considering finance", "Users with an eligible old Galaxy phone to trade in", "Mobile visitors on the PDP"). Use null ONLY when the slide has zero audience-shaped signal in ANY of the above — not because the word "audience" is missing.

11. NOTES — the notes column is a catch-all for meaningful test context the other columns don't already carry. Populate it, in this order, joined with " | " between segments:
    a) `Goal: <one sentence>` — the test's stated goal / objective / opportunity / expected value drawn from Background. Skip if the hypothesis already fully covers it.
    b) `Caveats: <list>` — exclusions, watch-outs, risks, dependencies, known limitations.
    c) `Result: ...` clause when RULE 9 applies (concluded tests / results section).
    d) Any other short piece of speaker-note context that isn't already in another column (test-run window, exposure %, rollout plan, follow-up test link).
    Keep each segment tight — one sentence or a short semicolon-separated list. Skip a segment when it would add nothing. Use null only when there is genuinely nothing meaningful outside the other columns.

Return ONLY the JSON object. No prose, no code fences."""


def extract_test_metadata(
    client: OpenAI,
    model: str,
    payload: dict,
    concepts: list[str],
    components: list[str],
    products: list[str],
) -> dict:
    """Ask the LLM to project one test slide onto the historical_test table schema."""
    system_prompt = (
        TEST_EXTRACTION_PROMPT
        .replace("{CONCEPTS_LIST}", "\n".join(f"- {c}" for c in concepts))
        .replace("{COMPONENTS_LIST}", "\n".join(f"- {c}" for c in components))
        .replace("{PRODUCTS_LIST}", "\n".join(f"- {p}" for p in products))
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)
