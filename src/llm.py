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
- Text lines aligned into a grid (same x0/x1 across rows AND same y0 across columns) MAY be a table — but grid alignment alone is not enough. Apply the strict table test below before emitting one.
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
- codename: campaign code name (e.g. "Miracle"). Only set if literally on the slide.
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

- tables: array of TEXT-ONLY tables. A REAL TABLE has ALL of:
    (i) A row of 2+ short column HEADERS at similar y (e.g. "Product KV" / "Disclaimer", "Format" / "File name", "Element" / "Spec" / "Notes", "Surface" / "Hex" / "Usage").
    (ii) AT LEAST TWO DATA ROWS below the headers, each row having a cell in every column and all cells of a row at similar y.
    (iii) Cells in each column are PARALLEL IN KIND across rows — e.g. one column of product-KV names paired with one column of disclaimer text, one column of filenames paired with one column of format numbers.

  If any of these fails — ESPECIALLY if there is only ONE row of prose bodies beneath the headings — it is NOT a table. In that case, emit each column as its OWN subheader entry (title = the column heading, detail = its prose body). Examples that ARE multi-column subheader layouts, NOT tables:
    - "Position" / "Eco label size" / "Clear Space" / "Visibility" / "Responsibility" each with one paragraph beneath.
    - "AP(Gaming)" / "Display Innovation" each with one paragraph beneath.
    - "Size" / "Arrangement" / "Hierarchy" each with one paragraph beneath.

  Table entry format:
  {
    "title": "<any caption or title printed above the table, or \"\" if none>",
    "columns": ["<first column header exactly as printed>", "<second column header>", ...],
    "rows": [
      ["<row 1 cell 1>", "<row 1 cell 2>", ...],
      ["<row 2 cell 1>", "<row 2 cell 2>", ...]
    ]
  }
  Use the column headers actually printed on the slide; never invent, rename, translate, or substitute a header.
  Cells often contain multiple lines — join those with "\\n" inside the single cell string.
  NUMBERED BADGE CELLS: small isolated single digits ("1", "2", "3", ...) inside a table cell are numbered badges — return the digit verbatim as the cell value. Never leave the cell empty.
  When one logical table is laid out visually as TWO side-by-side identical-header column pairs (e.g. two "Format | File name" pairs stacked side by side), treat it as ONE table with one set of column headers and all rows concatenated in reading order.
  Return "tables": [] if the slide has no tables.

- subheaders: array describing every distinct heading + descriptive-text pair on the slide, other than the main slide title itself. A subheader is any bold or larger-font short label that introduces a block of descriptive body text OR a nested sub-block. Includes:
    * sub-titles that horizontally divide the slide into sections;
    * bold column headings at the top of side-by-side text blocks in a multi-column layout — these ARE column anchors (see COLUMN STRUCTURE);
    * labels marking each cell of a grid layout, with a paragraph next to or below;
    * bold captions under figures that name each panel type — text sitting under a figures[i].bbox is a caption (see above); text sitting BETWEEN two figures with descriptive text below IS a subheader.
  Capture ALL such headings in reading order (top-to-bottom then left-to-right, respecting COLUMN STRUCTURE above). For each one, put the full descriptive body text next to/below that heading into the subheader's `detail` field, verbatim and complete.

  STRICT RULES:
    (a) EVERY heading must be its OWN entry with its heading text in the `title` field. Do NOT collapse multiple headings into one subheader's `detail` as a bulleted list.
    (b) `title` contains ONLY the heading text — never the description, never a leading dash or bullet.
    (c) `detail` contains the full descriptive body text for THAT subheader only — never other subheaders' titles as bullets, never content that belongs to a different column.
    (d) Do not invent headings, and do not use the main slide title as a subheader.
    (e) NESTING: if a subheader's visual area contains another labeled sub-block below it (e.g. "4:1 proportion" column contains a "How to build layout:" heading with a numbered list beneath), put that inner sub-block in the parent's `children` array — do NOT flatten it, do NOT stuff the child's content into the parent's `detail`, and do NOT lift the child to the slide-level detail. A child subheader has the same schema as its parent and can itself have `children`.
    (f) A subheader must have at least one of: non-empty `detail`, non-empty `tables`, non-empty `children`. If a candidate heading has none of these, it's a caption or noise — omit it entirely (see EMPTY SUBHEADER RULE).

  Each entry:
  {
    "title": "<the heading text exactly as printed>",
    "detail": "<all descriptive body text under/next to this heading (excluding any child subheaders' content), verbatim; \"\" if none>",
    "tables": [ <text-only table objects that belong to this subheader> ],
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


def extract_slide(client: OpenAI, model: str, payload: dict) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)
