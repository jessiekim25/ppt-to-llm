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
All bboxes are TOP-LEFT origin, fractions of page (0-1). Reading order = top to bottom, then left to right (compare y0 first, then x0).

Use text_lines' geometry and typography to reconstruct layout:
- Larger `size` or `bold: true` marks a heading (section label, slide title, subheader).
- Text lines whose bboxes share the same x0/x1 across multiple rows are a single column.
- Multiple columns with parallel bold headings at similar y0 form a multi-column subheader layout.
- Text lines aligned into a grid (same x0/y0 patterns across rows and columns) are a table.
- Text lines whose bbox falls INSIDE (or directly below) a `figures[i].bbox` are captions of that figure — do NOT put them in `detail` or `subheaders`; the caption is already in `figures[i].label`.

Return a JSON object with these fields.

Slide-level string fields ("" if not visible):
- product: general phone series (e.g. "Galaxy S"). "" if generic.
- codename: campaign code name (e.g. "Miracle"). Only set if literally on the slide.
- section: top-level section label at the very top-left of the slide (e.g. "01 Brand Basics", "Campaign Assets"). "" if none.
- sub_section: the slide's main title/heading (typically the largest text near the top of the slide, not counting the section label).
- model: specific phone model shown (e.g. "Galaxy S26 Ultra"). "" if none.

Content fields:
- detail: general body text on the slide that is NOT tied to any subheader (see below) — introductory paragraphs, footnotes, do's & don'ts.
  IMPORTANT: numbered legends must be captured here in full. A numbered legend is a vertical or side-by-side list of items where each item begins with a small isolated single digit (1, 2, 3, ...) followed by a short label and (optionally) a description. Capture every legend item verbatim, one per line, formatted as "N: <label> — <description>" (drop "—" if there's no description). A short isolated digit text line adjacent to a descriptive text line is almost certainly a numbered legend entry, even though the circle around the digit doesn't appear in the payload.
  Preserve specifics (hex codes, pixel values, ratios). "" if there is truly no slide-level body text at all.

  Slide-wide text that must always land in this slide-level `detail` field, regardless of layout:
    * running text above a horizontal rule that introduces the slide (e.g. "Type family and weight distribution.");
    * footnotes, disclaimers, or fine print at the very bottom of the slide (small `size`, near the bottom of the page).
  Do NOT pull caption text that sits under a figure into `detail` — captions are in `figures[i].label` and should not be duplicated.

- tables: array of TEXT-ONLY tables — text lines arranged in a grid (same x0/x1 across rows). Each entry:
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

- subheaders: array describing every distinct heading + descriptive-text pair on the slide, other than the main slide title itself. A subheader is any bold or larger-font short label that introduces a block of descriptive body text. Includes:
    * sub-titles that horizontally divide the slide into sections;
    * bold column headings at the top of side-by-side text blocks in a multi-column layout;
    * labels marking each cell of a grid layout, with a paragraph next to or below;
    * bold captions under figures that name each panel type — text sitting under a figures[i].bbox is a caption (see above); text sitting BETWEEN two figures with descriptive text below IS a subheader.
  Capture ALL such headings in reading order (top-to-bottom then left-to-right). For each one, put the full descriptive body text next to/below that heading into the subheader's `detail` field, verbatim and complete.

  STRICT RULES:
    (a) EVERY heading must be its OWN entry with its heading text in the `title` field. Do NOT collapse multiple headings into one subheader's `detail` as a bulleted list.
    (b) `title` contains ONLY the heading text — never the description, never a leading dash or bullet.
    (c) `detail` contains the full descriptive body text for THAT subheader only — never other subheaders' titles as bullets.
    (d) Do not invent headings, and do not use the main slide title as a subheader.
    (e) NESTING: if a subheader's visual area contains another labeled sub-block below it, put that inner sub-block in the parent's `children` array — do NOT flatten it and do NOT stuff the child's content into the parent's `detail`. A child subheader has the same schema as its parent and can itself have `children`.

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
    """Serialize a PageLayout into the compact JSON payload the LLM prompt expects."""
    text_lines = []
    for tl in layout.text_lines:
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
