import base64
import json
from pathlib import Path

from openai import OpenAI

SYSTEM_PROMPT = """You extract structured data from a single slide of a Samsung campaign visual identity guideline deck.

Return a JSON object with these fields.

Slide-level string fields ("" if not visible):
- product: general phone series (e.g. "Galaxy S"). "" if generic.
- codename: campaign code name (e.g. "Miracle"). Only set if literally on the slide.
- section: top-level section label at the very top-left of the slide (e.g. "01 Brand Basics", "Campaign Assets"). "" if none.
- sub_section: the slide's main title/heading.
- model: specific phone model shown (e.g. "Galaxy S26 Ultra"). "" if none.

Content fields:
- detail: general body text on the slide that is NOT tied to any subheader (see below) — introductory paragraphs, footnotes, do's & don'ts.
  IMPORTANT: numbered legends must be captured here in full. A numbered legend is a vertical or side-by-side list of items where each item begins with a small number in a circle (1, 2, 3, ...) followed by a short label and (optionally) a description. These legends explain the numbered callouts shown elsewhere on the slide (usually on top of an image or diagram). Capture every legend item verbatim, one per line, formatted as "N: <label> — <description>" (drop "—" if there's no description). Do NOT skip legends because they look like a caption or key.
  Preserve specifics (hex codes, pixel values, ratios). "" if there is truly no slide-level body text at all.

  Slide-wide text that must always land in this slide-level `detail` field, regardless of layout:
    * running text above a horizontal rule that introduces the slide (e.g. "Type family and weight distribution.");
    * footnotes, disclaimers, or fine print at the very bottom of the slide.
  Do NOT pull short caption sentences printed under an image into detail — those belong to the image and should stay part of it.
- tables: array of TEXT-ONLY tables on the slide that are not tied to any subheader. Skip tables whose cells contain images/graphics — those go to panels (see below). Each table entry captures its own real column headers and row cells verbatim from the slide:
  {
    "title": "<any caption or title printed above the table, or \"\" if none>",
    "columns": ["<first column header exactly as printed>", "<second column header>", ...],
    "rows": [
      ["<row 1 cell 1>", "<row 1 cell 2>", ...],
      ["<row 2 cell 1>", "<row 2 cell 2>", ...]
    ]
  }
  Real slide tables use whatever column labels the designer wrote — "Format" / "File name", "Logo/lock-up" / "Name", "Element" / "Spec" / "Notes", etc. Use the column headers actually printed on the slide; never invent, rename, translate, or substitute a header (in particular, do NOT stamp "Format" onto columns whose header is something else).
  Cells often contain multiple lines (e.g. several file names stacked in one File name cell) — join those with "\\n" inside the single cell string; do not split a multi-line cell into extra rows.
  When one logical table is laid out visually as TWO side-by-side identical-header column pairs (e.g. two "Format | File name" pairs stacked side by side, left carrying rows 1-2 and right carrying rows 3-4), treat it as ONE table with one set of column headers and all rows concatenated in the natural reading order.
  Return "tables": [] if the slide has no text-only tables (including when the only tables are mixed image+text — those are panels).
- subheaders: array describing every distinct heading + descriptive-text pair on the slide, other than the main slide title itself. A subheader is any bolded, highlighted, or otherwise-emphasized short label that introduces a block of descriptive body text. This includes but is not limited to:
    * sub-titles that horizontally divide the slide into sections (e.g. "AP(Gaming)" / "Display Innovation");
    * bold column headings at the top of side-by-side text blocks in a multi-column layout (e.g. three columns headed "Size" / "Arrangement" / "Hierarchy", each with descriptive text below);
    * color-highlighted labels marking each cell of a grid layout (e.g. yellow-highlighted "KV order", "Product logos", "Combining visuals", "Product positioning of X Series KV") — the highlight color/box marks the heading, and the paragraph next to or below it is that subheader's detail;
    * bold captions under a row of images that name each panel type (e.g. "Single panel" / "L-shaped panel" / "Multi-panels", each followed by 1-2 sentences of description). Text captions sitting UNDER images always count — do not skip them because they look like image annotations.
  Capture ALL such headings on the slide, in reading order (top-to-bottom then left-to-right). For each one, put the full descriptive body text next to/below that heading into the subheader's `detail` field, verbatim and complete — do not summarize and do not drop sentences. Return "subheaders": [] only when the slide is truly one flat block of content with no repeated column/section headings anywhere.

  STRICT RULES — read carefully:
    (a) EVERY heading on the slide must be its OWN entry in the subheaders array with its heading text in the `title` field. Do NOT collapse multiple headings into a single subheader's `detail` as a bulleted or dash-prefixed list. If your output would contain a line like "- Size: ..." or "Size: ..." followed by "- Arrangement: ..." inside one subheader's `detail`, you have merged separate subheaders into one — split them into distinct entries instead.
    (b) The `title` field must contain ONLY the heading text (e.g. "Size", "KV order", "Single panel") — never the description, never a leading dash or bullet.
    (c) The `detail` field must contain the full descriptive body text for THAT subheader only — never other subheaders' titles as bullets.
    (d) Do not invent headings, and do not use the main slide title as a subheader.
    (e) NESTING: subheaders can nest. If a subheader's visual area contains another labeled sub-block below it (e.g. a "4:1 proportion" column that contains a "How to build layout:" sub-heading with a numbered list underneath), put that inner sub-block in the parent's `children` array — do NOT flatten it into the top-level subheaders list, and do NOT stuff the child's content into the parent's `detail`. This is what preserves "this How-to-build-layout belongs to 4:1, that one belongs to 6:1". A child subheader has the same schema as its parent and can itself have `children`.

  Each entry:
  {
    "title": "<the heading text exactly as printed>",
    "detail": "<all descriptive body text under/next to this heading (excluding any child subheaders' content), verbatim; \"\" if none>",
    "tables": [ <text-only table objects that belong to this subheader, same schema as the slide-level `tables` field> ],
    "children": [ <nested subheader entries with the same schema; [] if none> ]
  }
- panels: array of the labeled visual blocks on the slide. Use this for:
    (i) diagrams, annotated illustrations, layout examples, or any compound visual unit that reads as a self-contained titled block (each panel has a heading/label above its graphic and may contain numbered callouts, captions, or dimension lines);
    (ii) MIXED-CONTENT TABLES — any table that contains images/graphics in one or more cells alongside text in other cells (e.g. a two-column table with a "Logo/lock-up" column showing product logo images and a "Name" column showing text names). For a mixed table, return ONE panel whose bbox spans the ENTIRE table including its header row and every row of cells. Do NOT try to describe such tables as text entries in `table` or in `detail` — the whole table becomes an image.
  Return [] for slides whose visuals are just a single unlabeled product mockup or photo (native image extraction handles those).
  Each panel:
  {
    "label": "<the panel's title/heading exactly as printed on the slide; for a mixed-content table with no separate title, use the concatenated column headers (e.g. \"Logo/lock-up | Name\")>",
    "bbox_pct": [x1, y1, x2, y2],  // fractions 0-1 of slide width/height (left, top, right, bottom). Include the panel's TITLE at the top, the visual itself, any numbered callouts on the visual, AND any caption text below it. Give a generous margin so nothing is cut off — err on the larger side. Two panels' boxes must not overlap.
    "description": "<one sentence describing what the panel shows>",
    "subheader_path": [ "<outermost subheader title>", ..., "<innermost subheader title>" ]  // the ordered list of subheader titles (from the `subheaders`/`children` tree) whose visual area encloses this panel, outermost first. E.g. a diagram sitting inside a "How to build layout:" child of a "4:1 proportion" parent -> ["4:1 proportion", "How to build layout:"]. Return [] if the panel is not inside any subheader (e.g. a full-slide diagram directly under the main title).
  }

Return ONLY the JSON object. No prose, no code fences."""


def extract_slide(client: OpenAI, model: str, image_path: Path) -> dict:
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    data_url = f"data:image/png;base64,{b64}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "Extract the fields from this slide."},
                ],
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)
