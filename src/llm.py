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
- table: array of Format entries that belong to the slide as a whole (not tied to any subheader). Format the same way as subheader tables (see below). Return [] if no such table or if all tables are tied to subheaders.
- subheaders: array describing the sub-titles that visually divide the slide into sections BELOW the main slide title. Many slides have one or more subheaders (e.g. a Campaign Assets slide split into "AP(Gaming)" and "Display Innovation" side by side). Each subheader groups the text/table/images that sit under it. If the slide has no subheaders (only a main title and one flat block of content), return "subheaders": [].
  Each entry:
  {
    "title": "<the subheader text exactly as printed>",
    "detail": "<body text under this subheader, preserving specifics; \"\" if none>",
    "table": [ <Format entries that belong to this subheader> ]
  }
- panels: array of the labeled visual blocks on the slide. Use this for slides that show one or more diagrams, annotated illustrations, layout examples, or any compound visual unit that reads as a self-contained titled block (each panel has a heading/label above its graphic and may contain numbered callouts, captions, or dimension lines). Return [] for slides whose visuals are just a single unlabeled product mockup or photo (native image extraction handles those).
  Each panel:
  {
    "label": "<the panel's title/heading exactly as printed on the slide>",
    "bbox_pct": [x1, y1, x2, y2],  // fractions 0-1 of slide width/height (left, top, right, bottom). Include the panel's TITLE at the top, the visual itself, any numbered callouts on the visual, AND any caption text below it. Give a generous margin so nothing is cut off — err on the larger side. Two panels' boxes must not overlap.
    "description": "<one sentence describing what the panel shows>"
  }

Table entry format (used in both `table` and `subheaders[].table`):
  {"format": "<Format column value, usually a number>", "file_names": ["<file>", "<file>", ...]}
  file_names is a list of every file name listed under that Format cell, in the order they appear (cells often contain multiple names on separate lines).
  IMPORTANT: when a table under a subheader shows one logical list laid out as TWO side-by-side Format/File-name column pairs (e.g. the left pair carries Format 1-2 and the right pair carries Format 3-4), treat them as ONE continuous list sorted by Format number — do not emit duplicate columns and do not skip the right-hand pair.

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
