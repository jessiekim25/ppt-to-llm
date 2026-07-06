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
- detail: general body text on the slide that is NOT tied to a specific image region — introductory paragraphs, footnotes, do's & don'ts that describe the whole slide. Preserve specifics (hex codes, pixel values, ratios). "" if all text is region-specific.
- table: array of objects, one per row of any Format / File name table on the slide. Each row: {"format": "<value in Format column, usually a number>", "file_name": "<value in File name column>"}. Return [] if no such table.
- regions: array describing every visually distinct image region on the slide — photos, phone mockups, logo lockups, illustration panels, gray placeholder boxes marked "TBU", each visually separated graphic block. Do NOT include text boxes, tables, section headers, or the slide title as regions. Each region is an object:
  {
    "label": "<the visible number/name attached to the region if any, e.g. '1' or '2'; if unnumbered, a short descriptor like 'Main visual' or 'AP(Gaming)'>",
    "bbox_pct": [x1, y1, x2, y2],   // left, top, right, bottom as fractions 0-1 of slide width/height
    "description": "<one sentence: what the region shows>",
    "associated_text": "<any text on the slide that specifically refers to this region, including table rows whose Format matches the label>"
  }

If the slide has no image regions at all, return "regions": [].
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
