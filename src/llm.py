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
- table: array of objects, one entry per Format value in any Format / File name table on the slide.
  IMPORTANT: some slides show one logical table laid out as TWO side-by-side Format/File-name column pairs (e.g. the left pair carries Format 1-2 and the right pair carries Format 3-4). Treat these as ONE continuous list of entries sorted by Format number — do not emit duplicate columns and do not skip the right-hand pair.
  Each entry: {"format": "<value in Format column, usually a number>", "file_names": ["<file>", "<file>", ...]}. file_names is a list of every file name listed under that Format cell, in the order they appear (cells often contain multiple names on separate lines). Return [] if no such table.

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
