import base64
import json
from pathlib import Path

from openai import OpenAI

SYSTEM_PROMPT = """You extract structured metadata from a single slide of a Samsung campaign visual identity guideline deck.

Return a JSON object with exactly these string fields (use "" when a field is not visible on the slide):

- product: the general phone series/line shown (e.g. "Galaxy S", "Galaxy Z Flip", "Galaxy Watch"). "" if the slide is generic and shows no product line.
- codename: the campaign project code name (e.g. "Miracle"). Only set this if the code name literally appears on the slide; do not guess from context.
- section: the top-level section label printed at the very top-left of the slide (e.g. "01 Brand Basics", "Logo", "Color System", "Typography"). This deck is divided into four sections and each slide carries its section marker in the top-left. "" if nothing is there.
- sub_section: the slide's main title/heading (usually the largest text near the top).
- detail: the main body text of the slide as plain text — bullet points, captions, spec descriptions, do's and don'ts, all preserved as readable prose. Keep it faithful; do not summarize away specifics like hex codes, pixel values, or ratios.
- model: the specific phone model depicted (e.g. "Galaxy S26 Ultra"). "" if none identifiable.

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
