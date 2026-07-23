# ppt-to-llm

Convert a Samsung campaign visual identity PDF (or PPT export) into one structured **JSON record per slide**, written as a per-deck `slides.jsonl` file alongside per-slide image crops. The JSONL is designed to be fed straight to an LLM (or indexed for retrieval) without a database in the middle — layout is flexible, so hundreds of slides with wildly different structures all fit the same schema.

## Slide record schema

Each line in `slides.jsonl` is one slide. Fields are optional — omitted rather than `null` — so a title slide with no tables just has no `tables` key.

```json
{
  "doc_id": "2026_Galaxy_Miracle_VIS_Guidelines_v1_6",
  "source_file": "/path/to/deck.pdf",
  "source_type": "pdf",
  "slide_num": 42,
  "slide_id": "2026_Galaxy_Miracle_VIS_Guidelines_v1_6#042",

  "product": "Galaxy S26",
  "codename": "Miracle",
  "model": "Galaxy S26 Ultra",
  "section": "01 Brand Basics",
  "sub_section": "Hero Key Visual",

  "detail": "Slide-level body text that isn't tied to any subheader.",

  "subheaders": [
    {
      "title": "4:1 proportion",
      "detail": "...",
      "children": [
        { "title": "How to build layout", "detail": "1. ...\n2. ..." }
      ]
    }
  ],

  "tables": [
    {
      "title": "Approved backgrounds",
      "columns": ["Surface", "Hex", "Usage"],
      "rows": [["Primary", "#111111", "Global"]]
    }
  ],

  "images": [
    {
      "idx": 1,
      "label": "hero_front",
      "bbox_pct": [0.05, 0.10, 0.95, 0.70],
      "path": "slide_042_img_01__hero_front.png"
    }
  ],
  "slide_image_path": "slide_042.png",

  "extraction": {
    "model": "gpt-4o",
    "extracted_at": "2026-07-23T10:15:00+00:00",
    "source_hash": "sha256:..."
  }
}
```

Notes:

- **`slide_id`** = `{doc_id}#{slide_num:03d}` — stable primary key across re-runs, easy to reference from LLM outputs.
- **`subheaders`** stays recursive (subheaders can have `children` with the same schema), so nested layout survives round-trips.
- **`extraction.source_hash`** lets you re-run only slides from decks whose file hash changed after a prompt/model bump.

## How it works

1. Render each PDF page to PNG with `pypdfium2` (pure-Python, no Poppler needed).
2. Send each PNG to an OpenAI vision model (`gpt-4o` by default) with a strict JSON extraction prompt that returns structured text fields plus `images[]` (one entry per distinct visual region with a label + bbox).
3. Crop the rendered slide to each `bbox_pct` with generous padding and save it as `img_NN__<label>.png` in a per-slide assets folder.
4. Write one JSON record per slide, appended to `<output-dir>/<deck-stem>/slides.jsonl`.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The OpenAI key lives in **AWS Secrets Manager** — nothing sensitive touches the repo or `.env`:

| secret name | required keys                                              |
| ----------- | ---------------------------------------------------------- |
| `LLMKeys`   | `OPENAI_API_KEY`, `OPENAI_MODEL` (optional; default `gpt-4o`) |

Override the secret name with `LLM_SECRET_NAME` if needed. See `secrets.example.json` for the expected shape.

AWS credentials are picked up from the standard boto3 chain (`AWS_PROFILE`, `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, IAM role, `~/.aws/credentials`). Region comes from `AWS_REGION` or your profile.

## Run against the Galaxy Miracle guideline

The source file is a zip — the CLI unpacks it automatically if you point `--pdf` at the `.zip`.

```bash
# Windows
python -m src.extract ^
  --pdf "C:\Users\yebin.kim\2026 Galaxy Miracle VIS Guidelines_v1.6_260116_compressed.pdf.zip" ^
  --codename "Miracle" ^
  --product "Galaxy S26" ^
  --output-dir "C:\Users\yebin.kim\brand_guideline_images"
```

`--codename` and `--product` are optional fallbacks used only when the model can't read them from the slide itself.

### Options

| flag           | default          | notes                                                                                     |
| -------------- | ---------------- | ----------------------------------------------------------------------------------------- |
| `--output-dir` | `output/images`  | Rendered slide PNGs, per-slide assets folders, and `slides.jsonl` all land here.          |
| `--codename`   | `""`             | Fallback for the `codename` field when not visible on a slide.                            |
| `--product`    | `""`             | Fallback for the `product` field when not visible on a slide.                             |
| `--dpi`        | `150`            | Render DPI for slide PNGs.                                                                |
| `--limit`      | `0` (all)        | Only process the first N slides. Ignored if `--pages` is set.                             |
| `--pages`      | `""` (all)       | Specific slide numbers, e.g. `42` or `10-15,42,100-105`.                                  |
| `--dry-run`    | off              | Print records to stdout instead of writing `slides.jsonl`.                                |

### Try a few slides first

```bash
python -m src.extract --pdf "...pdf.zip" --pages 10-15 --dry-run
```

## Output layout

```
output/images/<deck-stem>/
  slides.jsonl                  # one JSON record per slide
  slide_001.png                 # full-page render (source for crops)
  slide_001/                    # per-slide assets folder
    slide_001_img_01__<label>.png   # cropped visual regions, one per image on the slide
    slide_001_img_02__<label>.png
    ...
  slide_002.png
  slide_002/
    ...
```

Each `slide_NNN_img_NN__<label>.png` is named after the header/title text next to the visual on the slide. When a visual is marked only by a numbered gray-circle badge (common in Format tables), the badge digit becomes the label (`slide_042_img_01__1.png`, `slide_042_img_02__2.png`, …), so files line up 1-to-1 with the `Format N: …` rows in the extracted content. Filenames include the slide number so they stay unique when copied out to a flat folder — the JSON records store only the basename, so relocating the images just means pointing your reader at whichever folder holds them.

## Layout

```
src/
  extract.py         # CLI entry point; builds slide records and writes slides.jsonl
  pdf_utils.py       # render_pdf_pages, crop helpers
  llm.py             # OpenAI vision extraction
shared/
  aws_secrets.py     # cached get_secret(name) via boto3
  settings.py        # get_settings() -> frozen Settings dataclass
secrets.example.json # template for the AWS Secrets Manager secret payload
requirements.txt
```
