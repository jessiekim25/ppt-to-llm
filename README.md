# ppt-to-llm

Convert a Samsung campaign visual identity PDF (or PPT export) into one structured **JSON record per slide**, written as a per-deck `slides.jsonl` file alongside one PNG screenshot per slide. The JSONL is designed to be fed straight to an LLM (or indexed for retrieval) without a database in the middle — layout is flexible, so hundreds of slides with wildly different structures all fit the same schema.

## Slide record schema

Each line in `slides.jsonl` is one slide. Slide-level fields are optional (omitted rather than `null`).

```json
{
  "doc_id": "2026_Galaxy_Miracle_VIS_Guidelines_v1_6",
  "slide_num": 42,
  "slide_id": "2026_Galaxy_Miracle_VIS_Guidelines_v1_6#042",

  "product": "Galaxy S26",
  "codename": "Miracle",
  "model": "Galaxy S26 Ultra",
  "section": "01 Brand Basics",
  "sub_section": "Hero Key Visual",

  "detail": [
    { "body": "Slide-level body text that isn't tied to any subheader." },
    { "table": "Approved backgrounds\nSurface | Hex\nPrimary | #111111\nAccent | #E4002B" },
    {
      "subheader": "Product Logo",
      "body": "The height of product logo should not exceed 90% of the SAMSUNG lettermark s-height.",
      "children": [
        { "table": "Sizing\nContext | Size\nPrint | 90%\nOOH | 80%" }
      ]
    },
    {
      "subheader": "Size ratio",
      "children": [
        { "body": "Size ratio (For OOH/Retails, please apply 80% of lettermark)" }
      ]
    }
  ],

  "slide_image_path": "slide_042.png"
}
```

Notes:

- **`slide_id`** = `{doc_id}#{slide_num:03d}` — stable primary key across re-runs, easy to reference from LLM outputs.
- **`detail`** is a list of blocks in reading order. A block has any of:
  - `subheader` — heading text.
  - `body` — paragraph text.
  - `table` — one table rendered as `title\ncol1 | col2 | ...\ncell11 | cell12 | ...`; multi-line cells are joined with ` / `. Each table is its own block, never mixed into a body string.
  - `children` — nested blocks with the same shape.
  Blocks omit fields they don't have — a slide-level paragraph is just `{"body": "..."}`, a table is `{"table": "..."}`, a heading that only owns a nested child is `{"subheader": "...", "children": [...]}`.
- **`slide_image_path`** is a basename (e.g. `slide_042.png`) so the images can be moved to any folder without breaking references.

## How it works

1. For each slide, walk the PDF with `pdfminer.six` to collect text lines (with bboxes, font size, bold flag) and vector/raster primitives (`LTImage`, `LTCurve`, `LTRect`, `LTLine`), then cluster nearby primitives into figure regions so the LLM knows which text lines are captions to skip.
2. Serialize the layout into a compact JSON payload — text lines + figure bboxes — and send it to an OpenAI text model (`gpt-4o` by default). The LLM returns the slide-level fields (product, codename, section, sub_section, model) plus a structured hierarchy of subheaders + tables. No image is sent to the LLM.
3. Render the slide to `slide_NNN.png` with `pypdfium2`.
4. Turn the LLM's subheader hierarchy (and any tables) into the `detail` block list, attach the screenshot basename as `slide_image_path`, and append one JSON record per slide to `<output-dir>/<deck-stem>/slides.jsonl`.

Text extraction is geometric (pdfminer) — the LLM only interprets typography + coordinates to reconstruct hierarchy. This eliminates vision-token cost and keeps proprietary slide artwork inside your environment.

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
| `--output-dir` | `output/images`  | Per-slide screenshots and `slides.jsonl` land here.                                       |
| `--codename`   | `""`             | Fallback for the `codename` field when not visible on a slide.                            |
| `--product`    | `""`             | Fallback for the `product` field when not visible on a slide.                             |
| `--dpi`        | `150`            | Render DPI for the per-slide screenshots.                                                 |
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
  slides.jsonl        # one JSON record per slide
  slide_001.png       # full-slide screenshot referenced by that record's slide_image_path
  slide_002.png
  ...
```

The JSON records store `slide_image_path` as a basename only, so the screenshots can be moved to any folder — as long as your reader knows where they live, references stay valid.

## Layout

```
src/
  extract.py         # CLI entry point; builds slide records and writes slides.jsonl
  pdf_layout.py      # pdfminer.six layout: text lines + clustered figure regions
  pdf_utils.py       # page rendering (pypdfium2) + zip input handling
  llm.py             # OpenAI text-only extraction (positioned text -> structured JSON)
shared/
  aws_secrets.py     # cached get_secret(name) via boto3
  settings.py        # get_settings() -> frozen Settings dataclass
secrets.example.json # template for the AWS Secrets Manager secret payload
requirements.txt
```
