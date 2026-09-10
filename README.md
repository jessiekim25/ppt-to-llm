# ppt-to-llm

Convert Samsung campaign visual identity **PDFs or PowerPoint (`.pptx`) decks** into structured **JSON records per slide**, one per-file `slides.jsonl` for display, one per-file `chunks.jsonl` for retrieval, and a combined `corpus/chunks.jsonl` that concatenates every file for a downstream RAG agent. Layout is flexible, so hundreds of slides with wildly different structures all fit the same schema.

## Slide record schema

Each line in `slides.jsonl` is one slide. Slide-level fields are optional (omitted rather than `null`).

```json
{
  "slide_id": "2026_Galaxy_Miracle_VIS_Guidelines_v1_6#042",
  "doc_id": "2026_Galaxy_Miracle_VIS_Guidelines_v1_6",

  "product": "Galaxy S26",
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

- **`slide_id`** = `{doc_id}#{NNN}` (three-digit page number) — globally unique primary key, easy to reference from LLM outputs and stable across re-runs.
- **`detail`** is a list of blocks in reading order. A block has any of:
  - `subheader` — heading text.
  - `body` — paragraph text.
  - `table` — one table rendered as `title\ncol1 | col2 | ...\ncell11 | cell12 | ...`; multi-line cells are joined with ` / `. Each table is its own block, never mixed into a body string.
  - `children` — nested blocks with the same shape.
  Blocks omit fields they don't have — a slide-level paragraph is just `{"body": "..."}`, a table is `{"table": "..."}`, a heading that only owns a nested child is `{"subheader": "...", "children": [...]}`.
- **`slide_image_path`** is a basename (e.g. `slide_042.png`) so the images can be moved to any folder without breaking references.

## Retrieval chunks (`chunks.jsonl`)

For each file we also emit a retrieval-shaped file alongside `slides.jsonl`. One line per slide, flat scalars only, ready for a downstream RAG agent to embed and index:

```json
{
  "slide_id":   "2026_Galaxy_Miracle_VIS_Guidelines_v1_6#042",
  "doc_id":     "2026_Galaxy_Miracle_VIS_Guidelines_v1_6",
  "image_path": "files/2026_Galaxy_Miracle_VIS_Guidelines_v1_6/slide_042.png",
  "embed_text": "Galaxy S26 — 01 Brand Basics / Hero Key Visual. Product Logo: Height must not exceed 90% of lettermark. Table: Sizing. Context — Print, Size — 90%. Context — OOH, Size — 80%.",
  "metadata": {
    "product": "Galaxy S26",
    "model": "Galaxy S26 Ultra",
    "section": "01 Brand Basics",
    "sub_section": "Hero Key Visual",
    "has_table": true,
    "has_subheaders": true,
    "is_visual_only": false
  }
}
```

- **`embed_text`** — deterministic flatten of the slide (product header + subheader-prefixed body sentences + tables rendered as prose). Feed this straight to any embedding model.
- **`metadata`** — flat scalars for hybrid-search filters. Nested structure omitted so any vector DB (pgvector, Qdrant, LanceDB, Pinecone) can filter on them.
- **`image_path`** — path relative to the `output/` root, so the combined corpus file is self-contained.
- **`is_visual_only`** — flagged when `embed_text` is nearly empty; downstream can decide whether to caption these separately (this repo doesn't call an LLM for that).

## Corpus (`output/corpus/chunks.jsonl`)

After extract, all per-file `chunks.jsonl` files are concatenated into one combined file for downstream ingestion:

```
output/
  files/
    <doc_id>/
      slides.jsonl        # display shape
      chunks.jsonl        # retrieval shape (authoritative per file)
      slide_NNN.png
  corpus/
    chunks.jsonl          # concatenation of every file's chunks.jsonl
    manifest.json         # {doc_id: {chunk_count, sha256, built_at, source}}
```

Per-file `chunks.jsonl` stays authoritative. `corpus/chunks.jsonl` is a derived artifact — safe to delete and rebuild anytime. `slide_id` is globally unique (`{doc_id}#NNN`), so concatenation has no collisions and downstream can upsert incrementally.

Extract auto-rebuilds the corpus after each run — so if you extract file A today and file B tomorrow, `corpus/chunks.jsonl` includes both after tomorrow's run without any extra step. Skip the rebuild with `--no-corpus` if you want to batch several extractions before consolidating.

## How it works

The pipeline dispatches on input type; both paths produce the same slide/chunk schema.

**PDF path** (`--pdf`):
1. For each slide:
   - `pdfminer.six` collects text lines (with bboxes, font size, bold flag) at paragraph (LTTextBox) granularity, plus vector/raster primitives that cluster into figure regions.
   - `pdfplumber` detects any ruled tables on the page and returns their columns/rows/bboxes. Text lines whose center falls inside a detected table bbox are dropped from the LLM payload so the pre-extracted table content stays authoritative.
2. Serialize the layout into a compact JSON payload — text lines + figure bboxes — and send it to an OpenAI text model (`gpt-4o` by default). The LLM returns the slide-level fields (product, section, sub_section, model) plus a structured hierarchy of subheaders. No image is sent to the LLM.
3. Render the slide to `slide_NNN.png` with `pypdfium2`.
4. Compose the `detail` block list — slide body, pdfplumber's tables, then the LLM's subheader hierarchy — attach the screenshot basename as `slide_image_path`, and append one JSON record per slide to `<output-dir>/<file-stem>/slides.jsonl`.

Text and table extraction are geometric (pdfminer + pdfplumber) — the LLM only interprets typography + coordinates for hierarchy. This eliminates vision-token cost and keeps proprietary slide artwork inside your environment.

**PPTX path** (`--pptx`):
1. For each slide:
   - `python-pptx` walks the slide's shape tree and emits one text line per paragraph (with bbox, font size, bold) plus tables (columns + rows) and picture bboxes — no geometric reconstruction needed because pptx already stores positions natively.
2. Same LLM call as the PDF path, but with a **PPT-specific system prompt**: no top-left "section" chrome to extract. Instead the LLM tags each slide with `is_section_intro: true|false`.
   - A section-intro slide is one whose sole purpose is naming a new section — a big title like "Roadmap", "March review", or "Live tests" with little else. Its title becomes the `section` for that slide **and every subsequent slide** until the next section-intro slide.
   - Regular content slides leave `section` empty; the extractor propagates the last-seen intro's section forward in post-processing.
3. Render the slide to `slide_NNN.png` by first converting the pptx to a companion PDF via `soffice --headless --convert-to pdf` (LibreOffice), then reusing the same `pypdfium2` renderer as the PDF path. The intermediate PDF is cached next to the pptx and reused across runs.
4. Compose and write records exactly like the PDF path.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**Extra dependency for `--pptx`**: [LibreOffice](https://www.libreoffice.org/) so pptx slides can be rendered to PNG via a companion PDF.

```bash
# macOS
brew install --cask libreoffice
# Debian/Ubuntu
sudo apt-get install libreoffice
# Windows: download the installer from https://www.libreoffice.org/download
# The default install path (C:\Program Files\LibreOffice\program\soffice.exe)
# is auto-detected; you don't need to add it to PATH.
```

Only need the JSON output right now and don't want to install LibreOffice? Pass `--no-images` on the `--pptx` command to skip the render step — `slides.jsonl` and `chunks.jsonl` are still produced.

**Image-only slides.** When a `--pptx` slide is essentially a single picture with no extractable text (a screenshot, a chart image, a product shot), the deterministic detail builder can't produce anything. In that case the pipeline sends the rendered slide PNG to the same OpenAI model as a vision fallback and asks it to read `sub_section` + body text off of the image itself. The fallback fires when the slide has ≥1 picture shape and ≤1 non-title text line, and only when a rendered PNG exists (i.e. not under `--no-images`). Disable with `--no-vision` if you want to skip the extra API cost.

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
  --product "Galaxy S26" ^
  --output-dir "C:\Users\yebin.kim\brand_guideline_output\files"
```

Extract also builds this file's `chunks.jsonl` and refreshes the combined `corpus/chunks.jsonl`. Add `--no-chunk` or `--no-corpus` to skip either step (e.g. when batch-extracting several files before consolidating).

## Run against a PowerPoint deck

Point `--pptx` at a `.pptx` (or a `.zip` containing one). Everything else — output layout, chunks, corpus refresh — is identical to the PDF path.

```bash
# Windows
python -m src.extract ^
  --pptx "C:\Users\yebin.kim\March_campaign_review.pptx" ^
  --output-dir "C:\Users\yebin.kim\brand_guideline_output\files"
```

The first run per deck spends a few seconds converting the pptx to a companion PDF via LibreOffice; that PDF is cached beside the pptx and skipped on subsequent runs (regenerated only if the source pptx is newer).

**Picking one deck out of a multi-file .zip.** If the archive at `--pptx` holds several `.pptx` files, extract lists them and exits so you can pick one with `--pick <substring>` (case-insensitive filename match):

```bash
python -m src.extract --pptx path/to/many_decks.zip
# [pptx] many_decks.zip contains 3 .pptx files; choose one with --pick
#        .pptx files inside many_decks.zip:
#          - roadmap_deck.pptx
#          - march_review.pptx
#          - live_tests.pptx

python -m src.extract --pptx path/to/many_decks.zip --pick march
# extracts march_review.pptx and processes only that deck
```

`--pick` is ignored when the zip has a single `.pptx` or when `--pptx` points at a bare `.pptx`.

**Section behavior for pptx.** Section labels are not read from a top-left header (as they are in the guideline PDF). Instead, the extractor looks for "section-intro" slides — slides whose only content is a big title naming the next section (e.g. `Roadmap`, `March review`, `Live tests`). The intro slide's title becomes the `section` for that slide and for every following slide until the next intro slide.

## Test-section slides (`tests.jsonl` + MySQL sink)

Slides whose propagated `section` names a test bucket — e.g. `Live tests`, `Flagship tests`, `Upcoming tests`, `Blocked tests`, any section whose name matches the whole word `test`/`tests` — are treated as one-experiment specs and projected onto a fixed table schema instead of being emitted as free-form `detail` in `slides.jsonl`.

For each test slide the extractor:

1. Sends the flattened slide content + speaker notes to the LLM with the `cro.historical_test` schema and the approved-value lists (concepts, components, products, KPIs).
2. Coerces any drift — a `concept`, `component`, `product`, or `kpi` value that isn't an exact match from its approved list is dropped rather than shipped.
3. Saves every picture on the slide's right half (bbox-center x ≥ 0.5) to `<per-file-dir>/test_images/test_<slide-num>_<i>.<ext>` — the paths land in the row's `image_path` column so downstream can render the mockup next to the extracted fields.
4. Writes one row per test slide to `<per-file-dir>/tests.jsonl` with the column order below.
5. `REPLACE INTO cro.historical_test` with those rows (skipped by `--no-tests-upload` or when the MySQL secret isn't configured). List-valued columns (`component`, `product`, `kpi`, `image_path`) are JSON-encoded on the way to MySQL.
6. Replaces the slide's `detail` in `slides.jsonl` with a single-block pointer marker naming the MySQL table and `source` key, so downstream RAG doesn't re-embed the same content twice.

Table columns (order matches MySQL and `tests.jsonl`):

| column             | value                                                                                                        |
| ------------------ | ------------------------------------------------------------------------------------------------------------ |
| `source`           | `slide_id + "_" + section` — primary key                                                                     |
| `target_activity`  | reserved for later hand-tagging — always `null` in this pipeline                                             |
| `test_name`        | test's name — LLM-emitted, defaults to the slide's `sub_section`                                             |
| `concept`          | one of the approved concept labels (see `src/tests_table.py`), else `null`                                   |
| `component`        | non-empty list of page-types where the test runs, from: `Buy Page`, `Handraisers`, `Home Page`, `Home PCD`, `Landing Page`, `Multiple`, `Offer Page`, `PCD`, `PD-MMP`, `PDP`, `PF`, `PFP`. Falls back to `["Multiple"]` when unknown |
| `product`          | non-empty list of product groups being tested. Common buckets: `TV`, `DA`, `Tablet`, `Paradigm`, `Flip7/Fold7`, `B7Q7`, `Total`; specific products not in that list (e.g. `Galaxy S26`) are kept verbatim. Falls back to `["Total"]` when no product is named |
| `hypothesis`       | body under the slide's `Hypothesis` subheader, verbatim                                                      |
| `kpi`              | non-empty, de-duplicated list of every KPI the test tracks — `CVR`, `AOV`, `Engagement Rate`, `Add to Cart Rate`, `Revenue per Visitor` |
| `target_audience`  | free text (e.g. `all users`, `mobile only`)                                                                  |
| `notes`            | caveats / exclusions / watch-outs — `null` if none                                                           |
| `image_path`       | list of saved right-side image paths, relative to `<per-file-dir>` (e.g. `test_images/test_042_1.png`)       |
| `importDate`       | ISO date of the extraction run                                                                               |

MySQL credentials come from a separate AWS Secrets Manager secret:

| secret name  | required keys                                                              |
| ------------ | -------------------------------------------------------------------------- |
| `MySQL`      | `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`; optional `MYSQL_PORT` (3306), `MYSQL_DATABASE` (`cro`) |

Override with `MYSQL_SECRET_NAME`. See `secrets.example.json`. A missing or unreadable secret is not fatal — the run still produces `slides.jsonl` and `tests.jsonl`; only the MySQL upload is skipped.

### Re-chunk or rebuild the corpus without re-extracting

```bash
# Re-flatten one file's slides.jsonl into chunks.jsonl.
python -m src.chunk --file output/files/<doc_id>

# Re-flatten every file directory under a root.
python -m src.chunk --all --files-dir output/files

# Rebuild the combined corpus file from every file's chunks.jsonl.
python -m src.corpus --files-dir output/files --out output/corpus/chunks.jsonl
```

`--product` is an optional fallback used only when the model can't read it from the slide itself.

### Options

| flag           | default                            | notes                                                                                     |
| -------------- | ---------------------------------- | ----------------------------------------------------------------------------------------- |
| `--output-dir` | `output/files`                     | Root directory for per-file folders (each holds `slides.jsonl`, `chunks.jsonl`, PNGs).    |
| `--corpus-out` | `<output-dir>/../corpus/chunks.jsonl` | Path to the combined corpus file rebuilt after extract.                                  |
| `--no-chunk`   | off                                | Skip building this file's `chunks.jsonl` after extraction.                                |
| `--no-corpus`  | off                                | Skip rebuilding the combined corpus file after extraction.                                |
| `--no-tests-upload` | off                           | Skip pushing test-section rows to MySQL. `tests.jsonl` is still written and each test slide's `detail` is still replaced with the MySQL pointer marker. |
| `--product`    | `""`                               | Fallback for the `product` field when not visible on a slide.                             |
| `--dpi`        | `150`                              | Render DPI for the per-slide screenshots.                                                 |
| `--limit`      | `0` (all)                          | Only process the first N slides. Ignored if `--pages` is set.                             |
| `--pages`      | `""` (all)                         | Specific slide numbers, e.g. `42` or `10-15,42,100-105`.                                  |
| `--dry-run`    | off                                | Print records to stdout instead of writing files (skips chunk + corpus).                  |

### Try a few slides first

```bash
python -m src.extract --pdf "...pdf.zip" --pages 10-15 --dry-run
```

## Output layout

```
output/
  files/
    <doc_id>/
      slides.jsonl      # one JSON record per slide (display shape)
      chunks.jsonl      # one retrieval row per slide (embed_text + metadata)
      tests.jsonl       # one row per test-section slide (see Test-section slides)
      slide_001.png     # full-slide screenshot referenced by slide_image_path
      slide_002.png
      ...
  corpus/
    chunks.jsonl        # concatenation of every file's chunks.jsonl
    manifest.json       # per-file chunk_count, sha256, built_at
```

`slides.jsonl` stores `slide_image_path` as a basename (resolves inside the file folder). `chunks.jsonl` stores `image_path` as `files/<doc_id>/<basename>` so the corpus file is self-contained relative to the `output/` root.

## Layout

```
src/
  extract.py         # CLI entry point; dispatches PDF vs PPTX, then chunks + corpus
  chunk.py           # slides.jsonl -> chunks.jsonl (deterministic flatten)
  corpus.py          # every file's chunks.jsonl -> corpus/chunks.jsonl + manifest.json
  pdf_layout.py      # pdfminer.six text/figures + pdfplumber tables per page
  pdf_utils.py       # PDF page rendering (pypdfium2) + .pdf.zip input handling
  pptx_layout.py     # python-pptx text/tables/pictures per slide (PageLayout-shaped)
  pptx_utils.py     # pptx -> companion PDF (LibreOffice headless) + .pptx.zip input handling
  llm.py             # OpenAI text-only extraction (PDF + PPTX prompts, positioned text -> structured JSON)
  tests_table.py     # test-section slides -> historical_tests row + MySQL upload
shared/
  aws_secrets.py     # cached get_secret(name) via boto3
  settings.py        # get_settings() -> frozen Settings dataclass
secrets.example.json # template for the AWS Secrets Manager secret payload
requirements.txt
```
