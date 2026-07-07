# ppt-to-llm

Convert a Samsung campaign visual identity PDF into structured rows in `jihwi.brand_guidelines` — one row per slide, plus per-slide image crops on disk — so an LLM (or a plain SQL query) can retrieve any slide's content instantly.

Each slide becomes one row with these fields:

| column        | meaning                                                                                     |
| ------------- | ------------------------------------------------------------------------------------------- |
| `page`        | source slide number in the PDF (e.g. `42`)                                                  |
| `product`     | phone series/line the slide belongs to (e.g. `Galaxy S`)                                    |
| `codename`    | campaign project code name (e.g. `Miracle`)                                                 |
| `section`     | top-left section label of the slide (one of the four deck sections)                         |
| `sub_section` | slide title / heading                                                                       |
| `detail`      | slide body text — subheaders rendered as `## Heading`, nested children as `###`, plus tables |
| `model`       | specific phone model shown on the slide (e.g. `Galaxy S26 Ultra`)                           |
| `image_path`  | absolute path of the per-slide assets folder containing the cropped images                  |

## How it works

1. Render each PDF page to PNG with `pypdfium2` (pure-Python, no Poppler needed).
2. Send each PNG to an OpenAI vision model (`gpt-4o` by default) with a strict JSON extraction prompt that returns structured text fields plus `images[]` (one entry per distinct visual region with a label + bbox).
3. Crop the rendered slide to each `bbox_pct` with generous padding and save it as `img_NN__<label>.png` in a per-slide assets folder.
4. Insert one row per slide into `jihwi.brand_guidelines`; `image_path` points at the assets folder.

Subheaders in `detail` are rendered hierarchically so a downstream LLM can reconstruct slide layout:

```
## 4:1 proportion

### How to build layout:
1. ...
2. ...

## 6:1 proportion

### How to build layout:
1. ...
2. ...
```

Tables land in the actual header labels the designer used (`Format | File name`, `Element | Spec`, `Logo/lock-up | Name`, ...) rather than a stamped-in schema.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

All secrets live in **AWS Secrets Manager** — nothing sensitive touches the repo or `.env`. This project reads from two existing secrets:

| secret name | required keys                                                                                          |
| ----------- | ------------------------------------------------------------------------------------------------------ |
| `MySQL`     | `RDS_HOSTNAME`, `RDS_USERNAME_TESTDB`, `RDS_PASSWORD_TESTDB`, `RDS_DB_NAME`, `RDS_PORT` (optional; 3306) |
| `LLMKeys`   | `OPENAI_API_KEY`, `OPENAI_MODEL` (optional; defaults to `gpt-4o`)                                       |

Override the secret names with the `MYSQL_SECRET_NAME` / `LLM_SECRET_NAME` env vars if needed. See `secrets.example.json` for the expected shape.

AWS credentials are picked up from the standard boto3 chain (`AWS_PROFILE`, `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, IAM role, `~/.aws/credentials`). Region comes from `AWS_REGION` or your profile. `shared/settings.py` reads these secrets via `shared/aws_secrets.get_secret()` and returns a frozen `Settings` dataclass.

Create the table once (the CLI also runs `CREATE TABLE IF NOT EXISTS`, but running the schema explicitly is nice on a fresh DB):

```bash
mysql -u root -p < schema.sql
```

If you already have the table from an earlier version, add the `page` column:

```sql
ALTER TABLE jihwi.brand_guidelines ADD COLUMN page INT AFTER id;
```

## Run against the Galaxy Miracle guideline

The source file is a zip — the CLI unpacks it automatically if you point `--pdf` at the `.zip`.

```bash
# Windows
python -m src.extract ^
  --pdf "C:\Users\yebin.kim\2026 Galaxy Miracle VIS Guidelines_v1.6_260116_compressed.pdf.zip" ^
  --codename "Miracle" ^
  --product "Galaxy S" ^
  --output-dir "C:\Users\yebin.kim\brand_guideline_images"
```

`--codename` and `--product` are optional fallbacks used only when the model can't read them from the slide itself.

### Options

| flag           | default          | notes                                                                                     |
| -------------- | ---------------- | ----------------------------------------------------------------------------------------- |
| `--pdf`        | (required)       | Path to a `.pdf` or a `.zip` containing one (extracted next to the archive on first run). |
| `--output-dir` | `output/images`  | Rendered slide PNGs and per-slide assets folders land here.                               |
| `--codename`   | `""`            | Fallback for the `codename` column when not visible on a slide.                           |
| `--product`    | `""`            | Fallback for the `product` column when not visible on a slide.                            |
| `--dpi`        | `150`            | Render DPI for slide PNGs.                                                                |
| `--limit`      | `0` (all)        | Only process the first N slides. Ignored if `--pages` is set.                             |
| `--pages`      | `""` (all)      | Specific slide numbers, e.g. `42` or `10-15,42,100-105`.                                  |
| `--dry-run`    | off              | Print extracted rows as JSONL to stdout instead of writing to MySQL.                      |

### Try a few slides first

```bash
python -m src.extract --pdf "...pdf.zip" --pages 10-15 --dry-run
```

`--dry-run` prints one JSON object per slide to stdout instead of touching MySQL — useful for eyeballing extraction quality before running the whole deck.

## Output layout

```
output/images/<deck-stem>/
  slide_001.png                 # full-page render (source for crops)
  slide_001/                    # per-slide assets folder — image_path in the DB
    img_01__<label>.png         # cropped visual regions, one per image on the slide
    img_02__<label>.png
    ...
  slide_002.png
  slide_002/
    ...
```

Each `img_NN__<label>.png` is named after the header/title text next to the visual on the slide. When a visual is marked only by a numbered gray-circle badge (common in Format tables), the badge digit becomes the label (`img_01__1.png`, `img_02__2.png`, …), so files line up 1-to-1 with the `Format N: …` rows in `detail`.

## Layout

```
src/
  extract.py         # CLI entry point
  pdf_utils.py       # render_pdf_pages, crop helpers
  llm.py             # OpenAI vision extraction
  db.py              # MySQL writer
shared/
  aws_secrets.py     # cached get_secret(name) via boto3
  settings.py        # get_settings() -> frozen Settings dataclass
schema.sql           # jihwi.brand_guidelines DDL
secrets.example.json # template for the AWS Secrets Manager secret payload
requirements.txt
```
