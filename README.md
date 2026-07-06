# ppt-to-llm

Convert a Samsung campaign visual identity PDF into structured rows in `jihwi.brand_guidelines` so an LLM (or a plain SQL query) can retrieve any slide's content instantly.

Each slide becomes one row with these fields:

| column        | meaning                                                                 |
| ------------- | ----------------------------------------------------------------------- |
| `product`     | phone series/line the slide belongs to (e.g. `Galaxy S`)                |
| `codename`    | campaign project code name (e.g. `Miracle`)                             |
| `section`     | top-left section label of the slide (one of the four deck sections)     |
| `sub_section` | slide title / heading                                                   |
| `detail`      | main body text of the slide                                             |
| `model`       | specific phone model shown on the slide (e.g. `Galaxy S26 Ultra`)       |
| `image_path`  | absolute path of the rendered slide PNG on disk                         |

## How it works

1. Render each PDF page to PNG with `pypdfium2` (pure-Python, no Poppler needed).
2. Send each PNG to an OpenAI vision model (`gpt-4o` by default) with a strict JSON extraction prompt.
3. Insert the resulting row into `jihwi.brand_guidelines`. The rendered PNG stays on disk; `image_path` points at it.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

All secrets live in **AWS Secrets Manager** — nothing sensitive touches the repo or `.env`. This project reads from two existing secrets:

| secret name | required keys                                                                       |
| ----------- | ----------------------------------------------------------------------------------- |
| `MySQL`     | `RDS_HOSTNAME`, `RDS_USERNAME_TESTDB`, `RDS_PASSWORD_TESTDB`, `RDS_DB_NAME`, `RDS_PORT` (optional, defaults to 3306) |
| `LLMKeys`   | `OPENAI_API_KEY`, `OPENAI_MODEL` (optional, defaults to `gpt-4o`)                    |

Override the secret names with the `MYSQL_SECRET_NAME` / `LLM_SECRET_NAME` env vars if needed. See `secrets.example.json` for the expected shape.

AWS credentials are picked up from the standard boto3 chain (`AWS_PROFILE`, `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, IAM role, `~/.aws/credentials`). Region comes from `AWS_REGION` or your profile. `shared/settings.py` reads these secrets via `shared/aws_secrets.get_secret()` and returns a frozen `Settings` dataclass — mirrors the pattern from your other project.

Create the table once (the CLI will also `CREATE TABLE IF NOT EXISTS`, but running the schema explicitly is nice on a fresh DB):

```bash
mysql -u root -p < schema.sql
```

## Run against the Galaxy Miracle guideline

The source file is zipped — unzip it first, then point the CLI at the PDF.

```bash
# On Windows, after unzipping "2026 Galaxy Miracle VIS Guidelines_v1.6_260116_compressed.pdf.zip":
python -m src.extract ^
  --pdf "C:\Users\yebin.kim\2026 Galaxy Miracle VIS Guidelines_v1.6_260116_compressed.pdf" ^
  --codename "Miracle" ^
  --product "Galaxy S" ^
  --output-dir "C:\Users\yebin.kim\brand_guideline_images"
```

`--codename` and `--product` are optional fallbacks used only when the model can't read them from the slide itself.

### Try a few slides first

```bash
python -m src.extract --pdf path\to\deck.pdf --limit 5 --dry-run
```

`--dry-run` prints one JSON object per slide to stdout instead of touching MySQL — useful for eyeballing extraction quality before you write 117 rows.

## Layout

```
src/
  extract.py         # CLI entry point
  pdf_utils.py       # render_pdf_pages()
  llm.py             # OpenAI vision extraction
  db.py              # MySQL writer
shared/
  aws_secrets.py     # lazy dict-like proxy over the AWS secret
  settings.py        # get_settings() -> frozen Settings dataclass
schema.sql           # jihwi.brand_guidelines DDL
secrets.example.json # template for the AWS Secrets Manager secret payload
requirements.txt
```
