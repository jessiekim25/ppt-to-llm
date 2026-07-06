import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .db import connect, ensure_table, insert_row
from .llm import extract_slide
from .pdf_utils import render_pdf_pages

FIELDS = ("product", "codename", "section", "sub_section", "detail", "model")


def build_row(extracted: dict, image_path: Path, defaults: dict) -> dict:
    row = {f: str(extracted.get(f, "") or "").strip() for f in FIELDS}
    for k, v in defaults.items():
        if not row.get(k) and v:
            row[k] = v
    row["image_path"] = str(image_path.resolve())
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a campaign visual guideline PDF into rows in jihwi.brand_guidelines.",
    )
    p.add_argument("--pdf", required=True, type=Path, help="Path to the guideline PDF.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/images"),
        help="Directory that will receive rendered slide PNGs (a per-deck subfolder is created).",
    )
    p.add_argument("--codename", default="", help="Fallback codename when not visible on a slide.")
    p.add_argument("--product", default="", help="Fallback product/series when not visible on a slide.")
    p.add_argument("--dpi", type=int, default=150, help="Render DPI for slide PNGs.")
    p.add_argument("--limit", type=int, default=0, help="Only process the first N slides (0 = all).")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print extracted rows as JSONL to stdout instead of writing to MySQL.",
    )
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")

    per_deck_dir = args.output_dir / args.pdf.stem
    print(f"[render] {args.pdf} -> {per_deck_dir}")
    image_paths = render_pdf_pages(args.pdf, per_deck_dir, dpi=args.dpi)
    if args.limit:
        image_paths = image_paths[: args.limit]
    print(f"[render] {len(image_paths)} slide images")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    defaults = {"codename": args.codename, "product": args.product}

    rows: list[dict] = []
    for i, img in enumerate(image_paths, start=1):
        print(f"[extract] slide {i}/{len(image_paths)}: {img.name}")
        try:
            data = extract_slide(client, model, img)
        except Exception as e:  # keep going even if one slide fails
            print(f"  ! extraction failed: {e}")
            continue
        rows.append(build_row(data, img, defaults))

    if args.dry_run:
        for r in rows:
            print(json.dumps(r, ensure_ascii=False))
        return

    with connect(
        host=os.environ.get("MYSQL_HOST", "localhost"),
        port=int(os.environ.get("MYSQL_PORT", "3306")),
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
        database=os.environ.get("MYSQL_DATABASE", "jihwi"),
    ) as conn:
        ensure_table(conn)
        for r in rows:
            insert_row(conn, r)
    print(f"[db] inserted {len(rows)} rows into jihwi.brand_guidelines")


if __name__ == "__main__":
    main()
