import argparse
import json
from pathlib import Path

from openai import OpenAI

from shared.settings import get_settings

from .db import connect, ensure_table, insert_row
from .llm import extract_slide
from .pdf_utils import render_pdf_pages

FIELDS = ("product", "codename", "section", "sub_section", "detail", "model")


def parse_pages(spec: str) -> set[int]:
    """Parse '1,3,10-15,42' into a set of 1-indexed page numbers."""
    result: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            start, end = int(a), int(b)
            if start > end:
                start, end = end, start
            result.update(range(start, end + 1))
        else:
            result.add(int(chunk))
    return result


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
        "--pages",
        default="",
        help="Specific slide numbers to process, e.g. '42' or '10-15,42,100-105'. Overrides --limit.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print extracted rows as JSONL to stdout instead of writing to MySQL.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")

    settings = get_settings()

    pages = parse_pages(args.pages) if args.pages else None

    per_deck_dir = args.output_dir / args.pdf.stem
    print(f"[render] {args.pdf} -> {per_deck_dir}")
    image_paths = render_pdf_pages(args.pdf, per_deck_dir, dpi=args.dpi, pages=pages)
    if args.limit and not pages:
        image_paths = image_paths[: args.limit]
    print(f"[render] {len(image_paths)} slide images")

    client = OpenAI(api_key=settings.openai_api_key)
    defaults = {"codename": args.codename, "product": args.product}

    rows: list[dict] = []
    for i, img in enumerate(image_paths, start=1):
        print(f"[extract] slide {i}/{len(image_paths)}: {img.name}")
        try:
            data = extract_slide(client, settings.openai_model, img)
        except Exception as e:  # keep going even if one slide fails
            print(f"  ! extraction failed: {e}")
            continue
        rows.append(build_row(data, img, defaults))

    if args.dry_run:
        for r in rows:
            print(json.dumps(r, ensure_ascii=False))
        return

    with connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_database,
    ) as conn:
        ensure_table(conn)
        for r in rows:
            insert_row(conn, r)
    print(f"[db] inserted {len(rows)} rows into jihwi.brand_guidelines")


if __name__ == "__main__":
    main()
