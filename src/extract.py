import argparse
import json
from pathlib import Path

from openai import OpenAI

from shared.settings import get_settings

from .db import connect, ensure_table, insert_row
from .llm import extract_slide
from .pdf_utils import mask_regions_and_save, render_pdf_pages, resolve_pdf_input

FIELDS = ("product", "codename", "section", "sub_section", "detail", "model")


def _render_subheader(sh: dict, depth: int = 2) -> str:
    """Render one subheader (and its nested children) as a markdown-ish block.

    depth=2 -> "## title", depth=3 -> "### title", and so on. Nested children get
    depth+1 so downstream readers can rebuild the slide's section hierarchy.
    """
    title = str(sh.get("title", "")).strip()
    sh_detail = str(sh.get("detail", "")).strip()
    sh_tables = _format_tables(sh.get("tables") or [])
    children = sh.get("children") or []

    block: list[str] = []
    if title:
        prefix = "#" * max(2, min(depth, 6))
        block.append(f"{prefix} {title}")
    if sh_detail:
        block.append(sh_detail)
    if sh_tables:
        block.append("\n".join(sh_tables))
    for child in children:
        child_block = _render_subheader(child, depth=depth + 1)
        if child_block:
            block.append(child_block)
    return "\n\n".join(block)


def _format_tables(tables: list[dict]) -> list[str]:
    """Render each table as its actual "col1 | col2 | ..." header row plus cell rows.
    Multi-line cells are collapsed to " / " so each row stays on one line."""
    def cell(v: object) -> str:
        parts = [p.strip() for p in str(v).splitlines() if p.strip()]
        return " / ".join(parts)

    out: list[str] = []
    for i, t in enumerate(tables or []):
        if not isinstance(t, dict):
            continue
        if i > 0:
            out.append("")
        title = str(t.get("title", "")).strip()
        if title:
            out.append(title)
        columns = [str(c).strip() for c in (t.get("columns") or [])]
        if columns:
            out.append(" | ".join(columns))
        for row in t.get("rows") or []:
            cells = [cell(c) for c in row]
            if columns and len(cells) < len(columns):
                cells += [""] * (len(columns) - len(cells))
            out.append(" | ".join(cells))
    return out


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


def build_row(extracted: dict, slide_png: Path, defaults: dict, pdf_path: Path) -> dict:
    row = {f: str(extracted.get(f, "") or "").strip() for f in FIELDS}
    for k, v in defaults.items():
        if not row.get(k) and v:
            row[k] = v

    # One "content" image per slide: the raster slide render with every
    # already-captured text region painted white, so the file carries only
    # the graphics plus any uncaptured on-image annotations.
    text_regions = extracted.get("text_regions") or []
    content_png = slide_png.with_name(f"{slide_png.stem}_content.png")
    mask_regions_and_save(slide_png, text_regions, content_png)
    print(f"  [image] masked {len(text_regions)} text region(s) -> {content_png.name}")

    parts: list[str] = []
    slide_detail = row.get("detail", "").strip()
    if slide_detail:
        parts.append(slide_detail)

    slide_table_lines = _format_tables(extracted.get("tables") or [])
    if slide_table_lines:
        parts.append("\n".join(slide_table_lines))

    for sh in extracted.get("subheaders") or []:
        rendered = _render_subheader(sh, depth=2)
        if rendered:
            parts.append(rendered)

    row["detail"] = "\n\n".join(parts)
    row["image_path"] = str(content_png.resolve())
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a campaign visual guideline PDF into rows in jihwi.brand_guidelines.",
    )
    p.add_argument(
        "--pdf",
        required=True,
        type=Path,
        help="Path to the guideline PDF, or to a .zip containing one (extracted automatically).",
    )
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
        raise SystemExit(f"Input not found: {args.pdf}")

    pdf_path = resolve_pdf_input(args.pdf)

    settings = get_settings()

    pages = parse_pages(args.pages) if args.pages else None

    per_deck_dir = args.output_dir / pdf_path.stem
    print(f"[render] {pdf_path} -> {per_deck_dir}")
    image_paths = render_pdf_pages(pdf_path, per_deck_dir, dpi=args.dpi, pages=pages)
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
        rows.append(build_row(data, img, defaults, pdf_path))

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
