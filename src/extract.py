import argparse
import json
import re
from pathlib import Path

from openai import OpenAI
from PIL import Image

from shared.settings import get_settings

from .llm import extract_slide
from .pdf_layout import Figure, extract_page_layout
from .pdf_utils import crop_and_save, page_count, render_page, resolve_pdf_input

SLIDE_FIELDS = ("product", "codename", "section", "sub_section", "model")


def _slug(text: str, max_len: int = 60) -> str:
    """Filesystem-safe slug: ASCII alnum, hyphen, underscore; spaces to underscores."""
    text = re.sub(r"[^A-Za-z0-9\s\-_]", "", str(text))
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"_+", "_", text)
    return text[:max_len].strip("_-")


def _clean_subheader(sh: dict) -> dict | None:
    """Normalize an LLM subheader entry, dropping empties and recursing into children."""
    if not isinstance(sh, dict):
        return None
    title = str(sh.get("title", "") or "").strip()
    detail = str(sh.get("detail", "") or "").strip()
    tables = _clean_tables(sh.get("tables") or [])
    children_raw = sh.get("children") or []
    children = [c for c in (_clean_subheader(x) for x in children_raw) if c]
    if not (title or detail or tables or children):
        return None
    out: dict = {}
    if title:
        out["title"] = title
    if detail:
        out["detail"] = detail
    if tables:
        out["tables"] = tables
    if children:
        out["children"] = children
    return out


def _clean_tables(tables: list) -> list[dict]:
    out: list[dict] = []
    for t in tables or []:
        if not isinstance(t, dict):
            continue
        title = str(t.get("title", "") or "").strip()
        columns = [str(c).strip() for c in (t.get("columns") or [])]
        rows = [[str(c) for c in (row or [])] for row in (t.get("rows") or [])]
        if not (columns or rows):
            continue
        entry: dict = {}
        if title:
            entry["title"] = title
        if columns:
            entry["columns"] = columns
        if rows:
            entry["rows"] = rows
        out.append(entry)
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


def _save_figure_crops(
    figures: list[Figure],
    rendered: Image.Image,
    page_num: int,
    assets_dir: Path,
) -> list[dict]:
    """Crop each pdfminer-detected figure from the rendered page image; return record entries."""
    images: list[dict] = []
    for idx, fig in enumerate(figures, start=1):
        label = fig.label.strip()
        label_slug = _slug(label)
        stem = f"slide_{page_num:03d}_img_{idx:02d}"
        filename = f"{stem}__{label_slug}.png" if label_slug else f"{stem}.png"
        out_path = assets_dir / filename
        if not crop_and_save(rendered, fig.bbox_pct, out_path):
            print(f"  ! figure {idx}: bad bbox {fig.bbox_pct!r}, skipping")
            continue
        entry: dict = {
            "idx": idx,
            "bbox_pct": [float(x) for x in fig.bbox_pct],
            "path": filename,
        }
        if label:
            entry["label"] = label
        images.append(entry)
    return images


def build_slide_record(
    extracted: dict,
    page_num: int,
    figures: list[Figure],
    rendered: Image.Image,
    defaults: dict,
    doc_id: str,
    assets_dir: Path,
) -> dict:
    fields: dict = {}
    for f in SLIDE_FIELDS:
        value = str(extracted.get(f, "") or "").strip()
        if not value and defaults.get(f):
            value = defaults[f]
        if value:
            fields[f] = value

    detail = str(extracted.get("detail", "") or "").strip()
    subheaders = [c for c in (_clean_subheader(x) for x in (extracted.get("subheaders") or [])) if c]
    tables = _clean_tables(extracted.get("tables") or [])

    images = _save_figure_crops(figures, rendered, page_num, assets_dir) if figures else []
    if images:
        print(f"  [images] {len(images)} figure crop(s) from pdfminer")
    else:
        print("  [images] no figures detected on this slide")

    record: dict = {
        "doc_id": doc_id,
        "slide_num": page_num,
        "slide_id": f"{doc_id}#{page_num:03d}",
    }
    record.update(fields)
    if detail:
        record["detail"] = detail
    if subheaders:
        record["subheaders"] = subheaders
    if tables:
        record["tables"] = tables
    if images:
        record["images"] = images
    return record


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a campaign visual guideline PDF into per-slide JSON records for LLM retrieval.",
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
        help="Directory that will receive per-slide figure crops and the per-deck slides.jsonl.",
    )
    p.add_argument("--codename", default="", help="Fallback codename when not visible on a slide.")
    p.add_argument("--product", default="", help="Fallback product/series when not visible on a slide.")
    p.add_argument("--dpi", type=int, default=150, help="Render DPI for figure crops.")
    p.add_argument("--limit", type=int, default=0, help="Only process the first N slides (0 = all).")
    p.add_argument(
        "--pages",
        default="",
        help="Specific slide numbers to process, e.g. '42' or '10-15,42,100-105'. Overrides --limit.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print records to stdout instead of writing slides.jsonl.",
    )
    return p.parse_args()


def _resolve_page_nums(pdf_path: Path, pages: set[int] | None, limit: int) -> list[int]:
    total = page_count(pdf_path)
    if pages:
        return sorted(p for p in pages if 1 <= p <= total)
    nums = list(range(1, total + 1))
    return nums[:limit] if limit else nums


def main() -> None:
    args = parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"Input not found: {args.pdf}")

    pdf_path = resolve_pdf_input(args.pdf)
    settings = get_settings()

    pages = parse_pages(args.pages) if args.pages else None
    page_nums = _resolve_page_nums(pdf_path, pages, args.limit)

    per_deck_dir = args.output_dir / pdf_path.stem
    per_deck_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=settings.openai_api_key)
    defaults = {"codename": args.codename, "product": args.product}
    doc_id = _slug(pdf_path.stem)

    records: list[dict] = []
    for i, page_num in enumerate(page_nums, start=1):
        print(f"[extract] slide {i}/{len(page_nums)}: page {page_num}")
        try:
            rendered = render_page(pdf_path, page_num, dpi=args.dpi)
            layout = extract_page_layout(pdf_path, page_num)
            data = extract_slide(client, settings.openai_model, rendered)
        except Exception as e:  # keep going even if one slide fails
            print(f"  ! extraction failed: {e}")
            continue
        assets_dir = per_deck_dir / f"slide_{page_num:03d}"
        records.append(
            build_slide_record(
                data,
                page_num=page_num,
                figures=layout.figures,
                rendered=rendered,
                defaults=defaults,
                doc_id=doc_id,
                assets_dir=assets_dir,
            )
        )

    if args.dry_run:
        for r in records:
            print(json.dumps(r, ensure_ascii=False))
        return

    out_path = per_deck_dir / "slides.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
    print(f"[jsonl] wrote {len(records)} slide record(s) to {out_path}")


if __name__ == "__main__":
    main()
