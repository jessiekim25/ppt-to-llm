import argparse
import json
import re
from pathlib import Path

from openai import OpenAI

from shared.settings import get_settings

from .llm import extract_slide
from .pdf_utils import (
    crop_bbox_and_save,
    render_pdf_pages,
    resolve_pdf_input,
)

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


def build_slide_record(
    extracted: dict,
    slide_png: Path,
    defaults: dict,
    doc_id: str,
) -> dict:
    """Turn one LLM extraction into a slide record matching the JSONL schema."""
    slide_num = int(slide_png.stem.rsplit("_", 1)[-1])

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

    # Image extraction: the LLM returns one entry per distinct visual region
    # on the slide with a label and a generous bbox_pct. We crop the rendered
    # slide to each bbox (with a large pad on top of the LLM's own margin) and
    # save with a header-slugged filename. Native pypdfium2 image extraction
    # doesn't work for these decks because most "images" are vector-drawn
    # (paths and shapes), not embedded rasters.
    assets_dir = slide_png.parent / slide_png.stem
    assets_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict] = []
    for idx, entry in enumerate(extracted.get("images") or [], start=1):
        if not isinstance(entry, dict):
            continue
        bbox = entry.get("bbox_pct") or [0.0, 0.0, 1.0, 1.0]
        label = str(entry.get("label") or "").strip()
        label_slug = _slug(label)
        stem = f"slide_{slide_num:03d}_img_{idx:02d}"
        filename = f"{stem}__{label_slug}.png" if label_slug else f"{stem}.png"
        out_path = assets_dir / filename
        _, cropped = crop_bbox_and_save(slide_png, bbox, out_path, pad_pct=0.10)
        if not cropped:
            print(f"  ! image {idx}: bad bbox {bbox!r}, skipping")
            continue
        image_entry: dict = {
            "idx": idx,
            "bbox_pct": [float(x) for x in bbox],
            "path": filename,
        }
        if label:
            image_entry["label"] = label
        images.append(image_entry)

    if images:
        print(f"  [images] {len(images)} LLM-directed crop(s)")
    else:
        (assets_dir / "slide.png").write_bytes(slide_png.read_bytes())
        print("  [images] no image regions returned; saved full slide.png as fallback")

    record: dict = {
        "doc_id": doc_id,
        "slide_num": slide_num,
        "slide_id": f"{doc_id}#{slide_num:03d}",
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
        help="Directory that will receive rendered slide PNGs and the per-deck slides.jsonl.",
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
        help="Print records to stdout instead of writing slides.jsonl.",
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
    image_paths = render_pdf_pages(pdf_path, per_deck_dir, dpi=args.dpi, pages=pages)
    if args.limit and not pages:
        image_paths = image_paths[: args.limit]

    client = OpenAI(api_key=settings.openai_api_key)
    defaults = {"codename": args.codename, "product": args.product}

    doc_id = _slug(pdf_path.stem)

    records: list[dict] = []
    for i, img in enumerate(image_paths, start=1):
        print(f"[extract] slide {i}/{len(image_paths)}: {img.name}")
        try:
            data = extract_slide(client, settings.openai_model, img)
        except Exception as e:  # keep going even if one slide fails
            print(f"  ! extraction failed: {e}")
            continue
        records.append(build_slide_record(data, img, defaults, doc_id=doc_id))

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
