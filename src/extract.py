import argparse
import json
import re
from pathlib import Path

from openai import OpenAI

from shared.settings import get_settings

from .llm import build_payload, extract_slide
from .pdf_layout import extract_page_layout
from .pdf_utils import page_count, render_page, resolve_pdf_input

SLIDE_FIELDS = ("product", "codename", "section", "sub_section", "model")


def _slug(text: str, max_len: int = 60) -> str:
    """Filesystem-safe slug: ASCII alnum, hyphen, underscore; spaces to underscores."""
    text = re.sub(r"[^A-Za-z0-9\s\-_]", "", str(text))
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"_+", "_", text)
    return text[:max_len].strip("_-")


def _render_table(t: dict) -> str:
    """Render one table as 'title\\ncol1 | col2 | ...\\ncell11 | cell12 | ...' — multi-line cells joined with ' / '."""
    def cell(v: object) -> str:
        parts = [p.strip() for p in str(v).splitlines() if p.strip()]
        return " / ".join(parts)

    lines: list[str] = []
    title = str(t.get("title", "") or "").strip()
    if title:
        lines.append(title)
    columns = [str(c).strip() for c in (t.get("columns") or [])]
    if columns:
        lines.append(" | ".join(columns))
    for row in t.get("rows") or []:
        cells = [cell(c) for c in row]
        if columns and len(cells) < len(columns):
            cells += [""] * (len(columns) - len(cells))
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def _table_blocks(tables: list) -> list[dict]:
    """One {"table": "..."} block per non-empty LLM table entry."""
    out: list[dict] = []
    for t in tables or []:
        if not isinstance(t, dict):
            continue
        rendered = _render_table(t).strip()
        if rendered:
            out.append({"table": rendered})
    return out


def _block_from_subheader(sh: dict) -> dict | None:
    """Turn one LLM subheader entry into a detail block; recurse into children.

    A subheader with only a title (no body, tables, or children) collapses to a
    plain body block — never lose the text, but don't advertise a section anchor
    that has no content beneath it.
    """
    if not isinstance(sh, dict):
        return None
    title = str(sh.get("title", "") or "").strip()
    body = str(sh.get("detail", "") or "").strip()
    child_blocks = _table_blocks(sh.get("tables") or [])
    for c in sh.get("children") or []:
        child = _block_from_subheader(c)
        if child:
            child_blocks.append(child)

    if title and not body and not child_blocks:
        return {"body": title}

    block: dict = {}
    if title:
        block["subheader"] = title
    if body:
        block["body"] = body
    if child_blocks:
        block["children"] = child_blocks
    return block or None


def _compose_detail(extracted: dict) -> list[dict]:
    """Return the slide's text hierarchy as a list of blocks in reading order."""
    blocks: list[dict] = []

    slide_body = str(extracted.get("detail", "") or "").strip()
    if slide_body:
        blocks.append({"body": slide_body})

    blocks.extend(_table_blocks(extracted.get("tables") or []))

    for sh in extracted.get("subheaders") or []:
        block = _block_from_subheader(sh)
        if block:
            blocks.append(block)

    return blocks


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
    page_num: int,
    slide_image_name: str,
    defaults: dict,
    doc_id: str,
) -> dict:
    fields: dict = {}
    for f in SLIDE_FIELDS:
        value = str(extracted.get(f, "") or "").strip()
        if not value and defaults.get(f):
            value = defaults[f]
        if value:
            fields[f] = value

    detail = _compose_detail(extracted)

    record: dict = {
        "doc_id": doc_id,
        "slide_num": page_num,
        "slide_id": f"{doc_id}#{page_num:03d}",
    }
    record.update(fields)
    if detail:
        record["detail"] = detail
    record["slide_image_path"] = slide_image_name
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
        help="Directory that will receive per-slide screenshots and the per-deck slides.jsonl.",
    )
    p.add_argument("--codename", default="", help="Fallback codename when not visible on a slide.")
    p.add_argument("--product", default="", help="Fallback product/series when not visible on a slide.")
    p.add_argument("--dpi", type=int, default=150, help="Render DPI for slide screenshots.")
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


def _backfill_sections(records: list[dict]) -> None:
    """Slides without an explicit `section` inherit the last non-empty one seen.

    Section indicators sit in the top-left corner of nearly every slide in a
    deck but the LLM sometimes misses them. Sections rarely change mid-deck,
    so carrying the last seen value forward fills the gaps correctly.
    """
    last = ""
    for r in records:
        current = r.get("section", "").strip()
        if current:
            last = current
        elif last:
            r["section"] = last


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
            layout = extract_page_layout(pdf_path, page_num)
            payload = build_payload(layout, page_num)
            data = extract_slide(client, settings.openai_model, payload)
        except Exception as e:  # keep going even if one slide fails
            print(f"  ! extraction failed: {e}")
            continue

        slide_image_name = f"slide_{page_num:03d}.png"
        rendered = render_page(pdf_path, page_num, dpi=args.dpi)
        rendered.save(per_deck_dir / slide_image_name, format="PNG")

        records.append(
            build_slide_record(
                data,
                page_num=page_num,
                slide_image_name=slide_image_name,
                defaults=defaults,
                doc_id=doc_id,
            )
        )

    _backfill_sections(records)

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
