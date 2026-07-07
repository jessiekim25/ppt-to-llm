import argparse
import json
from pathlib import Path

from openai import OpenAI

from shared.settings import get_settings

from .db import connect, ensure_table, insert_row
from .llm import extract_slide
from .pdf_utils import (
    crop_region,
    extract_page_images,
    extract_page_text,
    render_pdf_pages,
    resolve_pdf_input,
)

FIELDS = ("product", "codename", "section", "sub_section", "detail", "model")


def _reference_lines(reference_text: str) -> list[str]:
    """Split the PDF-native text dump into non-trivial lines for gap-filling."""
    lines: list[str] = []
    for raw in reference_text.splitlines():
        line = raw.strip()
        if len(line) < 4:
            continue
        lines.append(line)
    return lines


def _covered(line: str, haystack: str) -> bool:
    """Loose containment: exact substring match, case-insensitive, whitespace-collapsed."""
    def norm(s: str) -> str:
        return " ".join(s.split()).lower()
    return norm(line) in norm(haystack)


def _fold_missing_text(detail: str, extracted: dict, reference_text: str) -> str:
    """Append any reference-text lines the model failed to include anywhere in the output."""
    if not reference_text.strip():
        return detail
    haystack_parts = [detail]
    for field in ("sub_section", "section", "product", "codename", "model"):
        v = extracted.get(field)
        if v:
            haystack_parts.append(str(v))
    for sh in extracted.get("subheaders") or []:
        haystack_parts.append(str(sh.get("title") or ""))
        haystack_parts.append(str(sh.get("detail") or ""))
        for r in sh.get("table") or []:
            haystack_parts.append(str(r.get("format", "")))
            for fn in r.get("file_names") or []:
                haystack_parts.append(str(fn))
    for r in extracted.get("table") or []:
        haystack_parts.append(str(r.get("format", "")))
        for fn in r.get("file_names") or []:
            haystack_parts.append(str(fn))
    for p in extracted.get("panels") or []:
        haystack_parts.append(str(p.get("label") or ""))
        haystack_parts.append(str(p.get("description") or ""))
    haystack = "\n".join(haystack_parts)

    missing = [ln for ln in _reference_lines(reference_text) if not _covered(ln, haystack)]
    if not missing:
        return detail
    print(f"  [gap-fill] appending {len(missing)} missed line(s) to detail")
    tail = "\n".join(missing)
    return f"{detail}\n\n{tail}" if detail else tail


def _format_table(rows: list[dict]) -> list[str]:
    lines: list[str] = []
    for r in rows:
        fmt = str(r.get("format", "")).strip()
        file_names = r.get("file_names")
        if not file_names:
            single = r.get("file_name")
            file_names = [single] if single else []
        joined = " | ".join(str(x).strip() for x in file_names if str(x).strip())
        lines.append(f"Format {fmt}: {joined}" if joined else f"Format {fmt}:")
    return lines


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


def build_row(extracted: dict, slide_png: Path, defaults: dict, pdf_path: Path, reference_text: str = "") -> dict:
    row = {f: str(extracted.get(f, "") or "").strip() for f in FIELDS}
    for k, v in defaults.items():
        if not row.get(k) and v:
            row[k] = v

    # Per-slide assets folder (sibling of the slide PNG, sharing its stem).
    assets_dir = slide_png.parent / slide_png.stem
    assets_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    panels = extracted.get("panels") or []
    for idx, panel in enumerate(panels, start=1):
        bbox = panel.get("bbox_pct") or [0.0, 0.0, 1.0, 1.0]
        try:
            bbox_tuple = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        except (TypeError, ValueError, IndexError):
            print(f"  ! panel {idx}: bad bbox {bbox!r}, skipping")
            continue
        out = assets_dir / f"img_{idx:02d}.png"
        try:
            crop_region(slide_png, bbox_tuple, out, pad_pct=0.08)
            saved.append(out)
        except Exception as e:
            print(f"  ! panel {idx}: crop failed: {e}")

    if saved:
        print(f"  [images] {len(saved)} panel crop(s)")
    else:
        page_num = int(slide_png.stem.rsplit("_", 1)[-1])
        native = extract_page_images(pdf_path, page_num, assets_dir)
        if native:
            saved = native
            print(f"  [images] {len(saved)} extracted natively")
        else:
            (assets_dir / "slide.png").write_bytes(slide_png.read_bytes())
            print("  [images] no image regions or embedded images; saved full slide")

    parts: list[str] = []
    slide_detail = row.get("detail", "").strip()
    if slide_detail:
        parts.append(slide_detail)

    slide_table_lines = _format_table(extracted.get("table") or [])
    if slide_table_lines:
        parts.append("\n".join(slide_table_lines))

    for sh in extracted.get("subheaders") or []:
        title = str(sh.get("title", "")).strip()
        sh_detail = str(sh.get("detail", "")).strip()
        sh_table_lines = _format_table(sh.get("table") or [])
        block: list[str] = []
        if title:
            block.append(f"## {title}")
        if sh_detail:
            block.append(sh_detail)
        if sh_table_lines:
            block.append("\n".join(sh_table_lines))
        if block:
            parts.append("\n\n".join(block))

    detail = "\n\n".join(parts)
    detail = _fold_missing_text(detail, extracted, reference_text)
    row["detail"] = detail
    row["image_path"] = str(assets_dir.resolve())
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
        page_num = int(img.stem.rsplit("_", 1)[-1])
        reference_text = extract_page_text(pdf_path, page_num)
        try:
            data = extract_slide(client, settings.openai_model, img, reference_text)
        except Exception as e:  # keep going even if one slide fails
            print(f"  ! extraction failed: {e}")
            continue
        rows.append(build_row(data, img, defaults, pdf_path, reference_text))

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
