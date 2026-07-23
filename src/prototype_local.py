"""Prototype: deterministic text/table/image extraction with pdfplumber + Camelot + Tesseract.

Goal: run cheap local tools on a few slides and dump each stage's raw output side by side,
so we can eyeball how much of the current LLM-only pipeline the deterministic path already covers
before deciding what still needs an LLM.

Deterministic stages here are text-only — no font size, no font name, no heading
classification. Deciding what is title vs sub-section vs subheader stays with the LLM;
this tool just hands it the words (with positions so it can reason about layout).

Run:
    python -m src.prototype_local \\
      --pdf "...Miracle...pdf.zip" \\
      --pages 42,110,120 \\
      --output-dir output/local_prototype

Per page, writes:
    output/local_prototype/<deck-stem>/slide_042/
        page.png                  full-page render (source for overlays and OCR)
        plumber_text.txt          plain reading-order text from pdfplumber
        plumber_blocks.json       text blocks (text + bbox) clustered by proximity
        plumber_tables.json       pdfplumber's built-in table extractor
        plumber_images.json       embedded raster images pdfplumber sees
        camelot_lattice.json      Camelot lattice-flavor tables (bordered)
        camelot_stream.json       Camelot stream-flavor tables (whitespace)
        tesseract_full.txt        Tesseract OCR of the whole page (baseline)
        overlay_blocks.png        page render with text-block bboxes drawn
        overlay_tables.png        page render with detected table bboxes drawn
        report.md                 human-readable side-by-side summary + LLM-residue todo

Deliberately does not touch the DB or make any LLM calls. This is an eyeball tool.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .pdf_utils import render_pdf_pages, resolve_pdf_input

# Optional imports — surfaced as clear errors at runtime if missing.
try:
    import pdfplumber
except ImportError:  # pragma: no cover
    pdfplumber = None

try:
    import camelot
except ImportError:  # pragma: no cover
    camelot = None

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None


# ---------- pdfplumber: words -> text blocks (text + position only) ----------

@dataclass
class Block:
    """A cluster of adjacent lines. Just text and its bbox — no font info, no
    heading tags. The LLM decides what role each block plays on the slide."""

    text: str
    x0: float
    top: float
    x1: float
    bottom: float
    line_count: int = 1


def _cluster_words_into_lines(words: list[dict], y_tol: float = 3.0) -> list[list[dict]]:
    """Group words that share a baseline (within y_tol) into lines, sorted top->bottom, left->right."""
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    lines: list[list[dict]] = []
    current: list[dict] = []
    current_top: float | None = None
    for w in sorted_words:
        if current_top is None or abs(w["top"] - current_top) <= y_tol:
            current.append(w)
            current_top = w["top"] if current_top is None else current_top
        else:
            lines.append(sorted(current, key=lambda x: x["x0"]))
            current = [w]
            current_top = w["top"]
    if current:
        lines.append(sorted(current, key=lambda x: x["x0"]))
    return lines


def _lines_to_blocks(lines: list[list[dict]], gap_tol: float = 6.0) -> list[Block]:
    """Merge consecutive lines that sit close together vertically into a block.

    Grouping is on vertical proximity only — deliberately no font-based logic, so
    downstream doesn't accidentally start reasoning about typography here.
    """
    blocks: list[Block] = []
    for line in lines:
        if not line:
            continue
        text = " ".join(w["text"] for w in line).strip()
        if not text:
            continue
        top = min(w["top"] for w in line)
        bottom = max(w["bottom"] for w in line)
        x0 = min(w["x0"] for w in line)
        x1 = max(w["x1"] for w in line)

        if blocks:
            prev = blocks[-1]
            vertical_gap = top - prev.bottom
            if 0 <= vertical_gap <= gap_tol:
                prev.text = f"{prev.text}\n{text}"
                prev.bottom = bottom
                prev.x0 = min(prev.x0, x0)
                prev.x1 = max(prev.x1, x1)
                prev.line_count += 1
                continue
        blocks.append(Block(text=text, x0=x0, top=top, x1=x1, bottom=bottom))
    return blocks


def extract_plumber(pdf_path: Path, page_number: int) -> dict[str, Any]:
    if pdfplumber is None:
        return {"error": "pdfplumber not installed. `pip install pdfplumber`"}
    with pdfplumber.open(str(pdf_path)) as pdf:
        if page_number < 1 or page_number > len(pdf.pages):
            return {"error": f"page {page_number} out of range (pdf has {len(pdf.pages)})"}
        page = pdf.pages[page_number - 1]
        words = page.extract_words(keep_blank_chars=False, use_text_flow=False)
        lines = _cluster_words_into_lines(words)
        blocks = _lines_to_blocks(lines)
        return {
            "page_size": {"width": float(page.width), "height": float(page.height)},
            "text": page.extract_text() or "",
            "blocks": [asdict(b) for b in blocks],
            "tables": page.extract_tables() or [],
            "images": [
                {k: v for k, v in im.items() if k in {"x0", "y0", "x1", "y1", "width", "height", "name"}}
                for im in (page.images or [])
            ],
        }


# ---------- Camelot: lattice + stream tables ----------

def _serialize_camelot_tables(tables) -> list[dict]:
    out = []
    for t in tables:
        try:
            df = t.df
            out.append({
                "accuracy": getattr(t, "accuracy", None),
                "whitespace": getattr(t, "whitespace", None),
                "order": getattr(t, "order", None),
                "shape": list(df.shape),
                "bbox": list(getattr(t, "_bbox", []) or []),
                "columns": df.iloc[0].tolist() if df.shape[0] > 0 else [],
                "rows": df.iloc[1:].values.tolist() if df.shape[0] > 1 else [],
            })
        except Exception as e:
            out.append({"error": str(e)})
    return out


def extract_camelot(pdf_path: Path, page_number: int) -> dict[str, Any]:
    if camelot is None:
        return {"error": "camelot not installed. `pip install camelot-py[cv]` (also needs Ghostscript)."}
    result: dict[str, Any] = {}
    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(str(pdf_path), pages=str(page_number), flavor=flavor)
            result[flavor] = _serialize_camelot_tables(tables)
        except Exception as e:
            result[flavor] = {"error": f"{type(e).__name__}: {e}"}
    return result


# ---------- Tesseract: full-page baseline OCR ----------

def extract_tesseract(page_png: Path) -> dict[str, Any]:
    if pytesseract is None:
        return {"error": "pytesseract not installed. `pip install pytesseract` and install Tesseract binary."}
    try:
        img = Image.open(page_png)
        text = pytesseract.image_to_string(img)
        return {"text": text}
    except pytesseract.TesseractNotFoundError:
        return {"error": "Tesseract binary not found. Install from https://tesseract-ocr.github.io/"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ---------- overlay debug images ----------

def _pdf_to_pixel(bbox, page_w, page_h, img_w, img_h, y_flip: bool = True):
    """pdfplumber uses top-left origin already in the extracted objects, so no flip needed for it.
    Camelot / raw PDF coords have bottom-left origin, so pass y_flip=True."""
    x0, y0, x1, y1 = bbox
    sx = img_w / page_w
    sy = img_h / page_h
    if y_flip:
        return (x0 * sx, (page_h - y1) * sy, x1 * sx, (page_h - y0) * sy)
    return (x0 * sx, y0 * sy, x1 * sx, y1 * sy)


def draw_block_overlay(page_png: Path, out_png: Path, blocks: list[dict], page_size: dict) -> None:
    img = Image.open(page_png).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for b in blocks:
        px = _pdf_to_pixel(
            (b["x0"], b["top"], b["x1"], b["bottom"]),
            page_size["width"], page_size["height"], w, h, y_flip=False,
        )
        draw.rectangle(px, outline=(30, 120, 220), width=2)
    img.save(out_png)


def draw_table_overlay(page_png: Path, out_png: Path, camelot_result: dict, page_size: dict) -> None:
    img = Image.open(page_png).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    palette = {"lattice": (30, 180, 60), "stream": (200, 120, 30)}
    for flavor, tables in camelot_result.items():
        if not isinstance(tables, list):
            continue
        color = palette.get(flavor, (100, 100, 100))
        for t in tables:
            bbox = t.get("bbox") or []
            if len(bbox) != 4:
                continue
            px = _pdf_to_pixel(bbox, page_size["width"], page_size["height"], w, h, y_flip=True)
            draw.rectangle(px, outline=color, width=3)
            draw.text((px[0] + 2, px[1] + 2), f"{flavor} {t.get('accuracy', '')}", fill=color)
    img.save(out_png)


# ---------- report ----------

def render_report(page_num: int, plumber: dict, camelot_result: dict, tesseract: dict) -> str:
    lines: list[str] = [f"# Slide {page_num} — local-tools extraction\n"]

    lines.append("## pdfplumber text blocks\n")
    if plumber.get("error"):
        lines.append(f"> ERROR: {plumber['error']}\n")
    else:
        blocks = plumber.get("blocks", [])
        lines.append(f"_{len(blocks)} blocks in reading order_\n")
        for i, b in enumerate(blocks, 1):
            preview = b["text"].replace("\n", " / ")
            if len(preview) > 160:
                preview = preview[:160] + "…"
            lines.append(f"{i}. {preview}")
        lines.append("")

    lines.append("## pdfplumber tables (built-in)\n")
    tables = plumber.get("tables", []) if isinstance(plumber, dict) else []
    if not tables:
        lines.append("_none_\n")
    else:
        for i, t in enumerate(tables, 1):
            lines.append(f"### plumber-table {i} ({len(t)} rows)")
            for row in t[:8]:
                cells = [str(c or "").replace("\n", " / ")[:40] for c in row]
                lines.append("| " + " | ".join(cells) + " |")
            if len(t) > 8:
                lines.append(f"_… {len(t) - 8} more rows_")
            lines.append("")

    lines.append("## Camelot tables\n")
    for flavor in ("lattice", "stream"):
        r = camelot_result.get(flavor)
        if isinstance(r, dict) and r.get("error"):
            lines.append(f"### {flavor}: ERROR — {r['error']}\n")
            continue
        if not r:
            lines.append(f"### {flavor}: no tables\n")
            continue
        for i, t in enumerate(r, 1):
            acc = t.get("accuracy")
            lines.append(f"### camelot-{flavor} {i} (accuracy={acc}, shape={t.get('shape')})")
            cols = t.get("columns") or []
            if cols:
                lines.append("| " + " | ".join(str(c)[:40] for c in cols) + " |")
                lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
            for row in (t.get("rows") or [])[:8]:
                cells = [str(c or "").replace("\n", " / ")[:40] for c in row]
                lines.append("| " + " | ".join(cells) + " |")
            more = len(t.get("rows") or []) - 8
            if more > 0:
                lines.append(f"_… {more} more rows_")
            lines.append("")

    lines.append("## Tesseract full-page OCR (baseline)\n")
    if tesseract.get("error"):
        lines.append(f"> ERROR: {tesseract['error']}\n")
    else:
        text = tesseract.get("text", "").strip()
        lines.append("```")
        lines.append(text[:2000] + ("…" if len(text) > 2000 else ""))
        lines.append("```\n")

    lines.append("## Residue for the LLM (what still needs a model)\n")
    residue = classify_residue(plumber, camelot_result)
    for r in residue:
        lines.append(f"- {r}")
    lines.append("")

    return "\n".join(lines)


def classify_residue(plumber: dict, camelot_result: dict) -> list[str]:
    """What a downstream LLM still has to do on top of the deterministic output."""
    notes: list[str] = [
        "classify each text block by role (product / codename / section / sub_section / detail / model / subheader)",
        "group subheaders and their descriptive body text; handle nested subheaders (children)",
    ]

    blocks = plumber.get("blocks", []) if isinstance(plumber, dict) else []
    if not blocks:
        notes.append("pdfplumber found no text blocks — page is likely rasterized; rely on Tesseract for the text pass.")

    plumber_tables = plumber.get("tables", []) if isinstance(plumber, dict) else []
    cam_lat = camelot_result.get("lattice") if isinstance(camelot_result.get("lattice"), list) else []
    cam_str = camelot_result.get("stream") if isinstance(camelot_result.get("stream"), list) else []
    if not plumber_tables and not cam_lat and not cam_str:
        notes.append("no tables detected by any tool — if the slide has a Format table, its cells are drawn as shapes/badges and still need the LLM (or a vector-shape reader).")
    elif cam_lat and cam_str:
        low_acc = [t for t in cam_lat if (t.get("accuracy") or 0) < 80] + [t for t in cam_str if (t.get("accuracy") or 0) < 80]
        if low_acc:
            notes.append(f"{len(low_acc)} table(s) extracted with <80% accuracy — LLM may need to reconcile lattice vs stream.")

    images = plumber.get("images", []) if isinstance(plumber, dict) else []
    if not images:
        notes.append("no embedded raster images — visual regions are vector-drawn; keep the current render+bbox path for image crops.")

    return notes


# ---------- driver ----------

def run(
    pdf_path: Path,
    page_numbers: list[int],
    output_root: Path,
    dpi: int,
    write_db: bool = False,
    classifier_model: str = "gpt-4o-mini",
    codename: str = "",
    product: str = "",
) -> None:
    deck_dir = output_root / pdf_path.stem
    deck_dir.mkdir(parents=True, exist_ok=True)

    rendered = render_pdf_pages(pdf_path, deck_dir, dpi=dpi, pages=set(page_numbers))
    rendered_by_page = {int(p.stem.rsplit("_", 1)[-1]): p for p in rendered}

    # Lazy imports so the plain --dry-run path doesn't need OpenAI/AWS/MySQL wired up.
    db_ctx = None
    openai_client = None
    build_row = None
    classify = None
    insert_row_fn = None
    conn = None
    if write_db:
        from openai import OpenAI
        from shared.settings import get_settings
        from .db import connect, ensure_table, insert_row
        from .extract import build_row as _build_row
        from .llm_classify import classify_from_deterministic

        settings = get_settings()
        openai_client = OpenAI(api_key=settings.openai_api_key)
        build_row = _build_row
        classify = classify_from_deterministic
        insert_row_fn = insert_row
        db_ctx = connect(
            host=settings.mysql_host,
            port=settings.mysql_port,
            user=settings.mysql_user,
            password=settings.mysql_password,
            database=settings.mysql_database,
        )
        conn = db_ctx.__enter__()
        ensure_table(conn)
        defaults = {"codename": codename, "product": product}
        rows_written = 0

    for page_num in page_numbers:
        page_png = rendered_by_page.get(page_num)
        if page_png is None:
            print(f"[skip] no render for page {page_num}")
            continue
        slide_dir = deck_dir / f"slide_{page_num:03d}"
        slide_dir.mkdir(parents=True, exist_ok=True)

        (slide_dir / "page.png").write_bytes(page_png.read_bytes())

        print(f"[slide {page_num}] pdfplumber...", flush=True)
        plumber = extract_plumber(pdf_path, page_num)
        (slide_dir / "plumber_text.txt").write_text(plumber.get("text", "") or "", encoding="utf-8")
        (slide_dir / "plumber_blocks.json").write_text(json.dumps(plumber.get("blocks", []), ensure_ascii=False, indent=2))
        (slide_dir / "plumber_tables.json").write_text(json.dumps(plumber.get("tables", []), ensure_ascii=False, indent=2))
        (slide_dir / "plumber_images.json").write_text(json.dumps(plumber.get("images", []), ensure_ascii=False, indent=2))

        print(f"[slide {page_num}] camelot...", flush=True)
        camelot_result = extract_camelot(pdf_path, page_num)
        (slide_dir / "camelot_lattice.json").write_text(
            json.dumps(camelot_result.get("lattice"), ensure_ascii=False, indent=2, default=str)
        )
        (slide_dir / "camelot_stream.json").write_text(
            json.dumps(camelot_result.get("stream"), ensure_ascii=False, indent=2, default=str)
        )

        print(f"[slide {page_num}] tesseract...", flush=True)
        tess = extract_tesseract(slide_dir / "page.png")
        (slide_dir / "tesseract_full.txt").write_text(tess.get("text", "") or f"[error] {tess.get('error', '')}")

        if isinstance(plumber, dict) and plumber.get("page_size"):
            try:
                draw_block_overlay(slide_dir / "page.png", slide_dir / "overlay_blocks.png",
                                   plumber.get("blocks", []), plumber["page_size"])
                draw_table_overlay(slide_dir / "page.png", slide_dir / "overlay_tables.png",
                                   camelot_result, plumber["page_size"])
            except Exception as e:
                print(f"  ! overlay failed: {e}")

        report = render_report(page_num, plumber, camelot_result, tess)
        (slide_dir / "report.md").write_text(report, encoding="utf-8")
        print(f"[slide {page_num}] wrote {slide_dir}")

        if write_db:
            try:
                classified = classify(openai_client, classifier_model, plumber, camelot_result)
            except Exception as e:
                print(f"  ! classifier failed for slide {page_num}: {e}")
                continue
            (slide_dir / "classifier_output.json").write_text(
                json.dumps(classified, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            row = build_row(classified, page_png, defaults, pdf_path, slide_dpi=dpi)
            insert_row_fn(conn, row)
            rows_written += 1
            print(f"  [db] inserted brand_guidelines row for page {page_num}")

    if write_db and db_ctx is not None:
        db_ctx.__exit__(None, None, None)
        print(f"[db] inserted {rows_written} row(s) into brand_guidelines")


def parse_pages(spec: str) -> list[int]:
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
    return sorted(result)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pdf", required=True, type=Path, help="Path to the guideline PDF or .zip.")
    p.add_argument("--pages", required=True, type=str, help="Pages to prototype, e.g. '42,110,120-122'.")
    p.add_argument("--output-dir", type=Path, default=Path("output/local_prototype"))
    p.add_argument("--dpi", type=int, default=200, help="Render DPI (higher helps Tesseract).")
    p.add_argument(
        "--write-db",
        action="store_true",
        help="After the deterministic dump, call a cheap LLM to classify roles and insert one row per slide into brand_guidelines.",
    )
    p.add_argument(
        "--classifier-model",
        default="gpt-4o-mini",
        help="OpenAI model for the text-only classification pass (default gpt-4o-mini).",
    )
    p.add_argument("--codename", default="", help="Fallback codename when not visible on a slide.")
    p.add_argument("--product", default="", help="Fallback product/series when not visible on a slide.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.pdf.exists():
        raise SystemExit(f"Input not found: {args.pdf}")
    pdf_path = resolve_pdf_input(args.pdf)
    pages = parse_pages(args.pages)
    if not pages:
        raise SystemExit("no pages parsed from --pages")
    run(
        pdf_path,
        pages,
        args.output_dir,
        args.dpi,
        write_db=args.write_db,
        classifier_model=args.classifier_model,
        codename=args.codename,
        product=args.product,
    )


if __name__ == "__main__":
    main()
