import zipfile
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw


def resolve_pdf_input(pdf_or_zip: Path) -> Path:
    """If given a .zip, extract the first PDF inside next to the archive and return its path."""
    if pdf_or_zip.suffix.lower() != ".zip":
        return pdf_or_zip
    with zipfile.ZipFile(pdf_or_zip) as zf:
        pdf_members = [
            n for n in zf.namelist()
            if n.lower().endswith(".pdf") and not n.endswith("/")
        ]
        if not pdf_members:
            raise SystemExit(f"No PDF found inside {pdf_or_zip}")
        member = pdf_members[0]
        target = pdf_or_zip.parent / Path(member).name
        if not target.exists():
            print(f"[unzip] {pdf_or_zip.name} -> {target.name}")
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
    return target


def mask_and_crop(
    slide_png: Path,
    content_bbox: tuple[float, float, float, float] | list[float],
    text_regions: list[dict],
    out_path: Path,
    crop_pad_pct: float = 0.06,
    mask_pad_pct: float = 0.005,
    fill: str = "white",
) -> tuple[Path, bool]:
    """Paint each captured text region white, then crop to content_bbox and save.

    Returns (out_path, cropped) where `cropped` is True when a real crop was written
    and False when we saved the whole (still-masked) slide as a fallback (bbox
    missing/degenerate).
    """
    img = Image.open(slide_png).convert("RGB")
    w, h = img.size

    # 1. Whiteout every already-captured text region so it can't survive the crop.
    draw = ImageDraw.Draw(img)
    for r in text_regions or []:
        if not isinstance(r, dict):
            continue
        bbox = r.get("bbox_pct") or [0.0, 0.0, 0.0, 0.0]
        try:
            mx1, my1, mx2, my2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        except (TypeError, ValueError, IndexError):
            continue
        mx1 -= mask_pad_pct; my1 -= mask_pad_pct; mx2 += mask_pad_pct; my2 += mask_pad_pct
        mx1 = max(0.0, min(1.0, mx1)); mx2 = max(0.0, min(1.0, mx2))
        my1 = max(0.0, min(1.0, my1)); my2 = max(0.0, min(1.0, my2))
        if mx2 <= mx1 or my2 <= my1:
            continue
        draw.rectangle(
            [int(mx1 * w), int(my1 * h), int(mx2 * w), int(my2 * h)],
            fill=fill,
        )

    # 2. Crop the (now text-scrubbed) slide to the content bbox.
    try:
        x1, y1, x2, y2 = (float(content_bbox[0]), float(content_bbox[1]), float(content_bbox[2]), float(content_bbox[3]))
    except (TypeError, ValueError, IndexError):
        img.save(out_path, format="PNG")
        return out_path, False
    x1 -= crop_pad_pct; y1 -= crop_pad_pct; x2 += crop_pad_pct; y2 += crop_pad_pct
    x1 = max(0.0, min(1.0, x1)); x2 = max(0.0, min(1.0, x2))
    y1 = max(0.0, min(1.0, y1)); y2 = max(0.0, min(1.0, y2))
    if x2 <= x1 or y2 <= y1:
        img.save(out_path, format="PNG")
        return out_path, False
    box = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
    img.crop(box).save(out_path, format="PNG")
    return out_path, True


def render_pdf_pages(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 150,
    pages: set[int] | None = None,
) -> list[Path]:
    """Render PDF pages to PNGs. `pages` is a set of 1-indexed page numbers; None means all."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72
    paths: list[Path] = []
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        total = len(pdf)
        indices = sorted(pages) if pages else range(1, total + 1)
        for page_num in indices:
            if page_num < 1 or page_num > total:
                print(f"[render] skipping page {page_num} (PDF has {total} pages)")
                continue
            page = pdf[page_num - 1]
            image = page.render(scale=scale).to_pil()
            out = output_dir / f"slide_{page_num:03d}.png"
            image.save(out, format="PNG")
            paths.append(out)
    finally:
        pdf.close()
    return paths
