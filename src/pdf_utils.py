import zipfile
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image


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


def extract_content_image(
    pdf_path: Path,
    page_number: int,
    slide_png: Path,
    out_path: Path,
    dpi: int = 150,
    crop_pad_pct: float = 0.03,
    background_area_pct: float = 0.9,
) -> tuple[Path, bool, int]:
    """Save the "visual content" region of a slide as a single PNG.

    Method (deterministic, additive, no LLM guessing):
      1. Enumerate every NON-text page object on the PDF page (images, paths,
         shapes, shadings, form-XObjects) via pypdfium2. Text objects are
         skipped entirely, so slide chrome (title, section, subheaders,
         tables) is not part of the content region.
      2. Skip whole-slide background rectangles — any single object whose
         bbox covers more than `background_area_pct` of both page dimensions
         (e.g. a full-bleed white background rect).
      3. Union all remaining bboxes -> the content region.
      4. Pad by `crop_pad_pct` and crop the rendered slide to that region.

    Returns (out_path, cropped, object_count) where `cropped` is True if a
    real content region was found and cropped, False when the slide had no
    non-text objects (e.g. text-only slide) and the whole slide is saved as
    fallback.
    """
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        img = Image.open(slide_png).convert("RGB")
        w, h = img.size
        if page_number < 1 or page_number > len(pdf):
            img.save(out_path, format="PNG")
            return out_path, False, 0

        page = pdf[page_number - 1]
        page_width, page_height = page.get_size()
        scale = dpi / 72

        boxes: list[tuple[int, int, int, int]] = []
        for obj in page.get_objects(max_depth=20):
            if obj.type == pdfium_c.FPDF_PAGEOBJ_TEXT:
                continue
            try:
                left, bottom, right, top = obj.get_pos()
            except Exception:
                continue
            if right - left <= 0 or top - bottom <= 0:
                continue
            width_pct = (right - left) / page_width
            height_pct = (top - bottom) / page_height
            if width_pct >= background_area_pct and height_pct >= background_area_pct:
                continue  # full-slide background shape
            ix1 = max(0, int(left * scale))
            iy1 = max(0, int((page_height - top) * scale))
            ix2 = min(w, int(right * scale))
            iy2 = min(h, int((page_height - bottom) * scale))
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            boxes.append((ix1, iy1, ix2, iy2))

        if not boxes:
            img.save(out_path, format="PNG")
            return out_path, False, 0

        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)

        pad = int(min(w, h) * crop_pad_pct)
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
        img.crop((x1, y1, x2, y2)).save(out_path, format="PNG")
        return out_path, True, len(boxes)
    finally:
        pdf.close()


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
