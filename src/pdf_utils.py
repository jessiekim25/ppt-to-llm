import zipfile
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
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


def extract_content_image(
    pdf_path: Path,
    page_number: int,
    slide_png: Path,
    out_path: Path,
    dpi: int = 150,
    text_pad_px: int = 3,
    crop_pad_pct: float = 0.02,
    threshold: int = 240,
) -> tuple[Path, bool, int]:
    """Save the "visual content" region of a slide as a single PNG.

    Method (deterministic, no LLM guessing):
      1. Enumerate every text object on the PDF page via pypdfium2, get its
         precise bounding box in PDF points, convert to pixel coordinates in
         the rendered slide, and paint that rectangle white on `slide_png`.
      2. Threshold the result (any pixel darker than `threshold` counts as
         content) and find the tight bounding box of the remaining pixels.
      3. Pad by `crop_pad_pct` and crop.

    Text baked into a raster image (e.g. words on top of a photo) is NOT a
    text object in the PDF, so it survives step 1 — that's exactly what we
    want. Captured slide chrome (title, section, subheaders, tables) IS
    text-object text, so it gets scrubbed.

    Returns (out_path, cropped, masked_count) where `cropped` is True if a
    tight crop was produced, False if the slide was essentially blank after
    masking (fallback: save the masked whole slide).
    """
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        img = Image.open(slide_png).convert("RGB")
        w, h = img.size
        if page_number < 1 or page_number > len(pdf):
            img.save(out_path, format="PNG")
            return out_path, False, 0

        page = pdf[page_number - 1]
        _, page_height = page.get_size()
        scale = dpi / 72

        draw = ImageDraw.Draw(img)
        masked = 0
        for obj in page.get_objects():
            if obj.type != pdfium_c.FPDF_PAGEOBJ_TEXT:
                continue
            try:
                left, bottom, right, top = obj.get_pos()
            except Exception:
                continue
            ix1 = int(left * scale) - text_pad_px
            iy1 = int((page_height - top) * scale) - text_pad_px
            ix2 = int(right * scale) + text_pad_px
            iy2 = int((page_height - bottom) * scale) + text_pad_px
            ix1 = max(0, ix1); iy1 = max(0, iy1)
            ix2 = min(w, ix2); iy2 = min(h, iy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            draw.rectangle([ix1, iy1, ix2, iy2], fill="white")
            masked += 1

        content_mask = img.convert("L").point(lambda v: 255 if v < threshold else 0)
        bbox = content_mask.getbbox()
        if not bbox:
            img.save(out_path, format="PNG")
            return out_path, False, masked

        pad = int(min(w, h) * crop_pad_pct)
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
        img.crop((x1, y1, x2, y2)).save(out_path, format="PNG")
        return out_path, True, masked
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
