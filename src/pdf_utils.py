import zipfile
from pathlib import Path

import pypdfium2 as pdfium
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


def crop_region(
    slide_png: Path,
    bbox_pct: tuple[float, float, float, float],
    out_path: Path,
    pad_pct: float = 0.01,
) -> Path:
    """Crop `slide_png` to the given fractional bbox (left, top, right, bottom) plus a small pad."""
    img = Image.open(slide_png)
    w, h = img.size
    x1, y1, x2, y2 = bbox_pct
    x1 -= pad_pct; y1 -= pad_pct; x2 += pad_pct; y2 += pad_pct
    x1 = max(0.0, min(1.0, x1)); x2 = max(0.0, min(1.0, x2))
    y1 = max(0.0, min(1.0, y1)); y2 = max(0.0, min(1.0, y2))
    if x2 <= x1 or y2 <= y1:
        img.save(out_path, format="PNG")
        return out_path
    box = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
    img.crop(box).save(out_path, format="PNG")
    return out_path


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
