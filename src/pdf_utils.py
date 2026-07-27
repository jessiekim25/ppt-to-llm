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


def page_count(pdf_path: Path) -> int:
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return len(pdf)
    finally:
        pdf.close()


def render_page(pdf_path: Path, page_num: int, dpi: int = 150) -> Image.Image:
    """Render a single 1-indexed PDF page to an in-memory PIL RGB image."""
    scale = dpi / 72
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        if page_num < 1 or page_num > len(pdf):
            raise IndexError(f"page {page_num} out of range 1..{len(pdf)}")
        return pdf[page_num - 1].render(scale=scale).to_pil().convert("RGB")
    finally:
        pdf.close()
