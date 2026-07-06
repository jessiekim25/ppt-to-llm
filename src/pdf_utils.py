from pathlib import Path

import pypdfium2 as pdfium


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
