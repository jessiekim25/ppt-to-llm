from pathlib import Path

import pypdfium2 as pdfium


def render_pdf_pages(pdf_path: Path, output_dir: Path, dpi: int = 150) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72
    paths: list[Path] = []
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            image = page.render(scale=scale).to_pil()
            out = output_dir / f"slide_{i + 1:03d}.png"
            image.save(out, format="PNG")
            paths.append(out)
    finally:
        pdf.close()
    return paths
