"""pptx input handling: .zip resolution + one-shot LibreOffice pptx->pdf conversion.

python-pptx handles text/table/figure extraction natively, but nothing in the
Python ecosystem renders pptx slides to raster images. We shell out to
LibreOffice headless once per deck to produce a companion PDF, then reuse
pypdfium2 (via pdf_utils.render_page) for per-slide PNGs. Same visual output
as the PDF pipeline, one subprocess call per deck.
"""
import shutil
import subprocess
import zipfile
from pathlib import Path

from PIL import Image

from .pdf_utils import render_page as _render_pdf_page


def resolve_pptx_input(pptx_or_zip: Path) -> Path:
    """If given a .zip, extract the first .pptx inside next to the archive and return its path."""
    if pptx_or_zip.suffix.lower() != ".zip":
        return pptx_or_zip
    with zipfile.ZipFile(pptx_or_zip) as zf:
        members = [
            n for n in zf.namelist()
            if n.lower().endswith(".pptx") and not n.endswith("/")
        ]
        if not members:
            raise SystemExit(f"No .pptx found inside {pptx_or_zip}")
        member = members[0]
        target = pptx_or_zip.parent / Path(member).name
        if not target.exists():
            print(f"[unzip] {pptx_or_zip.name} -> {target.name}")
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
    return target


def _find_soffice() -> str:
    for candidate in ("soffice", "libreoffice"):
        found = shutil.which(candidate)
        if found:
            return found
    raise SystemExit(
        "LibreOffice not found on PATH. Install it (e.g. `apt-get install libreoffice` "
        "or `brew install --cask libreoffice`) so pptx slides can be rendered to PDF."
    )


def pptx_to_pdf(pptx_path: Path, out_dir: Path) -> Path:
    """Convert a .pptx to PDF via `soffice --headless --convert-to pdf`.

    The resulting PDF is written into `out_dir` with the same stem as the pptx.
    If it already exists and is newer than the pptx, we skip the conversion —
    re-runs against the same deck don't repay the LibreOffice startup cost.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{pptx_path.stem}.pdf"
    if pdf_path.exists() and pdf_path.stat().st_mtime >= pptx_path.stat().st_mtime:
        return pdf_path

    soffice = _find_soffice()
    print(f"[pptx->pdf] {pptx_path.name} -> {pdf_path}")
    result = subprocess.run(
        [
            soffice,
            "--headless",
            "--convert-to", "pdf",
            "--outdir", str(out_dir),
            str(pptx_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not pdf_path.exists():
        raise SystemExit(
            f"LibreOffice failed to convert {pptx_path}:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return pdf_path


def render_slide(pdf_path: Path, slide_num: int, dpi: int = 150) -> Image.Image:
    """Render one 1-indexed slide from the companion PDF to a PIL RGB image."""
    return _render_pdf_page(pdf_path, slide_num, dpi=dpi)
