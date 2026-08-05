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


def resolve_pptx_input(pptx_or_zip: Path, pick: str = "") -> Path:
    """If given a .zip, extract the requested .pptx inside next to the archive and return its path.

    `pick` selects which .pptx to pull from a multi-file zip:
      - "" (default): only allowed when the zip contains exactly one .pptx.
        With multiple, prints the list and exits so the caller can retry with --pick.
      - a substring: matches (case-insensitive) against the archive member's
        basename. Must match exactly one file; otherwise prints candidates and exits.
    """
    if pptx_or_zip.suffix.lower() != ".zip":
        return pptx_or_zip
    with zipfile.ZipFile(pptx_or_zip) as zf:
        members = [
            n for n in zf.namelist()
            if n.lower().endswith(".pptx") and not n.endswith("/")
        ]
        if not members:
            raise SystemExit(f"No .pptx found inside {pptx_or_zip}")

        if pick:
            needle = pick.lower()
            matches = [n for n in members if needle in Path(n).name.lower()]
            if not matches:
                _die_with_listing(pptx_or_zip, members, f"no .pptx in {pptx_or_zip.name} matches --pick {pick!r}")
            if len(matches) > 1:
                _die_with_listing(pptx_or_zip, matches, f"--pick {pick!r} matches {len(matches)} files; be more specific")
            member = matches[0]
        elif len(members) > 1:
            _die_with_listing(pptx_or_zip, members, f"{pptx_or_zip.name} contains {len(members)} .pptx files; choose one with --pick")
        else:
            member = members[0]

        target = pptx_or_zip.parent / Path(member).name
        if not target.exists():
            print(f"[unzip] {pptx_or_zip.name} -> {target.name}")
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
    # Always confirm which member matched — a shell-mangled --pick can quietly
    # resolve to a wrong-but-still-unique substring; printing the choice makes
    # that mismatch obvious instead of having the pipeline run on the wrong file.
    print(f"[pptx] picked from zip: {target.name}")
    return target


def _die_with_listing(archive: Path, members: list[str], reason: str) -> None:
    print(f"[pptx] {reason}", flush=True)
    print(f"       .pptx files inside {archive.name}:")
    for m in members:
        print(f"         - {m}")
    raise SystemExit(2)


_WINDOWS_SOFFICE_CANDIDATES = (
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
)


def _find_soffice() -> str:
    for candidate in ("soffice", "libreoffice", "soffice.exe"):
        found = shutil.which(candidate)
        if found:
            return found
    # On Windows the installer doesn't add LibreOffice to PATH by default;
    # look in the two standard install locations before giving up.
    for path in _WINDOWS_SOFFICE_CANDIDATES:
        if Path(path).exists():
            return path
    raise SystemExit(
        "LibreOffice not found on PATH.\n"
        "  Windows: install from https://www.libreoffice.org/download and, if\n"
        "           the installer didn't add it to PATH, ensure soffice.exe is\n"
        "           at 'C:\\Program Files\\LibreOffice\\program\\soffice.exe'.\n"
        "  macOS:   brew install --cask libreoffice\n"
        "  Linux:   apt-get install libreoffice  (or your distro's equivalent)\n"
        "This is required so .pptx slides can be rendered to PNGs. If you only\n"
        "need the JSON output right now, pass --no-images to skip the render step."
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
