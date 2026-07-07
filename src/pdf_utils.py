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


def crop_bbox_and_save(
    slide_png: Path,
    bbox_pct: tuple[float, float, float, float] | list[float],
    out_path: Path,
    pad_pct: float = 0.10,
) -> tuple[Path, bool]:
    """Crop `slide_png` to `bbox_pct` (fractions 0-1, left/top/right/bottom) with a large pad."""
    img = Image.open(slide_png).convert("RGB")
    w, h = img.size
    try:
        x1, y1, x2, y2 = (float(bbox_pct[0]), float(bbox_pct[1]), float(bbox_pct[2]), float(bbox_pct[3]))
    except (TypeError, ValueError, IndexError):
        return out_path, False
    x1 -= pad_pct; y1 -= pad_pct; x2 += pad_pct; y2 += pad_pct
    x1 = max(0.0, min(1.0, x1)); x2 = max(0.0, min(1.0, x2))
    y1 = max(0.0, min(1.0, y1)); y2 = max(0.0, min(1.0, y2))
    if x2 <= x1 or y2 <= y1:
        return out_path, False
    box = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
    img.crop(box).save(out_path, format="PNG")
    return out_path, True


def _bboxes_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    """Two PDF-coord bboxes (left, bottom, right, top) — do they overlap?"""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: list[int], x: int, y: int) -> None:
    rx, ry = _find(parent, x), _find(parent, y)
    if rx != ry:
        parent[rx] = ry


def extract_page_images_clustered(
    pdf_path: Path,
    page_number: int,
    slide_png: Path,
    out_dir: Path,
    dpi: int = 150,
    min_pixels: int = 40 * 40,
    filename_for: "callable[[int, tuple[int,int,int,int]], str] | None" = None,
) -> list[tuple[Path, tuple[int, int, int, int]]]:
    """Extract visual images from a page, clustering overlapping ones together.

    Method:
      1. Enumerate every FPDF_PAGEOBJ_IMAGE on the page via pypdfium2.
      2. Group images whose bounding boxes overlap into one cluster (any set
         of images layered on top of each other becomes a single output).
      3. For each cluster, take the union bbox and crop the RENDERED slide
         (from `slide_png`) to that region — so layered / composited scenes
         come out looking exactly as they appear on the slide.
      4. Sort clusters top-to-bottom then left-to-right and save as PNGs.

    filename_for(index, pixel_bbox) returns the filename to use for cluster
    N (1-indexed). If None, defaults to "img_NN.png".

    Returns [(path, pixel_bbox), ...] in reading order.
    """
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        if page_number < 1 or page_number > len(pdf):
            return []
        page = pdf[page_number - 1]
        _, page_height = page.get_size()
        scale = dpi / 72

        rendered = Image.open(slide_png).convert("RGB")
        w, h = rendered.size

        pdf_bboxes: list[tuple[float, float, float, float]] = []
        for obj in page.get_objects(max_depth=20):
            if obj.type != pdfium_c.FPDF_PAGEOBJ_IMAGE:
                continue
            try:
                left, bottom, right, top = obj.get_pos()
            except Exception:
                continue
            if right - left <= 0 or top - bottom <= 0:
                continue
            pixel_area = (right - left) * (top - bottom) * scale * scale
            if pixel_area < min_pixels:
                continue
            pdf_bboxes.append((float(left), float(bottom), float(right), float(top)))

        if not pdf_bboxes:
            return []

        n = len(pdf_bboxes)
        parent = list(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if _bboxes_overlap(pdf_bboxes[i], pdf_bboxes[j]):
                    _union(parent, i, j)

        clusters: dict[int, list[int]] = {}
        for i in range(n):
            clusters.setdefault(_find(parent, i), []).append(i)

        cluster_bboxes: list[tuple[float, float, float, float]] = []
        for members in clusters.values():
            lefts = [pdf_bboxes[i][0] for i in members]
            bottoms = [pdf_bboxes[i][1] for i in members]
            rights = [pdf_bboxes[i][2] for i in members]
            tops = [pdf_bboxes[i][3] for i in members]
            cluster_bboxes.append((min(lefts), min(bottoms), max(rights), max(tops)))

        # Reading order: top of page first (high PDF y), then left-to-right.
        cluster_bboxes.sort(key=lambda b: (-b[3], b[0]))

        out_dir.mkdir(parents=True, exist_ok=True)
        saved: list[tuple[Path, tuple[int, int, int, int]]] = []
        for i, (left, bottom, right, top) in enumerate(cluster_bboxes, start=1):
            ix1 = max(0, int(left * scale))
            iy1 = max(0, int((page_height - top) * scale))
            ix2 = min(w, int(right * scale))
            iy2 = min(h, int((page_height - bottom) * scale))
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            pixel_bbox = (ix1, iy1, ix2, iy2)
            filename = (filename_for or (lambda idx, _b: f"img_{idx:02d}.png"))(i, pixel_bbox)
            out = out_dir / filename
            rendered.crop(pixel_bbox).save(out, format="PNG")
            saved.append((out, pixel_bbox))
        return saved
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
