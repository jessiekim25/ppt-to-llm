"""python-pptx-based per-slide layout extraction: text lines + tables + figures.

Mirrors the shape of `src/pdf_layout.py` so extract.py can operate on either
format. Instead of pdfminer.six + pdfplumber, we walk `python-pptx` shape
trees directly — pptx already exposes positioned text runs, tables, and
picture bboxes without any geometry reconstruction.
"""
from dataclasses import dataclass, field
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Emu

from .pdf_layout import Figure, PageLayout, Table, TextLine, _extract_tables as _pdfplumber_tables


def _emu_to_pct(v: int | None, total: int) -> float:
    if v is None or total <= 0:
        return 0.0
    return float(v) / float(total)


def _shape_bbox_pct(shape, slide_w: int, slide_h: int) -> tuple[float, float, float, float]:
    left = shape.left if shape.left is not None else 0
    top = shape.top if shape.top is not None else 0
    width = shape.width if shape.width is not None else 0
    height = shape.height if shape.height is not None else 0
    x0 = _emu_to_pct(left, slide_w)
    y0 = _emu_to_pct(top, slide_h)
    x1 = _emu_to_pct(left + width, slide_w)
    y1 = _emu_to_pct(top + height, slide_h)
    return (x0, y0, x1, y1)


_BOLD_FONT_KEYWORDS = ("bold", "black", "heavy", "extrabold", "semibold", "demibold")


def _looks_bold_by_name(font_name: str) -> bool:
    n = (font_name or "").lower()
    return any(kw in n for kw in _BOLD_FONT_KEYWORDS)


def _resolve_bold(flag, font_name: str, para_flag=None, para_font: str = "") -> bool:
    """Best-effort bold resolution.

    python-pptx returns run.font.bold as True/False/None where None means
    "inherit from the placeholder/master". We can't fully resolve inherited
    styles without walking the theme, but two cheap fallbacks catch most
    real-world cases: (1) the paragraph's own font.bold, (2) the font name
    itself often carries the weight ("Roboto Bold", "Arial-BoldMT",
    "SamsungOneUI Semibold").
    """
    if flag is True:
        return True
    if flag is False:
        return False
    if para_flag is True:
        return True
    if _looks_bold_by_name(font_name):
        return True
    if _looks_bold_by_name(para_font):
        return True
    return False


def _paragraph_defaults(para) -> tuple[float | None, str, bool | None]:
    """Return the paragraph-level (size_pt, font_name, bold_flag) defaults.

    bold_flag is a tri-state (True/False/None) so callers can distinguish
    "explicit False" from "inherit" when resolving run bold.
    """
    para_font_name = para.font.name or ""
    para_bold_flag = para.font.bold  # True / False / None
    para_size = None
    try:
        if para.font.size is not None:
            para_size = float(para.font.size.pt)
    except Exception:
        para_size = None
    return para_size, para_font_name, para_bold_flag


def _paragraph_segments(para) -> list[tuple[str, float | None, str, bool]]:
    """Split a paragraph into (text, size, font, bold) segments, one per
    contiguous run of the same bold state.

    A pptx paragraph like "**KEY INITIATIVES**: SIA rework" (only the first
    part bold) becomes TWO segments so downstream can turn the bold prefix
    into a subheader and the non-bold tail into that subheader's body — the
    previous first-run-wins style detection collapsed the whole paragraph
    into one bold-or-not line and lost that split.
    """
    para_size, para_font_name, para_bold_flag = _paragraph_defaults(para)

    segments: list[tuple[str, float | None, str, bool]] = []
    cur_text_parts: list[str] = []
    cur_size: float | None = None
    cur_font: str = ""
    cur_bold: bool | None = None  # None until we see the first run

    def flush():
        if not cur_text_parts:
            return
        text = " ".join("".join(cur_text_parts).split())
        if text:
            segments.append((text, cur_size, cur_font, bool(cur_bold)))

    for run in para.runs:
        text = run.text or ""
        if not text.strip():
            # Whitespace-only run: attach to whatever segment is being built
            # so we don't accidentally split "KEY INITIATIVES" from ":" when
            # the colon happens to be its own space-only run.
            if cur_bold is not None:
                cur_text_parts.append(text)
            continue

        run_size = None
        try:
            if run.font.size is not None:
                run_size = float(run.font.size.pt)
        except Exception:
            run_size = None
        if run_size is None:
            run_size = para_size

        run_font = run.font.name or para_font_name
        run_bold = _resolve_bold(run.font.bold, run_font, para_bold_flag, para_font_name)

        if cur_bold is None:
            cur_bold = run_bold
            cur_size = run_size
            cur_font = run_font
            cur_text_parts.append(text)
        elif run_bold == cur_bold:
            cur_text_parts.append(text)
        else:
            flush()
            cur_text_parts = [text]
            cur_bold = run_bold
            cur_size = run_size
            cur_font = run_font

    flush()

    if not segments:
        # No runs at all — fall back to paragraph.text with paragraph-level style.
        text = " ".join((para.text or "").split())
        if text:
            segments.append((text, para_size, para_font_name,
                             _resolve_bold(para_bold_flag, para_font_name)))
    return segments


def _text_frame_lines(
    shape,
    slide_w: int,
    slide_h: int,
    text_lines: list[TextLine],
    group_id: int,
) -> None:
    """Emit one TextLine per bold/non-bold segment in the text frame's paragraphs.

    All segments in this frame share the same `group_id`. Per-paragraph bboxes
    are approximated by slicing the shape bbox vertically; segments within one
    paragraph share that paragraph's slice (real x-offsets within a run aren't
    exposed by python-pptx).

    Defensive dedup: within one frame we skip a segment that exactly matches
    the previous segment's (text, bold) — pptx placeholders sometimes echo
    the same run into a slide via layout inheritance.
    """
    tf = shape.text_frame
    paras = list(tf.paragraphs)
    non_empty = [p for p in paras if (p.text or "").strip()]
    if not non_empty:
        return

    x0, y0, x1, y1 = _shape_bbox_pct(shape, slide_w, slide_h)
    height = max(y1 - y0, 0.0)
    n = len(non_empty)
    slice_h = height / n if n else 0.0

    last_seen: tuple[str, bool] | None = None
    for i, para in enumerate(non_empty):
        py0 = y0 + i * slice_h
        py1 = y0 + (i + 1) * slice_h if i < n - 1 else y1
        segments = _paragraph_segments(para)
        for (seg_text, seg_size, seg_font, seg_bold) in segments:
            key = (seg_text.lower(), seg_bold)
            if key == last_seen:
                continue
            last_seen = key
            text_lines.append(
                TextLine(
                    bbox_pct=(x0, py0, x1, py1),
                    text=seg_text,
                    size=seg_size,
                    font=seg_font,
                    bold=seg_bold,
                    group_id=group_id,
                )
            )


def _table_to_model(shape, slide_w: int, slide_h: int) -> Table:
    tbl = shape.table
    rows_iter = list(tbl.rows)
    columns: list[str] = []
    body_rows: list[list[str]] = []

    if rows_iter:
        header_cells = list(rows_iter[0].cells)
        columns = [" ".join((c.text or "").split()) for c in header_cells]
        for row in rows_iter[1:]:
            body_rows.append([" ".join((c.text or "").split()) for c in row.cells])

    return Table(
        bbox_pct=_shape_bbox_pct(shape, slide_w, slide_h),
        columns=columns,
        rows=body_rows,
    )


def _walk_shapes(
    shapes,
    slide_w: int,
    slide_h: int,
    text_lines: list[TextLine],
    tables: list[Table],
    figures: list[Figure],
    counter: list[int],
) -> None:
    """`counter` is a single-element list used as a mutable next-group-id
    across the recursive walk (Python has no `nonlocal int` shorthand)."""
    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            _walk_shapes(shape.shapes, slide_w, slide_h, text_lines, tables, figures, counter)
            continue

        # `has_table` is only defined on graphic-frame shapes; getattr keeps
        # non-graphic shapes from raising AttributeError on older python-pptx.
        if getattr(shape, "has_table", False):
            tables.append(_table_to_model(shape, slide_w, slide_h))
            continue

        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            figures.append(Figure(bbox_pct=_shape_bbox_pct(shape, slide_w, slide_h)))
            # Pictures don't carry text; skip further processing.
            continue

        if getattr(shape, "has_text_frame", False):
            counter[0] += 1
            _text_frame_lines(shape, slide_w, slide_h, text_lines, counter[0])


def _table_iou(a: Table, b: Table) -> float:
    ax0, ay0, ax1, ay1 = a.bbox_pct
    bx0, by0, bx1, by1 = b.bbox_pct
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    a_area = max(0.0, (ax1 - ax0) * (ay1 - ay0))
    b_area = max(0.0, (bx1 - bx0) * (by1 - by0))
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def extract_slide_layout(
    pptx_path: Path,
    slide_num: int,
    companion_pdf_path: Path | None = None,
) -> PageLayout:
    """Extract text lines + tables + figure bboxes from a single 1-indexed slide.

    When `companion_pdf_path` is provided (the LibreOffice-rendered PDF the
    pipeline already builds for slide images), we ALSO run pdfplumber's
    ruled-line table detector against that page and merge any tables it
    finds. This catches "drawn tables" — grids built by hand from text
    boxes + rectangles — that python-pptx doesn't see as tables at all
    because they aren't native pptx table shapes. Tables that overlap an
    already-extracted native table by >=30% IoU are treated as duplicates
    and dropped.
    """
    prs = Presentation(str(pptx_path))
    slides = list(prs.slides)
    if slide_num < 1 or slide_num > len(slides):
        return PageLayout(page_num=slide_num, width=0, height=0)

    slide = slides[slide_num - 1]
    slide_w = int(prs.slide_width or 0)
    slide_h = int(prs.slide_height or 0)

    text_lines: list[TextLine] = []
    tables: list[Table] = []
    figures: list[Figure] = []

    _walk_shapes(slide.shapes, slide_w, slide_h, text_lines, tables, figures, counter=[0])

    # pdfplumber fallback: catch drawn-shape tables python-pptx misses.
    if companion_pdf_path is not None:
        for pt in _pdfplumber_tables(companion_pdf_path, slide_num):
            if any(_table_iou(pt, nt) >= 0.3 for nt in tables):
                continue
            tables.append(pt)

    # Drop text lines that fall inside any detected table's bbox — the table
    # extractor is authoritative for those (mirrors pdf_layout behavior).
    if tables:
        table_bboxes = [t.bbox_pct for t in tables]

        def _inside_any(b: tuple[float, float, float, float]) -> bool:
            tx0, ty0, tx1, ty1 = b
            cx = (tx0 + tx1) / 2
            cy = (ty0 + ty1) / 2
            for bx0, by0, bx1, by1 in table_bboxes:
                if bx0 <= cx <= bx1 and by0 <= cy <= by1:
                    return True
            return False

        text_lines = [tl for tl in text_lines if not _inside_any(tl.bbox_pct)]

    # EMU -> points for width/height so consumers can render the payload at
    # the same scale semantics as PDF (`page_size` in "PDF points"). 1 pt = 12700 EMU.
    width_pt = round(slide_w / 12700.0, 2) if slide_w else 0.0
    height_pt = round(slide_h / 12700.0, 2) if slide_h else 0.0

    return PageLayout(
        page_num=slide_num,
        width=width_pt,
        height=height_pt,
        text_lines=text_lines,
        figures=figures,
        tables=tables,
    )


def slide_count(pptx_path: Path) -> int:
    prs = Presentation(str(pptx_path))
    return len(list(prs.slides))


def extract_slide_notes(pptx_path: Path, slide_num: int) -> str:
    """Return the speaker-notes text for a 1-indexed slide, "" if none.

    Notes live in slide.notes_slide.notes_text_frame; we join paragraphs
    with newlines and collapse internal whitespace per paragraph so the
    return value is drop-in for a `body` field.
    """
    prs = Presentation(str(pptx_path))
    slides = list(prs.slides)
    if slide_num < 1 or slide_num > len(slides):
        return ""
    slide = slides[slide_num - 1]
    if not getattr(slide, "has_notes_slide", False):
        return ""
    tf = slide.notes_slide.notes_text_frame
    if tf is None:
        return ""
    lines: list[str] = []
    for para in tf.paragraphs:
        text = " ".join((para.text or "").split())
        if text:
            lines.append(text)
    return "\n".join(lines).strip()
