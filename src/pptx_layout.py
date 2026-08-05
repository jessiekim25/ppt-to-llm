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

from .pdf_layout import Figure, PageLayout, Table, TextLine


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


def _paragraph_style(para) -> tuple[float | None, str, bool]:
    """Return (font_size_pt, font_name, is_bold) from the first non-empty run.

    Size falls back to paragraph-level default; bold combines run flag,
    paragraph flag, and font-name inspection so slides that inherit weight
    from the theme still get flagged bold (which is how most subheaders and
    table column headers are styled).
    """
    para_font_name = para.font.name or ""
    para_bold_flag = para.font.bold  # True / False / None
    para_size = None
    try:
        if para.font.size is not None:
            para_size = float(para.font.size.pt)
    except Exception:
        para_size = None

    for run in para.runs:
        if not (run.text or "").strip():
            continue
        size = None
        try:
            if run.font.size is not None:
                size = float(run.font.size.pt)
        except Exception:
            size = None
        if size is None:
            size = para_size
        font = run.font.name or para_font_name
        bold = _resolve_bold(run.font.bold, font, para_bold_flag, para_font_name)
        return size, font, bold

    return para_size, para_font_name, _resolve_bold(para_bold_flag, para_font_name)


def _text_frame_lines(
    shape,
    slide_w: int,
    slide_h: int,
    text_lines: list[TextLine],
    group_id: int,
) -> None:
    """Emit one TextLine per non-empty paragraph in this shape's text frame.

    We approximate per-paragraph bboxes by slicing the shape's bbox vertically
    across N paragraphs. `group_id` tags every line from this one text frame
    with the same integer so the LLM can split alternating bold/non-bold
    paragraphs inside one box into subheader+body pairs without having to
    guess grouping from x-alignment alone.
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

    for i, para in enumerate(non_empty):
        text = " ".join((para.text or "").split())
        if not text:
            continue
        size, font, bold = _paragraph_style(para)
        py0 = y0 + i * slice_h
        py1 = y0 + (i + 1) * slice_h if i < n - 1 else y1
        text_lines.append(
            TextLine(
                bbox_pct=(x0, py0, x1, py1),
                text=text,
                size=size,
                font=font,
                bold=bold,
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

        if shape.has_table:
            tables.append(_table_to_model(shape, slide_w, slide_h))
            continue

        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            figures.append(Figure(bbox_pct=_shape_bbox_pct(shape, slide_w, slide_h)))
            # Pictures don't carry text; skip further processing.
            continue

        if shape.has_text_frame:
            counter[0] += 1
            _text_frame_lines(shape, slide_w, slide_h, text_lines, counter[0])


def extract_slide_layout(pptx_path: Path, slide_num: int) -> PageLayout:
    """Extract text lines + tables + figure bboxes from a single 1-indexed slide."""
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
