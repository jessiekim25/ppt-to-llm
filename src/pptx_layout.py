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


_DRAWINGML_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"


def _shape_has_table_xml(shape) -> bool:
    """True if the shape's underlying XML contains a DrawingML <a:tbl>.

    Some pptx files wrap tables inside placeholders or with non-standard
    graphicData URIs, so python-pptx's shape.has_table returns False even
    though the table is right there in the XML. This scans the element
    tree directly. Group shapes are handled by the recursion in
    _walk_shapes, so we never check them here.
    """
    try:
        return shape.element.find(f".//{{{_DRAWINGML_NS}}}tbl") is not None
    except Exception:
        return False


def _table_from_xml(shape, slide_w: int, slide_h: int) -> Table | None:
    """Reconstruct a Table from the raw <a:tbl> XML inside `shape`.

    Fallback for shapes where python-pptx doesn't expose .table but the
    underlying pptx still holds a proper DrawingML table. Cells joined
    via horizontal/vertical merge write empty strings for their
    continuation cells (matches the shape of _table_to_model).
    """
    ns = f"{{{_DRAWINGML_NS}}}"
    try:
        tbl = shape.element.find(f".//{ns}tbl")
    except Exception:
        return None
    if tbl is None:
        return None

    columns: list[str] = []
    body_rows: list[list[str]] = []

    for i, tr in enumerate(tbl.findall(f"{ns}tr")):
        cells: list[str] = []
        for tc in tr.findall(f"{ns}tc"):
            # A merge continuation cell (hMerge="1" or vMerge="1") holds no
            # text of its own — leave it empty so downstream renderers can
            # decide whether to fill-down.
            if tc.get("hMerge") == "1" or tc.get("vMerge") == "1":
                cells.append("")
                continue
            # Concatenate every <a:t> under this cell as its text.
            parts = [t.text or "" for t in tc.iter(f"{ns}t")]
            cells.append(" ".join("".join(parts).split()))
        if i == 0:
            columns = cells
        else:
            body_rows.append(cells)

    if not columns and not body_rows:
        return None

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

        # Native pptx table (GraphicFrame with the standard table URI).
        if getattr(shape, "has_table", False):
            tables.append(_table_to_model(shape, slide_w, slide_h))
            continue

        # Fallback: a table whose XML wrapper doesn't match python-pptx's
        # URI check (some exports/placeholders land here). Reconstruct the
        # table from the raw <a:tbl> element and skip further processing.
        if _shape_has_table_xml(shape):
            xml_table = _table_from_xml(shape, slide_w, slide_h)
            if xml_table is not None:
                tables.append(xml_table)
                continue

        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            figures.append(Figure(bbox_pct=_shape_bbox_pct(shape, slide_w, slide_h)))
            # Pictures don't carry text; skip further processing.
            continue

        if getattr(shape, "has_text_frame", False):
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


_PRESENTATIONML_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"


def diagnose_slide_shapes(pptx_path: Path, slide_num: int) -> str:
    """Return a multi-line dump of every shape on `slide_num`, showing the
    signals my walker uses (shape_type, has_table, has_text_frame, plus
    XML-descendant checks for <a:tbl> and <p:oleObj>). Use this when a
    table isn't being extracted and you want to see why.
    """
    from xml.etree import ElementTree as _ET  # cheap local import

    def _local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1] if "}" in tag else tag

    prs = Presentation(str(pptx_path))
    slides = list(prs.slides)
    if slide_num < 1 or slide_num > len(slides):
        return f"slide {slide_num} out of range (1..{len(slides)})"

    slide = slides[slide_num - 1]
    out: list[str] = [f"=== slide {slide_num} of {len(slides)} — "
                      f"{len(list(slide.shapes))} top-level shapes ==="]

    def dump(shape, depth: int = 0) -> None:
        pad = "  " * depth
        tag = _local(shape.element.tag)
        out.append(
            f"{pad}- name={getattr(shape, 'name', '?')!r} "
            f"shape_type={getattr(shape, 'shape_type', None)} "
            f"element=<{tag}>"
        )
        out.append(
            f"{pad}    has_table={getattr(shape, 'has_table', 'N/A')} "
            f"has_text_frame={getattr(shape, 'has_text_frame', 'N/A')}"
        )
        try:
            tbl = shape.element.find(f".//{{{_DRAWINGML_NS}}}tbl")
            out.append(f"{pad}    descendant <a:tbl>: {'YES' if tbl is not None else 'no'}")
        except Exception as e:
            out.append(f"{pad}    <a:tbl> check errored: {e}")
        try:
            gd = shape.element.find(f".//{{{_DRAWINGML_NS}}}graphic/{{{_DRAWINGML_NS}}}graphicData")
            if gd is not None:
                out.append(f"{pad}    graphicData uri={gd.get('uri')!r}")
        except Exception:
            pass
        try:
            ole = shape.element.find(f".//{{{_PRESENTATIONML_NS}}}oleObj")
            if ole is not None:
                out.append(
                    f"{pad}    contains <p:oleObj> (embedded object): prog={ole.get('progId')!r}"
                )
        except Exception:
            pass
        # Recurse into groups
        try:
            for child in getattr(shape, "shapes", []):
                dump(child, depth + 1)
        except Exception:
            pass

    for shape in slide.shapes:
        dump(shape)

    out.append("")
    out.append("Legend:")
    out.append("  has_table=True                     -> extracted natively")
    out.append("  descendant <a:tbl>: YES            -> caught by XML fallback")
    out.append("  contains <p:oleObj> ...            -> embedded Excel object; not extractable")
    out.append("                                        (only a visual snapshot is stored)")
    return "\n".join(out)


def extract_slide_notes(pptx_path: Path, slide_num: int) -> str:
    """Return the speaker-notes text for a 1-indexed slide, "" if none.

    Preserves the note's line structure: each paragraph is one line, blank
    paragraphs stay as blank lines (so a note written as two prose blocks
    separated by an empty line keeps that gap), and soft line breaks
    inside a paragraph (Shift-Enter, DrawingML `<a:br/>` — python-pptx
    exposes these as `\\v`) become real newlines. Only leading and
    trailing blank lines are trimmed.
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
        raw = (para.text or "").replace("\v", "\n")
        # Preserve internal spacing within each visual line; trim trailing whitespace.
        for line in raw.split("\n"):
            lines.append(line.rstrip())

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)
