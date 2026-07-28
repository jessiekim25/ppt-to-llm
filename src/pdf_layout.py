"""pdfminer.six-based per-page layout extraction: text lines + figure clusters.

pypdfium2 only sees raster image objects, but most "images" in these decks
are vector-drawn (clip-masked shapes, paths, fills). pdfminer walks the full
LT-object tree and gives us the vector primitives, so we can cluster nearby
paths/rects/lines/images into coherent figure regions and attach the nearest
short text label as a caption.
"""
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
from pdfminer.high_level import extract_pages
from pdfminer.layout import (
    LAParams,
    LTChar,
    LTCurve,
    LTFigure,
    LTImage,
    LTLine,
    LTRect,
    LTTextBox,
    LTTextLine,
)


@dataclass
class TextLine:
    """One text line in top-left-origin percent coords."""

    bbox_pct: tuple[float, float, float, float]  # (x0, y0, x1, y1), fractions of page
    text: str
    size: float | None
    font: str = ""
    bold: bool = False


@dataclass
class Figure:
    """One clustered figure region in top-left-origin percent coords."""

    bbox_pct: tuple[float, float, float, float]
    label: str = ""


@dataclass
class Table:
    """One pdfplumber-detected table in top-left-origin percent coords."""

    bbox_pct: tuple[float, float, float, float]
    columns: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)


@dataclass
class PageLayout:
    page_num: int
    width: float
    height: float
    text_lines: list[TextLine] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)


def _to_pct_top_left(
    bbox_pdf: tuple[float, float, float, float], w: float, h: float
) -> tuple[float, float, float, float]:
    """PDF bottom-left bbox -> top-left origin, fractions of page."""
    x0, y0, x1, y1 = bbox_pdf
    return (x0 / w, (h - y1) / h, x1 / w, (h - y0) / h)


def _near_or_overlap(a, b, gap: float) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 + gap < bx0 or bx1 + gap < ax0 or ay1 + gap < by0 or by1 + gap < ay0)


def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent, x, y):
    rx, ry = _find(parent, x), _find(parent, y)
    if rx != ry:
        parent[rx] = ry


def _first_char_style(container) -> tuple[float | None, str]:
    """Font size + name of the first LTChar found in a container (walks one level of lines)."""
    for child in container:
        if isinstance(child, LTTextLine):
            for c in child:
                if isinstance(c, LTChar):
                    return c.size, (c.fontname or "")
            continue
        if isinstance(child, LTChar):
            return child.size, (child.fontname or "")
    return None, ""


def _walk(node, texts: list, shapes: list) -> None:
    """Collect text as PARAGRAPH-level entries (LTTextBox) and shape primitives.

    A LTTextBox is pdfminer's paragraph unit: all its wrapped lines belong to one
    logical block. We join them into a single text string so a caption like
    "Do not create new product / image arrangements or modify screen images."
    lands as ONE entry, not four, and the LLM can't split it into subheader+body.
    """
    if isinstance(node, LTTextBox):
        text = " ".join(node.get_text().split())
        if text:
            size, font = _first_char_style(node)
            texts.append((node.bbox, text, size, font))
        return
    if isinstance(node, LTTextLine):
        # Loose text line outside any box — rare, but capture it.
        size = None
        font = ""
        for c in node:
            if isinstance(c, LTChar):
                size = c.size
                font = c.fontname or ""
                break
        text = node.get_text().strip()
        if text:
            texts.append((node.bbox, text, size, font))
        return
    if isinstance(node, (LTImage, LTCurve, LTRect, LTLine)):
        shapes.append((node.bbox, type(node).__name__))
    if isinstance(node, LTFigure) or hasattr(node, "__iter__"):
        try:
            for child in node:
                _walk(child, texts, shapes)
        except TypeError:
            pass


def _cluster_shapes(
    shapes: list,
    page_w: float,
    page_h: float,
    gap: float = 6.0,
    min_area: float = 900.0,
) -> list[tuple[float, float, float, float]]:
    """Union nearby shape bboxes into figure clusters, drop noise/dividers."""
    n = len(shapes)
    parent = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if _near_or_overlap(shapes[i][0], shapes[j][0], gap=gap):
                _union(parent, i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(_find(parent, i), []).append(i)

    clusters: list[tuple[float, float, float, float]] = []
    for members in groups.values():
        types = [shapes[i][1] for i in members]
        bs = [shapes[i][0] for i in members]
        x0, y0 = min(b[0] for b in bs), min(b[1] for b in bs)
        x1, y1 = max(b[2] for b in bs), max(b[3] for b in bs)
        width, height = x1 - x0, y1 - y0

        if all(t == "LTLine" for t in types) and (height < 3 or width < 3):
            continue  # divider rule
        if width * height < min_area:
            continue  # too small to be a figure
        if width > 0.95 * page_w and height > 0.95 * page_h:
            continue  # full-page background

        clusters.append((x0, y0, x1, y1))
    return clusters


def _attach_labels(
    clusters: list[tuple[float, float, float, float]],
    text_lines_pdf: list,
) -> list[tuple[tuple[float, float, float, float], str]]:
    """For each cluster, prefer nearest short caption below; else badge digit above."""
    out: list[tuple[tuple[float, float, float, float], str]] = []
    for c in clusters:
        cx0, cy0, cx1, cy1 = c
        best = ""
        best_dist: float | None = None

        for tbbox, ttext, _size, _font in text_lines_pdf:
            if len(ttext) > 60:
                continue
            tx0, ty0, tx1, ty1 = tbbox
            if ty1 <= cy0 and (cy0 - ty1) <= 25 and tx1 >= cx0 and tx0 <= cx1:
                d = cy0 - ty1
                if best_dist is None or d < best_dist:
                    best_dist, best = d, ttext

        if not best:
            for tbbox, ttext, _size, _font in text_lines_pdf:
                if len(ttext) > 4:
                    continue
                tx0, ty0, tx1, ty1 = tbbox
                if ty0 >= cy1 and (ty0 - cy1) <= 15 and tx1 >= cx0 and tx0 <= cx1:
                    d = ty0 - cy1
                    if best_dist is None or d < best_dist:
                        best_dist, best = d, ttext

        out.append((c, best))
    return out


def _fill_merged_cells_down(rows: list[list[str]]) -> list[list[str]]:
    """Propagate the last non-empty value down each column across empty cells.

    pdfplumber returns the merged-cell text in the top row of the merge and
    empty strings in the other rows — so an empty cell almost always means
    "merged with the cell above." Filling down restores the visual meaning
    (both product rows share the same disclaimer). Legitimately-empty cells
    in spec tables are rare enough that the tradeoff favors fill-down.
    """
    if not rows:
        return rows
    ncols = max(len(r) for r in rows)
    last: list[str] = [""] * ncols
    out: list[list[str]] = []
    for row in rows:
        padded = list(row) + [""] * (ncols - len(row))
        new_row: list[str] = []
        for c, cell in enumerate(padded):
            if cell and cell.strip():
                last[c] = cell
                new_row.append(cell)
            else:
                new_row.append(last[c])
        out.append(new_row)
    return out


def _extract_tables(pdf_path: Path, page_num: int) -> list[Table]:
    """Detect tables via pdfplumber using visible ruling lines.

    Restricted to lines-only detection (`vertical_strategy`/`horizontal_strategy` = "lines")
    to avoid false-positive tables on multi-column subheader layouts that only
    happen to align on a grid. If your deck uses borderless tables and needs
    text-alignment fallback, loosen these settings.
    """
    tables_out: list[Table] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_num < 1 or page_num > len(pdf.pages):
                return tables_out
            page = pdf.pages[page_num - 1]
            page_w = float(page.width) or 1.0
            page_h = float(page.height) or 1.0
            for t in page.find_tables(table_settings={
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
            }):
                data = t.extract() or []
                if not data:
                    continue
                columns = [str(c or "").strip() for c in data[0]]
                raw_rows = [[str(c or "").strip() for c in row] for row in data[1:]]
                rows = _fill_merged_cells_down(raw_rows)
                if not any(columns) and not any(any(r) for r in rows):
                    continue
                x0, top, x1, bottom = t.bbox
                tables_out.append(
                    Table(
                        bbox_pct=(x0 / page_w, top / page_h, x1 / page_w, bottom / page_h),
                        columns=columns,
                        rows=rows,
                    )
                )
    except Exception as e:
        print(f"  ! pdfplumber table detection failed for page {page_num}: {e}")
    return tables_out


def _text_inside_any_bbox(
    text_bbox_pct: tuple[float, float, float, float],
    table_bboxes: list[tuple[float, float, float, float]],
) -> bool:
    """True if the text bbox's center falls inside any table bbox."""
    tx0, ty0, tx1, ty1 = text_bbox_pct
    cx = (tx0 + tx1) / 2
    cy = (ty0 + ty1) / 2
    for bx0, by0, bx1, by1 in table_bboxes:
        if bx0 <= cx <= bx1 and by0 <= cy <= by1:
            return True
    return False


def extract_page_layout(pdf_path: Path, page_num: int) -> PageLayout:
    """Extract text lines + figure clusters + tables from a single 1-indexed page.

    pdfplumber owns table detection (visible ruling lines). Text lines whose
    center falls inside a detected table's bbox are dropped from `text_lines`
    so the LLM never re-encodes them as subheaders or body — the pre-extracted
    table content is authoritative.
    """
    tables = _extract_tables(pdf_path, page_num)
    table_bboxes = [t.bbox_pct for t in tables]

    la = LAParams()
    for page in extract_pages(
        str(pdf_path), page_numbers=[page_num - 1], laparams=la
    ):
        page_w = float(page.width)
        page_h = float(page.height)
        texts_raw: list = []
        shapes: list = []
        _walk(page, texts_raw, shapes)

        clusters_pdf = _cluster_shapes(shapes, page_w, page_h)
        labeled = _attach_labels(clusters_pdf, texts_raw)
        # Reading order: top-to-bottom (high PDF y first), then left-to-right.
        labeled.sort(key=lambda cl: (-cl[0][3], cl[0][0]))

        text_lines = [
            TextLine(
                bbox_pct=_to_pct_top_left(b, page_w, page_h),
                text=t,
                size=s,
                font=f,
                bold="bold" in (f or "").lower(),
            )
            for (b, t, s, f) in texts_raw
        ]
        if table_bboxes:
            text_lines = [
                tl for tl in text_lines
                if not _text_inside_any_bbox(tl.bbox_pct, table_bboxes)
            ]
        figures = [
            Figure(bbox_pct=_to_pct_top_left(b, page_w, page_h), label=lbl)
            for (b, lbl) in labeled
        ]
        return PageLayout(
            page_num=page_num,
            width=page_w,
            height=page_h,
            text_lines=text_lines,
            figures=figures,
            tables=tables,
        )

    return PageLayout(page_num=page_num, width=0, height=0, tables=tables)
