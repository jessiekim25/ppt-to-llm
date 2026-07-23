"""pdfminer.six-based per-page layout extraction: text lines + figure clusters.

pypdfium2 only sees raster image objects, but most "images" in these decks
are vector-drawn (clip-masked shapes, paths, fills). pdfminer walks the full
LT-object tree and gives us the vector primitives, so we can cluster nearby
paths/rects/lines/images into coherent figure regions and attach the nearest
short text label as a caption.
"""
from dataclasses import dataclass, field
from pathlib import Path

from pdfminer.high_level import extract_pages
from pdfminer.layout import (
    LAParams,
    LTChar,
    LTCurve,
    LTFigure,
    LTImage,
    LTLine,
    LTRect,
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
class PageLayout:
    page_num: int
    width: float
    height: float
    text_lines: list[TextLine] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)


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


def _walk(node, texts: list, shapes: list) -> None:
    if isinstance(node, LTTextLine):
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


def extract_page_layout(pdf_path: Path, page_num: int) -> PageLayout:
    """Extract text lines + figure clusters from a single 1-indexed page."""
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
        )

    return PageLayout(page_num=page_num, width=0, height=0)
