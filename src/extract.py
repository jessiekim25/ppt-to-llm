import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

from shared.settings import get_settings

from .chunk import chunk_file
from .corpus import build_corpus
from .llm import build_payload, extract_slide, extract_slide_from_image
from .pdf_layout import Table, TextLine, extract_page_layout
from .pdf_utils import page_count, render_page, resolve_pdf_input
from .pptx_layout import (
    diagnose_slide_shapes as diagnose_pptx_slide_shapes,
    extract_slide_layout as extract_pptx_slide_layout,
    extract_slide_notes as extract_pptx_slide_notes,
    slide_count as pptx_slide_count,
)
from .pptx_utils import pptx_to_pdf, resolve_pptx_input

SLIDE_FIELDS = ("product", "section", "sub_section", "model")


def _slug(text: str, max_len: int = 60) -> str:
    """Filesystem-safe slug: ASCII alnum, hyphen, underscore; spaces to underscores."""
    text = re.sub(r"[^A-Za-z0-9\s\-_]", "", str(text))
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"_+", "_", text)
    return text[:max_len].strip("_-")


def _render_table(t: dict) -> str:
    """Render one table as 'title\\ncol1 | col2 | ...\\ncell11 | cell12 | ...' — multi-line cells joined with ' / '."""
    def cell(v: object) -> str:
        parts = [p.strip() for p in str(v).splitlines() if p.strip()]
        return " / ".join(parts)

    lines: list[str] = []
    title = str(t.get("title", "") or "").strip()
    if title:
        lines.append(title)
    columns = [str(c).strip() for c in (t.get("columns") or [])]
    if columns:
        lines.append(" | ".join(columns))
    for row in t.get("rows") or []:
        cells = [cell(c) for c in row]
        if columns and len(cells) < len(columns):
            cells += [""] * (len(columns) - len(cells))
        lines.append(" | ".join(cells))
    return "\n".join(lines)


_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+\s*[.\):]|[-•*])\s+")


def _looks_like_list_marker(text: str) -> bool:
    """True if text starts with a numbered-step marker ('1.', '2)', '3:') or a
    dash/bullet marker ('- ', '• ').

    Numbered step items and bullet items are body content, not subheader labels
    — even when the first wrapped line comes in bold or gets split off by
    pdfminer. Rejecting them as subheaders forces `_block_from_subheader` to
    demote them into a plain body block that keeps the wrapped continuation.
    """
    return bool(_LIST_MARKER_RE.match(text or ""))


def _is_valid_subheader_title(title: str) -> bool:
    """Enforce the prompt's (α) rule: <=10 words, no sentence-terminal punctuation,
    and no list-marker prefix.

    A trailing colon is fine ("How to build layout:" is a real subheader).
    A trailing period/question/exclamation mark means it's a sentence, not a label.
    A leading step marker ("2. ...") or dash bullet ("- ...") means it's an
    enumerated body item whose first source line got styled/split and then
    promoted by the LLM — demote it.
    """
    t = title.rstrip()
    if not t:
        return False
    if t[-1] in ".?!":
        return False
    if len(t.split()) > 10:
        return False
    if _looks_like_list_marker(t):
        return False
    return True


def _block_from_subheader(sh: dict) -> list[dict]:
    """Turn one LLM subheader entry into a LIST of detail blocks (usually 1).

    - A subheader whose title fails validation (too long, or ends in a period)
      is demoted: the title is prepended to the body as prose, children hoist
      as sibling blocks in the parent's list. That way misclassified sentences
      keep their text and their real children.
    - A subheader with only a title collapses to a plain body block.
    - LLM no longer emits `tables`; any such field is ignored.
    """
    if not isinstance(sh, dict):
        return []
    title = str(sh.get("title", "") or "").strip()
    body = str(sh.get("detail", "") or "").strip()

    child_blocks: list[dict] = []
    for c in sh.get("children") or []:
        child_blocks.extend(_block_from_subheader(c))

    if title and not _is_valid_subheader_title(title):
        merged = f"{title} {body}".strip() if body else title
        demoted = [{"body": merged}] if merged else []
        return demoted + child_blocks

    if title and not body and not child_blocks:
        return [{"body": title}]

    block: dict = {}
    if title:
        block["subheader"] = title
    if body:
        block["body"] = body
    if child_blocks:
        block["children"] = child_blocks
    return [block] if block else []


def _blocks_from_group_paragraphs(tls: list[TextLine]) -> list[dict]:
    """Convert one text frame's paragraphs (sorted top-to-bottom) into blocks.

    - Emphasized paragraph (bold OR underline) -> a subheader block.
    - Non-emphasized paragraph -> body of the most recent subheader (or a
      plain body block at top level if no subheader has been seen yet).
    - Nesting: if the group has more than one distinct emphasized `size`, an
      emphasized paragraph strictly smaller than the max becomes a child of
      the current parent — so a "KEY INITIATIVES" at 14pt nests under a
      "Week 9-14, March" at 20pt in the same text box.
    """
    if not tls:
        return []

    def _emph(tl: TextLine) -> bool:
        return tl.bold or tl.underline

    emphasized_sizes = {tl.size for tl in tls if _emph(tl) and tl.size is not None}
    parent_size = max(emphasized_sizes) if len(emphasized_sizes) >= 2 else None

    top_level: list[dict] = []
    current_parent: dict | None = None
    current_child: dict | None = None

    for tl in tls:
        text = tl.text.strip()
        if not text:
            continue
        # A bold-styled paragraph that starts with a list marker ("- ", "1.",
        # "2)", …) is still a body item, not a subheader — some templates set
        # every bullet in bold and pdf/pptx extractors preserve that styling.
        is_subheader = (
            _emph(tl)
            and len(text.split()) <= 10
            and not _looks_like_list_marker(text)
        )
        if is_subheader:
            is_child = (
                parent_size is not None
                and tl.size is not None
                and tl.size < parent_size
                and current_parent is not None
            )
            if is_child:
                current_child = {"subheader": text}
                current_parent.setdefault("children", []).append(current_child)
            else:
                current_parent = {"subheader": text}
                current_child = None
                top_level.append(current_parent)
        else:
            target = current_child if current_child is not None else current_parent
            if target is None:
                # Extend a preceding standalone-body block instead of starting a
                # new one so a group of consecutive non-bold paragraphs (e.g. a
                # legend: BACKLOG / UX/BUILD/QA / LIVE/DONE) stays as one body.
                if top_level and set(top_level[-1].keys()) == {"body"}:
                    top_level[-1]["body"] += f"\n{text}"
                else:
                    top_level.append({"body": text})
            else:
                existing = target.get("body", "")
                target["body"] = f"{existing}\n{text}" if existing else text

    return top_level


# Groups wider than this fraction of the slide are treated as slide-wide
# (titles, intro paragraphs, legends) and are NEVER clustered with columns.
_COLUMN_MAX_WIDTH_FRAC = 0.6
# Fraction of the smaller group's x-width that two groups must overlap on
# to be considered part of the same visual column.
_COLUMN_X_OVERLAP_MIN = 0.5
# Maximum vertical gap between two groups (as a fraction of slide height)
# for them to still count as one column. Stops a bottom-of-page legend that
# happens to sit under column 1 from being absorbed into that column's body.
_COLUMN_Y_GAP_MAX = 0.05


def _cluster_columns_by_x(group_geom: dict[int, tuple[float, float, float, float]]) -> dict[int, int]:
    """Union-find groups into column-clusters by x-overlap AND y-adjacency.

    Slide-wide groups (width > _COLUMN_MAX_WIDTH_FRAC) never merge — they'd
    otherwise pull an entire multi-column band into one cluster because
    their x-range engulfs every column. Two groups also need to be
    vertically adjacent (either overlapping in y or separated by less than
    _COLUMN_Y_GAP_MAX of the slide height); this keeps a bottom-of-page
    legend or footnote out of the column body directly above it even when
    x-alignment matches. Returns {group_id: cluster_root_id}.
    """
    gids = list(group_geom.keys())
    parent = {g: g for g in gids}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    def column_like(g):
        x0, _, x1, _ = group_geom[g]
        return (x1 - x0) <= _COLUMN_MAX_WIDTH_FRAC

    for i, g1 in enumerate(gids):
        if not column_like(g1):
            continue
        x0a, y0a, x1a, y1a = group_geom[g1]
        wa = x1a - x0a
        for g2 in gids[i + 1:]:
            if not column_like(g2):
                continue
            x0b, y0b, x1b, y1b = group_geom[g2]
            wb = x1b - x0b
            overlap = max(0.0, min(x1a, x1b) - max(x0a, x0b))
            min_w = min(wa, wb) or 1.0
            if overlap / min_w < _COLUMN_X_OVERLAP_MIN:
                continue
            # y-adjacency: negative gap = they overlap in y (still cluster).
            y_gap = max(y0a, y0b) - min(y1a, y1b)
            if y_gap > _COLUMN_Y_GAP_MAX:
                continue
            union(g1, g2)

    return {g: find(g) for g in gids}


def _build_detail_from_pptx_groups(
    text_lines: list[TextLine],
    tables: list[Table],
    exclude_haystack: str,
    slide_notes: str = "",
) -> list[dict]:
    """Deterministically build the per-slide `detail` tree from pptx text frames.

    The LLM is not asked to structure `detail` for pptx — every paragraph is
    already tagged with `group_id` (its source text frame) plus `bold`/`size`
    typography, which is enough to reconstruct subheader/body pairs and
    parent/child nesting without any LLM guessing.

    Text frames that share an x-range (>=50% overlap on the narrower one,
    and each width <=60% of the slide) are first fused into a virtual
    "column". This is what lets a Roadmap slide's three "Week X" heading
    boxes each nest their own separately-boxed "KEY INITIATIVES + bullets"
    frame beneath them, even though the heading and the body live in two
    different text frames per column.

    Slide-wide groups (titles, intro paragraphs, legends) never fuse — their
    x-range engulfs every column and would collapse the whole band.

    `exclude_haystack` is a normalized string of slide-level values already
    emitted elsewhere on the record (currently `sub_section`); any paragraph
    whose normalized text is a substring of that haystack is dropped from
    detail to avoid duplication.

    Tables append as `{"table": ...}` blocks. Speaker notes, when present,
    are added as a final `{"subheader": "slide note", "body": ...}` block.
    """
    if not text_lines and not tables and not slide_notes:
        return []

    groups: dict[int, list[TextLine]] = defaultdict(list)
    for tl in text_lines:
        norm = _normalize_for_match(tl.text)
        # Exclude only when the paragraph IS the sub_section (whole-string
        # equality), not when it's a substring of it. Previously a slide
        # title like "July highlights & challenges" would silently drop
        # every downstream subheader named "Highlights" or "Challenges" —
        # `"highlights"` is a substring of the title, so the whole
        # subheader disappeared before block-building ever ran.
        if norm and exclude_haystack and norm == exclude_haystack:
            continue
        groups[tl.group_id or 0].append(tl)

    group_geom: dict[int, tuple[float, float, float, float]] = {}
    for gid, tls in groups.items():
        x0 = min(t.bbox_pct[0] for t in tls)
        y0 = min(t.bbox_pct[1] for t in tls)
        x1 = max(t.bbox_pct[2] for t in tls)
        y1 = max(t.bbox_pct[3] for t in tls)
        group_geom[gid] = (x0, y0, x1, y1)

    cluster_of = _cluster_columns_by_x(group_geom)

    # Fuse all paragraphs from groups in the same column-cluster into one
    # ordered stream and hand that to the block builder — this is what
    # nests the "KEY INITIATIVES" text frame beneath its column's "Week X"
    # heading frame.
    columns: dict[int, list[TextLine]] = defaultdict(list)
    for gid, tls in groups.items():
        columns[cluster_of[gid]].extend(tls)

    column_infos: list[dict] = []
    for tls in columns.values():
        tls.sort(key=lambda t: (t.bbox_pct[1], t.bbox_pct[0]))
        blocks = _blocks_from_group_paragraphs(tls)
        if not blocks:
            continue
        y0 = min(t.bbox_pct[1] for t in tls)
        x0 = min(t.bbox_pct[0] for t in tls)
        column_infos.append({"y0": y0, "x0": x0, "blocks": blocks})

    # y-band to ~5% of page so parallel columns stay adjacent in output.
    column_infos.sort(key=lambda g: (round(g["y0"] * 20), g["x0"]))

    result: list[dict] = []
    for ci in column_infos:
        result.extend(ci["blocks"])

    for t in tables:
        rendered = _render_table({"columns": t.columns, "rows": t.rows}).strip()
        if rendered:
            result.append({"table": rendered})

    notes = (slide_notes or "").strip()
    if notes:
        result.append({"subheader": "slide note", "body": notes})

    return result


def _compose_detail(extracted: dict, pre_extracted_tables: list[Table] = ()) -> list[dict]:
    """Return the slide's text hierarchy as a list of blocks in reading order.

    Tables come exclusively from pdfplumber. The LLM no longer emits `tables`,
    so grid-aligned image labels or short captions become subheaders instead of
    being packed into a fake table.
    """
    blocks: list[dict] = []

    slide_body = str(extracted.get("detail", "") or "").strip()
    if slide_body:
        blocks.append({"body": slide_body})

    for t in pre_extracted_tables:
        rendered = _render_table({"columns": t.columns, "rows": t.rows}).strip()
        if rendered:
            blocks.append({"table": rendered})

    for sh in extracted.get("subheaders") or []:
        blocks.extend(_block_from_subheader(sh))

    return blocks


def parse_pages(spec: str) -> set[int]:
    """Parse '1,3,10-15,42' into a set of 1-indexed page numbers."""
    result: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            start, end = int(a), int(b)
            if start > end:
                start, end = end, start
            result.update(range(start, end + 1))
        else:
            result.add(int(chunk))
    return result


def build_slide_record(
    extracted: dict,
    page_num: int,
    slide_image_name: str,
    defaults: dict,
    doc_id: str,
    pre_extracted_tables: list[Table] = (),
) -> dict:
    fields: dict = {}
    for f in SLIDE_FIELDS:
        value = str(extracted.get(f, "") or "").strip()
        if not value and defaults.get(f):
            value = defaults[f]
        if value:
            fields[f] = value

    detail = _compose_detail(extracted, pre_extracted_tables)

    record: dict = {
        "slide_id": f"{doc_id}#{page_num:03d}",
        "doc_id": doc_id,
    }
    record.update(fields)
    if detail:
        record["detail"] = detail
    if slide_image_name:
        record["slide_image_path"] = slide_image_name
    return record


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a PDF or PPTX deck into per-slide JSON records for LLM retrieval.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--pdf",
        type=Path,
        help="Path to a PDF, or to a .zip containing one (extracted automatically).",
    )
    src.add_argument(
        "--pptx",
        type=Path,
        help="Path to a .pptx, or to a .zip containing one or more (extracted automatically).",
    )
    p.add_argument(
        "--pick",
        default="",
        help="With --pptx pointing at a multi-file .zip, substring of the .pptx filename to extract "
        "(case-insensitive; must match exactly one). Ignored when the zip has a single .pptx or "
        "when --pptx points at a bare .pptx.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/files"),
        help="Root directory for per-file folders (each holds slides.jsonl, chunks.jsonl, and screenshots).",
    )
    p.add_argument(
        "--corpus-out",
        type=Path,
        default=None,
        help="Path to the combined corpus file. Defaults to <output-dir>/../corpus/chunks.jsonl.",
    )
    p.add_argument(
        "--no-chunk",
        action="store_true",
        help="Skip building this deck's chunks.jsonl after extraction.",
    )
    p.add_argument(
        "--no-corpus",
        action="store_true",
        help="Skip rebuilding the combined corpus file after extraction.",
    )
    p.add_argument(
        "--no-images",
        action="store_true",
        help="Skip rendering per-slide PNGs. Useful for --pptx runs on machines "
        "without LibreOffice installed — you still get slides.jsonl/chunks.jsonl.",
    )
    p.add_argument(
        "--no-vision",
        action="store_true",
        help="Skip the vision-LLM fallback that reads text off of image-only "
        "pptx slides (slides with a picture and essentially no extractable text). "
        "Vision calls use the same model and API key as the text extraction.",
    )
    p.add_argument(
        "--diagnose-shapes",
        type=int,
        default=0,
        metavar="SLIDE_NUM",
        help="With --pptx: print every shape on slide N (1-indexed) with the "
        "signals the table extractor uses (has_table, descendant <a:tbl>, "
        "graphicData uri, OLE embed) and exit. Nothing is written to disk. "
        "Use this when a table isn't being detected to see exactly what "
        "python-pptx sees for that slide.",
    )
    p.add_argument("--product", default="", help="Fallback product/series when not visible on a slide.")
    p.add_argument("--dpi", type=int, default=150, help="Render DPI for slide screenshots.")
    p.add_argument("--limit", type=int, default=0, help="Only process the first N slides (0 = all).")
    p.add_argument(
        "--pages",
        default="",
        help="Specific slide numbers to process, e.g. '42' or '10-15,42,100-105'. Overrides --limit.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print records to stdout instead of writing slides.jsonl.",
    )
    return p.parse_args()


def _resolve_page_nums(total: int, pages: set[int] | None, limit: int) -> list[int]:
    if pages:
        return sorted(p for p in pages if 1 <= p <= total)
    nums = list(range(1, total + 1))
    return nums[:limit] if limit else nums


def _backfill_sections(records: list[dict]) -> None:
    """Slides without an explicit `section` inherit the last non-empty one seen.

    Section indicators sit in the top-left corner of nearly every slide in a
    guideline deck but the LLM sometimes misses them. Sections rarely change
    mid-deck, so carrying the last seen value forward fills the gaps correctly.
    """
    last = ""
    for r in records:
        current = r.get("section", "").strip()
        if current:
            last = current
        elif last:
            r["section"] = last


def _propagate_sections_from_intros(records: list[dict]) -> None:
    """PPTX section rule: each section starts with an intro slide whose sole
    purpose is naming a new section (e.g. a "Roadmap" or "March review" cover
    slide). Every subsequent slide belongs to that section until the next
    intro slide.

    Content slides ALWAYS take their `section` from the last-seen intro,
    overwriting whatever the LLM may have put there — the intro-driven
    pattern is authoritative and any per-slide LLM guess (often derived
    from a subtly-styled watermark or footer) is noise that would break
    the pattern.

    If a content slide precedes the first intro, its `section` is cleared
    so downstream doesn't see a stray value.
    """
    current = ""
    for r in records:
        if r.pop("is_section_intro", False):
            intro_section = r.get("section", "").strip()
            if intro_section:
                current = intro_section
        else:
            r.pop("is_section_intro", None)
            if current:
                r["section"] = current
            elif "section" in r:
                del r["section"]


_RECORD_KEY_ORDER = (
    "slide_id",
    "doc_id",
    "product",
    "section",
    "sub_section",
    "model",
    "detail",
    "slide_image_path",
)


def _reorder_record(record: dict) -> dict:
    """Return a new dict with keys in the canonical output order.

    Necessary because fields like `section` can be set (or overwritten) by
    post-processing after `build_slide_record` returned, which puts them at
    the END of the dict — dict insertion order is what the JSON writer emits.
    """
    ordered: dict = {}
    for k in _RECORD_KEY_ORDER:
        if k in record:
            ordered[k] = record[k]
    for k, v in record.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def _normalize_for_match(s: str) -> str:
    return " ".join(str(s).lower().split())


def _build_source_haystack(text_lines: list[TextLine]) -> str:
    """Normalized concatenation of every pdfminer text_line — the source-of-truth
    corpus for grounding checks."""
    return _normalize_for_match(" ".join(tl.text for tl in text_lines))


def _is_grounded(candidate: str, haystack: str, min_overlap: float = 0.6) -> bool:
    """True if enough of `candidate`'s meaningful words appear in the source.

    Anything with fewer than 3 meaningful words (3+ chars each) skips the check —
    short strings like "Galaxy S26" are too short to verify by overlap and are
    unlikely to be hallucinated anyway. Trivial short words (a, the, of, ...)
    don't count toward the overlap denominator.
    """
    words = [w for w in re.findall(r"\w+", (candidate or "").lower()) if len(w) >= 3]
    if len(words) < 3:
        return True
    hits = sum(1 for w in words if w in haystack)
    return hits / len(words) >= min_overlap


def _sanitize_subheader(sh: dict, haystack: str) -> dict | None:
    if not isinstance(sh, dict):
        return None
    title = str(sh.get("title", "") or "").strip()
    detail = str(sh.get("detail", "") or "").strip()
    if title and not _is_grounded(title, haystack):
        print(f"  ! dropped ungrounded subheader title: {title!r}")
        title = ""
    if detail and not _is_grounded(detail, haystack):
        print(f"  ! dropped ungrounded subheader body ({len(detail)} chars)")
        detail = ""
    children = [
        c for c in (_sanitize_subheader(x, haystack) for x in (sh.get("children") or [])) if c
    ]
    if not (title or detail or children):
        return None
    out: dict = {}
    if title:
        out["title"] = title
    if detail:
        out["detail"] = detail
    if children:
        out["children"] = children
    return out


def _sanitize_llm_output(extracted: dict, haystack: str) -> dict:
    """Drop any LLM-emitted string whose words aren't grounded in the pdfminer source.

    Slide-level fields fall back to empty (and then to CLI defaults where
    applicable). Subheader titles/bodies are individually cleared if
    ungrounded; empty subheaders are removed.
    """
    out = dict(extracted)
    for k in ("product", "codename", "sub_section", "model", "section"):
        v = str(out.get(k, "") or "").strip()
        if v and not _is_grounded(v, haystack):
            print(f"  ! dropped ungrounded {k}: {v!r}")
            out[k] = ""

    detail = str(out.get("detail", "") or "").strip()
    if detail and not _is_grounded(detail, haystack):
        print(f"  ! dropped ungrounded slide-level detail ({len(detail)} chars)")
        detail = ""
    out["detail"] = detail

    out["subheaders"] = [
        sh for sh in (_sanitize_subheader(s, haystack) for s in (out.get("subheaders") or [])) if sh
    ]
    return out


def _collect_block_text(block: dict, parts: list[str]) -> None:
    if not isinstance(block, dict):
        return
    for k in ("subheader", "body", "table"):
        v = block.get(k)
        if v:
            parts.append(str(v))
    for child in block.get("children") or []:
        _collect_block_text(child, parts)


def _collect_output_text(record: dict) -> str:
    """Concatenate every string this record emits, normalized for substring matching."""
    parts: list[str] = []
    for k in SLIDE_FIELDS:
        v = record.get(k)
        if v:
            parts.append(str(v))
    for block in record.get("detail") or []:
        _collect_block_text(block, parts)
    return _normalize_for_match(" ".join(parts))


def _is_page_chrome_bbox(bbox_pct: tuple[float, float, float, float]) -> bool:
    """Skip page-chrome text lines so the coverage check doesn't re-add them.

    Matches the PAGE CHROME rule in the prompt: top-right stamps and anything
    hugging the bottom edge. The top-left section indicator is deliberately
    NOT excluded — it should have been captured as `section`, and if it wasn't,
    _backfill_sections handles it before we get here.
    """
    x0, y0, x1, y1 = bbox_pct
    if y1 < 0.05 and x0 > 0.7:
        return True
    if y0 > 0.94:
        return True
    return False


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Coarse sentence split — good enough for dedupe. Keeps bullet fragments intact."""
    return [p.strip() for p in _SENTENCE_SPLIT.split(text.strip()) if p.strip()]


def _dedupe_block(block: dict, seen: set[str]) -> None:
    """Drop already-seen sentences from body and already-seen strings from table.

    Body dedupe is SENTENCE-level: the coverage-check often appends a full
    pdfminer paragraph when the LLM captured only its first sentence, so
    exact-body matches miss the duplication. Split on sentence terminators
    and drop any sentence whose normalized form appeared earlier.
    """
    if not isinstance(block, dict):
        return
    body = block.get("body")
    if body:
        kept: list[str] = []
        for sent in _split_sentences(body):
            norm = _normalize_for_match(sent)
            if len(norm) < 5:
                # too short to dedupe meaningfully; keep as-is without adding to seen
                kept.append(sent)
                continue
            if norm in seen:
                continue
            seen.add(norm)
            kept.append(sent)
        if kept:
            block["body"] = " ".join(kept)
        else:
            del block["body"]
    table = block.get("table")
    if table:
        norm = _normalize_for_match(table)
        if norm in seen:
            del block["table"]
        else:
            seen.add(norm)
    for child in block.get("children") or []:
        _dedupe_block(child, seen)


def _prune_empty_blocks(blocks: list[dict]) -> list[dict]:
    """Recursively remove blocks left empty after dedupe."""
    out: list[dict] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        children = _prune_empty_blocks(b.get("children") or [])
        if children:
            b["children"] = children
        else:
            b.pop("children", None)
        if b.get("subheader") or b.get("body") or b.get("table") or b.get("children"):
            out.append(b)
    return out


def _dedupe_detail(record: dict) -> None:
    """Ensure no body/table content appears twice in the record's detail tree."""
    seen: set[str] = set()
    for block in record.get("detail") or []:
        _dedupe_block(block, seen)
    record["detail"] = _prune_empty_blocks(record.get("detail") or [])
    if not record["detail"]:
        del record["detail"]


def _append_missing_text(record: dict, text_lines: list[TextLine]) -> None:
    """Append any pdfminer text_line whose content is missing from the record's output.

    Belt-and-suspenders for the prompt's COMPLETENESS RULE. Uses a case- and
    whitespace-insensitive substring check: if a text_line's content doesn't
    appear anywhere the LLM emitted, drop it verbatim into a fallback body
    block at the end of `detail`.
    """
    output = _collect_output_text(record)
    missing: list[str] = []
    for tl in text_lines:
        if _is_page_chrome_bbox(tl.bbox_pct):
            continue
        norm = _normalize_for_match(tl.text)
        if not norm or len(norm) < 3:
            continue
        if norm in output:
            continue
        missing.append(tl.text.strip())
    if not missing:
        return
    if "detail" not in record:
        record["detail"] = []
    record["detail"].append({"body": "\n".join(missing)})


def main() -> None:
    args = parse_args()

    input_path: Path = args.pptx if args.pptx is not None else args.pdf
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    # Which flag was used is authoritative — the file's extension is not.
    # A pptx-carrying zip is usually named "*.zip", not "*.pptx.zip", so
    # extension-sniffing was misrouting `--pptx some.zip` into the PDF path
    # and silently extracting the first .pdf inside.
    is_pptx = args.pptx is not None

    if is_pptx and args.diagnose_shapes:
        pptx_path = resolve_pptx_input(input_path, pick=args.pick)
        print(diagnose_pptx_slide_shapes(pptx_path, args.diagnose_shapes))
        return

    settings = get_settings()

    if is_pptx:
        pptx_path = resolve_pptx_input(input_path, pick=args.pick)
        source_stem = pptx_path.stem
        per_file_dir = args.output_dir / source_stem
        per_file_dir.mkdir(parents=True, exist_ok=True)
        # Companion PDF is only needed for rendering per-slide PNGs. Skip the
        # LibreOffice call entirely when --no-images is set so the pipeline
        # runs on machines without LibreOffice.
        render_pdf_path = None if args.no_images else pptx_to_pdf(pptx_path, pptx_path.parent)
        total_slides = pptx_slide_count(pptx_path)
        extract_layout = lambda n: extract_pptx_slide_layout(pptx_path, n)
        llm_kind = "pptx"
    else:
        pdf_path = resolve_pdf_input(input_path)
        source_stem = pdf_path.stem
        per_file_dir = args.output_dir / source_stem
        per_file_dir.mkdir(parents=True, exist_ok=True)
        render_pdf_path = pdf_path
        total_slides = page_count(pdf_path)
        extract_layout = lambda n: extract_page_layout(pdf_path, n)
        llm_kind = "pdf"

    pages = parse_pages(args.pages) if args.pages else None
    page_nums = _resolve_page_nums(total_slides, pages, args.limit)

    client = OpenAI(api_key=settings.openai_api_key)
    defaults = {"product": args.product}
    doc_id = _slug(source_stem)

    records: list[dict] = []
    text_lines_by_slide: dict[str, list[TextLine]] = {}
    for i, page_num in enumerate(page_nums, start=1):
        print(f"[extract] slide {i}/{len(page_nums)}: page {page_num}")
        try:
            layout = extract_layout(page_num)
            payload = build_payload(layout, page_num)
            data = extract_slide(client, settings.openai_model, payload, kind=llm_kind)
        except Exception as e:  # keep going even if one slide fails
            print(f"  ! extraction failed: {e}")
            continue

        is_section_intro = bool(data.get("is_section_intro", False)) if is_pptx else False
        # Geometric override for the LLM's is_section_intro claim. The LLM's
        # payload doesn't include tables at all, so a table-heavy slide with
        # just a title in text_lines looks "sparse" to it and gets mis-flagged
        # as an intro — which would then wipe the whole detail (table included)
        # and taint section-propagation for every following slide. Any of these
        # signals means we're on a real content slide, not an intro:
        #   - at least one detected table
        #   - more than 4 text-line paragraphs after excluding the sub_section
        if is_section_intro and is_pptx:
            sub_section_norm = _normalize_for_match(str(data.get("sub_section", "") or ""))
            non_title_lines = sum(
                1 for tl in layout.text_lines
                if _normalize_for_match(tl.text) and _normalize_for_match(tl.text) not in sub_section_norm
            )
            if layout.tables or non_title_lines > 4:
                print(
                    f"  ! overriding LLM is_section_intro=true on page {page_num}: "
                    f"{len(layout.tables)} table(s), {non_title_lines} non-title text lines"
                )
                is_section_intro = False
                data["is_section_intro"] = False
                # The prompt tells the LLM to put the intro title in `section`
                # and leave `sub_section` empty on intro slides. On a demoted
                # content slide that same title is the slide's own title, so
                # promote it to sub_section (only when sub_section is empty —
                # if the LLM already gave one, keep it).
                llm_section = str(data.get("section", "") or "").strip()
                llm_sub_section = str(data.get("sub_section", "") or "").strip()
                if llm_section and not llm_sub_section:
                    data["sub_section"] = llm_section
                # Section value only comes from real intro slides; clear the
                # LLM's guess so propagation doesn't carry it forward.
                data["section"] = ""

        data = _sanitize_llm_output(data, _build_source_haystack(layout.text_lines))

        if args.no_images or render_pdf_path is None:
            slide_image_name = ""  # signals build_slide_record to omit slide_image_path
        else:
            slide_image_name = f"slide_{page_num:03d}.png"
            rendered = render_page(render_pdf_path, page_num, dpi=args.dpi)
            rendered.save(per_file_dir / slide_image_name, format="PNG")

        record = build_slide_record(
            data,
            page_num=page_num,
            slide_image_name=slide_image_name,
            defaults=defaults,
            doc_id=doc_id,
            pre_extracted_tables=layout.tables,
        )
        if is_pptx:
            # Rebuild `detail` deterministically from the layout — the LLM
            # keeps missing within-shape subheader splits (e.g. KEY INITIATIVES
            # nested under Week 33-35) despite prompt guidance, and typography
            # + group_id give us everything we need to do this without asking.
            # Section-intro slides have no meaningful body — leave detail off.
            if is_section_intro:
                record.pop("detail", None)
                record["is_section_intro"] = True
            else:
                exclude_haystack = _normalize_for_match(record.get("sub_section", ""))
                notes = extract_pptx_slide_notes(pptx_path, page_num)
                new_detail = _build_detail_from_pptx_groups(
                    layout.text_lines, layout.tables, exclude_haystack,
                    slide_notes=notes,
                )
                # Vision fallback for image-only slides: the deterministic
                # builder needs typography (bold/underline/size) to work, so
                # a slide whose content is a single picture with no
                # meaningful text lines produces empty detail. Ask the LLM
                # to read the rendered PNG directly and give us sub_section
                # + body scraped from the image itself.
                _non_title_text_lines = sum(
                    1 for tl in layout.text_lines
                    if _normalize_for_match(tl.text)
                    and _normalize_for_match(tl.text) != exclude_haystack
                )
                needs_vision = (
                    not args.no_vision
                    and slide_image_name
                    and layout.figures
                    and _non_title_text_lines <= 1
                )
                if needs_vision:
                    image_path = per_file_dir / slide_image_name
                    try:
                        print(f"  [vision] image-only slide detected on page {page_num}; asking LLM")
                        vdata = extract_slide_from_image(
                            client, settings.openai_model, image_path
                        )
                    except Exception as e:
                        print(f"  ! vision fallback failed on page {page_num}: {e}")
                        vdata = {}
                    v_sub = str(vdata.get("sub_section", "") or "").strip()
                    v_body = str(vdata.get("body", "") or "").strip()
                    v_fig = str(vdata.get("figure_description", "") or "").strip()
                    # Only promote fields the deterministic path left empty
                    # so we don't overwrite a good sub_section from the LLM
                    # text pass with a vision guess.
                    if v_sub and not record.get("sub_section"):
                        record["sub_section"] = v_sub
                    v_blocks: list[dict] = []
                    if v_body:
                        v_blocks.append({"body": v_body})
                    if v_fig:
                        v_blocks.append({"subheader": "figure description", "body": v_fig})
                    if v_blocks:
                        new_detail = (new_detail or []) + v_blocks

                if new_detail:
                    record["detail"] = new_detail
                else:
                    record.pop("detail", None)
        records.append(record)
        text_lines_by_slide[record["slide_id"]] = layout.text_lines

    if is_pptx:
        _propagate_sections_from_intros(records)
    else:
        _backfill_sections(records)
    for record in records:
        tls = text_lines_by_slide.get(record["slide_id"], [])
        # PDF path only: the deterministic pptx builder already covers every
        # paragraph in the source, so the belt-and-suspenders "append what the
        # LLM dropped" step would just re-dump the same text.
        if not is_pptx:
            _append_missing_text(record, tls)
        _dedupe_detail(record)

    records = [_reorder_record(r) for r in records]

    if args.dry_run:
        for r in records:
            print(json.dumps(r, ensure_ascii=False))
        return

    out_path = per_file_dir / "slides.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
    print(f"[jsonl] wrote {len(records)} slide record(s) to {out_path}")

    if args.no_chunk:
        return
    chunk_file(per_file_dir)

    if args.no_corpus:
        return
    corpus_out = args.corpus_out or (args.output_dir.parent / "corpus" / "chunks.jsonl")
    build_corpus(args.output_dir, corpus_out)


if __name__ == "__main__":
    main()
