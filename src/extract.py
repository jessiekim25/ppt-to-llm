import argparse
import json
import re
from pathlib import Path

from openai import OpenAI

from shared.settings import get_settings

from .chunk import chunk_file
from .corpus import build_corpus
from .llm import build_payload, extract_slide
from .pdf_layout import Table, TextLine, extract_page_layout
from .pdf_utils import page_count, render_page, resolve_pdf_input
from .pptx_layout import extract_slide_layout as extract_pptx_slide_layout
from .pptx_layout import slide_count as pptx_slide_count
from .pptx_utils import pptx_to_pdf, resolve_pptx_input

SLIDE_FIELDS = ("product", "section", "sub_section", "model")


def _is_pptx_input(path: Path) -> bool:
    """True if `path` points at a .pptx (directly or wrapped in a .zip named *.pptx.zip)."""
    name = path.name.lower()
    if name.endswith(".pptx"):
        return True
    if name.endswith(".pptx.zip"):
        return True
    return False


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


def _is_valid_subheader_title(title: str) -> bool:
    """Enforce the prompt's (α) rule: <=10 words, no sentence-terminal punctuation.

    A trailing colon is fine ("How to build layout:" is a real subheader).
    A trailing period/question/exclamation mark means it's a sentence, not a label.
    """
    t = title.rstrip()
    if not t:
        return False
    if t[-1] in ".?!":
        return False
    if len(t.split()) > 10:
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
        help="Path to a .pptx, or to a .zip containing one (extracted automatically).",
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
    intro slide. The LLM tags intro slides with `is_section_intro: true` and
    puts the section title in `section`; content slides leave `section`
    empty and inherit it here.
    """
    current = ""
    for r in records:
        if r.pop("is_section_intro", False):
            intro_section = r.get("section", "").strip()
            if intro_section:
                current = intro_section
        else:
            r.pop("is_section_intro", None)
            if not r.get("section", "").strip() and current:
                r["section"] = current


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

    is_pptx = _is_pptx_input(input_path)
    settings = get_settings()

    if is_pptx:
        pptx_path = resolve_pptx_input(input_path)
        source_stem = pptx_path.stem
        per_file_dir = args.output_dir / source_stem
        per_file_dir.mkdir(parents=True, exist_ok=True)
        # Companion PDF lives beside the pptx, not inside output/. Rendered
        # PNGs are the artifact; the intermediate PDF is a cache.
        render_pdf_path = pptx_to_pdf(pptx_path, pptx_path.parent)
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
        data = _sanitize_llm_output(data, _build_source_haystack(layout.text_lines))

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
        if is_pptx and is_section_intro:
            record["is_section_intro"] = True
        records.append(record)
        text_lines_by_slide[record["slide_id"]] = layout.text_lines

    if is_pptx:
        _propagate_sections_from_intros(records)
    else:
        _backfill_sections(records)
    for record in records:
        tls = text_lines_by_slide.get(record["slide_id"], [])
        _append_missing_text(record, tls)
        _dedupe_detail(record)

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
