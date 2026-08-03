"""Turn per-deck slides.jsonl into per-deck chunks.jsonl for downstream RAG.

Each output row is one slide, retrieval-shaped:

    {
      "slide_id":   "<doc>#NNN",
      "doc_id":     "<doc>",
      "slide_num":  42,
      "image_path": "decks/<doc>/slide_042.png",   # relative to the corpus root
      "embed_text": "<flattened prose the embedder consumes verbatim>",
      "metadata":   { flat scalars for filtering }
    }

The flatten is deterministic — no LLM call, no vision captioning. Slides with
almost no text still emit a row (with is_visual_only=true) so downstream sees
them and can decide what to do.
"""

import argparse
import json
from pathlib import Path

# Slides with fewer than this many characters of flattened text are flagged
# is_visual_only so downstream can decide whether to caption them.
_VISUAL_ONLY_THRESHOLD = 40

# Long tables get truncated in embed_text so one giant reference table can't
# dominate a slide's embedding. Head + tail of rows are kept; the middle is
# elided with a row-count marker.
_TABLE_ROW_HEAD = 10
_TABLE_ROW_TAIL = 10
_TABLE_ROW_MAX = _TABLE_ROW_HEAD + _TABLE_ROW_TAIL


def _flatten_table(rendered: str) -> str:
    """Convert 'title\\ncol1 | col2\\ncell11 | cell12' into embedder-friendly prose.

    A table in a slide record is one string with pipe-delimited columns. For
    embedding we want it as sentences — "Table: {title}. {col1} — {cell11}, {col2} — {cell12}."
    — so the embedding text reads naturally instead of as a table dump.
    """
    lines = [ln.strip() for ln in rendered.splitlines() if ln.strip()]
    if not lines:
        return ""

    # A leading line without pipes is the title.
    title = ""
    if "|" not in lines[0]:
        title = lines[0]
        lines = lines[1:]

    if not lines:
        return f"Table: {title}." if title else ""

    header = [c.strip() for c in lines[0].split("|")]
    body_rows = [[c.strip() for c in ln.split("|")] for ln in lines[1:]]

    if len(body_rows) > _TABLE_ROW_MAX:
        head = body_rows[:_TABLE_ROW_HEAD]
        tail = body_rows[-_TABLE_ROW_TAIL:]
        elided = len(body_rows) - _TABLE_ROW_HEAD - _TABLE_ROW_TAIL
        kept = head + [["…", f"({elided} more rows)"]] + tail
    else:
        kept = body_rows

    row_sentences: list[str] = []
    for row in kept:
        pairs = []
        for i, cell in enumerate(row):
            if not cell:
                continue
            col = header[i] if i < len(header) else ""
            pairs.append(f"{col} — {cell}" if col else cell)
        if pairs:
            row_sentences.append(", ".join(pairs) + ".")

    parts: list[str] = []
    if title:
        parts.append(f"Table: {title}.")
    parts.extend(row_sentences)
    return " ".join(parts)


def _walk_blocks(blocks: list, out: list[str], parent_label: str = "") -> None:
    """Depth-first walk of the detail tree, appending sentences to `out`.

    Subheaders are emitted as "Label: " prefixes so a body owned by a
    subheader keeps its context in the embedding text. `parent_label` is the
    joined-with-slashes chain of ancestor subheaders (no trailing separator),
    empty at the top level.
    """
    for b in blocks or []:
        if not isinstance(b, dict):
            continue
        sub = str(b.get("subheader", "") or "").strip()
        body = str(b.get("body", "") or "").strip()
        table = str(b.get("table", "") or "").strip()
        children = b.get("children") or []

        if sub and parent_label:
            label = f"{parent_label} / {sub}"
        elif sub:
            label = sub
        else:
            label = parent_label

        if body:
            out.append(f"{label}: {body}" if label else body)
        elif sub and not children and not table:
            out.append(f"{label}.")

        if table:
            flat = _flatten_table(table)
            if flat:
                out.append(f"{label}: {flat}" if label else flat)

        if children:
            _walk_blocks(children, out, label)


def _build_embed_text(slide: dict) -> str:
    """Deterministic flatten: '{product} {codename} — {section} / {sub_section}. <detail sentences>.'"""
    product = str(slide.get("product", "") or "").strip()
    codename = str(slide.get("codename", "") or "").strip()
    model = str(slide.get("model", "") or "").strip()
    section = str(slide.get("section", "") or "").strip()
    sub_section = str(slide.get("sub_section", "") or "").strip()

    header_bits: list[str] = []
    if product or codename:
        header_bits.append(" ".join(x for x in (product, codename) if x))
    if model and model != product:
        header_bits.append(f"({model})")
    section_bits = [x for x in (section, sub_section) if x]
    if section_bits:
        prefix = " ".join(header_bits) + " — " if header_bits else ""
        header = f"{prefix}{' / '.join(section_bits)}."
    elif header_bits:
        header = " ".join(header_bits) + "."
    else:
        header = ""

    sentences: list[str] = []
    _walk_blocks(slide.get("detail") or [], sentences)

    parts = [header] if header else []
    parts.extend(s.rstrip(".") + "." for s in sentences if s.strip())
    return " ".join(parts).strip()


def _has_any(blocks: list, key: str) -> bool:
    for b in blocks or []:
        if not isinstance(b, dict):
            continue
        if b.get(key):
            return True
        if _has_any(b.get("children") or [], key):
            return True
    return False


def slide_to_chunk(slide: dict, doc_id: str, image_dir: str | None = None) -> dict:
    """Return the retrieval row for one slide record.

    `image_dir` is the on-disk folder name under decks/ that holds this slide's
    PNG (may differ from doc_id when the source filename slugged differently).
    """
    slide_num = int(slide.get("slide_num", 0))
    slide_id = slide.get("slide_id") or f"{doc_id}#{slide_num:03d}"
    image_basename = str(slide.get("slide_image_path", "") or "").strip()
    folder = image_dir if image_dir else doc_id
    image_path = f"decks/{folder}/{image_basename}" if image_basename else ""

    embed_text = _build_embed_text(slide)
    detail = slide.get("detail") or []

    metadata: dict = {
        "product": str(slide.get("product", "") or "").strip(),
        "codename": str(slide.get("codename", "") or "").strip(),
        "model": str(slide.get("model", "") or "").strip(),
        "section": str(slide.get("section", "") or "").strip(),
        "sub_section": str(slide.get("sub_section", "") or "").strip(),
        "has_table": _has_any(detail, "table"),
        "has_subheaders": _has_any(detail, "subheader"),
        "text_len": len(embed_text),
        "is_visual_only": len(embed_text) < _VISUAL_ONLY_THRESHOLD,
    }
    # Drop empty scalar fields so metadata stays sparse.
    metadata = {k: v for k, v in metadata.items() if v not in ("", None)}

    return {
        "slide_id": slide_id,
        "doc_id": doc_id,
        "slide_num": slide_num,
        "image_path": image_path,
        "embed_text": embed_text,
        "metadata": metadata,
    }


def chunk_deck(deck_dir: Path) -> Path:
    """Read <deck>/slides.jsonl and write <deck>/chunks.jsonl beside it. Returns the output path.

    doc_id comes from each slide record (authoritative — extract slugs it),
    not the folder name. The image_path we emit uses the folder name so it
    resolves against the actual filesystem.
    """
    slides_path = deck_dir / "slides.jsonl"
    if not slides_path.exists():
        raise FileNotFoundError(f"slides.jsonl not found: {slides_path}")

    deck_name = deck_dir.name
    out_path = deck_dir / "chunks.jsonl"
    count = 0
    with slides_path.open("r", encoding="utf-8") as src, out_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            slide = json.loads(line)
            doc_id = str(slide.get("doc_id", "") or "").strip() or deck_name
            chunk = slide_to_chunk(slide, doc_id, image_dir=deck_name)
            dst.write(json.dumps(chunk, ensure_ascii=False))
            dst.write("\n")
            count += 1
    print(f"[chunk] wrote {count} chunk(s) to {out_path}")
    return out_path


def _iter_deck_dirs(root: Path):
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "slides.jsonl").exists():
            yield child


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Flatten per-deck slides.jsonl into retrieval-ready chunks.jsonl.",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--deck", type=Path, help="Path to a single deck directory containing slides.jsonl.")
    g.add_argument(
        "--all",
        dest="all_decks",
        action="store_true",
        help="Re-chunk every deck under --decks-dir.",
    )
    p.add_argument(
        "--decks-dir",
        type=Path,
        default=Path("output/decks"),
        help="Root directory holding per-deck folders (only used with --all).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.all_decks:
        if not args.decks_dir.exists():
            raise SystemExit(f"Decks dir not found: {args.decks_dir}")
        decks = list(_iter_deck_dirs(args.decks_dir))
        if not decks:
            print(f"[chunk] no decks with slides.jsonl found under {args.decks_dir}")
            return
        for deck in decks:
            chunk_deck(deck)
    else:
        chunk_deck(args.deck)


if __name__ == "__main__":
    main()
