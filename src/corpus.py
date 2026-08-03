"""Concatenate every deck's chunks.jsonl into one corpus file for downstream RAG.

Layout produced:

    output/
      decks/
        <doc_id>/
          slides.jsonl
          chunks.jsonl              # authoritative per-deck
          slide_NNN.png
      corpus/
        chunks.jsonl                # concatenation of every deck's chunks.jsonl
        manifest.json               # {doc_id: {chunk_count, sha256, built_at}}

The corpus file is a derived artifact — safe to delete and rebuild anytime. It's
also the single hand-off surface for the downstream RAG agent.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _iter_deck_chunks_files(decks_dir: Path):
    for child in sorted(decks_dir.iterdir()):
        if not child.is_dir():
            continue
        chunks_path = child / "chunks.jsonl"
        if chunks_path.exists():
            yield child.name, chunks_path


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def build_corpus(decks_dir: Path, out_path: Path) -> tuple[int, int]:
    """Rewrite the corpus file from all deck chunks.jsonl. Returns (deck_count, chunk_count)."""
    if not decks_dir.exists():
        raise FileNotFoundError(f"Decks dir not found: {decks_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path.parent / "manifest.json"

    total_chunks = 0
    manifest: dict[str, dict] = {}
    built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as dst:
        for doc_id, chunks_path in _iter_deck_chunks_files(decks_dir):
            deck_count = 0
            with chunks_path.open("r", encoding="utf-8") as src:
                for line in src:
                    if not line.strip():
                        continue
                    dst.write(line if line.endswith("\n") else line + "\n")
                    deck_count += 1
            manifest[doc_id] = {
                "chunk_count": deck_count,
                "sha256": _hash_file(chunks_path),
                "built_at": built_at,
                "source": str(chunks_path.relative_to(decks_dir.parent))
                if decks_dir.parent in chunks_path.parents
                else str(chunks_path),
            }
            total_chunks += deck_count

    tmp_path.replace(out_path)

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(
        f"[corpus] wrote {total_chunks} chunk(s) from {len(manifest)} deck(s) "
        f"to {out_path}; manifest at {manifest_path}"
    )
    return len(manifest), total_chunks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Concatenate all per-deck chunks.jsonl into one corpus file.",
    )
    p.add_argument(
        "--decks-dir",
        type=Path,
        default=Path("output/decks"),
        help="Root directory holding per-deck folders (each with chunks.jsonl).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("output/corpus/chunks.jsonl"),
        help="Path to the combined corpus file.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_corpus(args.decks_dir, args.out)


if __name__ == "__main__":
    main()
