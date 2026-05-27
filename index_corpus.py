"""index_corpus.py — Index all markdown files in sandbox/papers/ into FAISS memory.

Run once (or re-run to refresh):
    uv run index_corpus.py
    uv run index_corpus.py --clear      # wipe index first, then re-index
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from gateway import ensure_gateway
import memory

PAPERS_DIR = Path(__file__).parent / "sandbox" / "papers"
CHUNK_SIZE = 400
OVERLAP = 80


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    stride = max(1, size - overlap)
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + size]))
        if i + size >= len(words):
            break
        i += stride
    return chunks


def index_file(path: Path, run_id: str) -> int:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        print(f"  [skip] {path.name} — empty")
        return 0
    chunks = _chunk_text(text)
    source = f"papers:{path.name}"
    for i, chunk in enumerate(chunks):
        preview = chunk[:80].replace("\n", " ")
        descriptor = f"[{source} chunk {i+1}/{len(chunks)}] {preview}"
        memory.add_fact(
            descriptor=descriptor,
            value={
                "chunk": chunk,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source": source,
                "file": path.name,
            },
            keywords=[path.stem.lower()] + [w.lower() for w in path.stem.split("_")],
            source=source,
            run_id=run_id,
        )
    return len(chunks)


def main() -> None:
    if "--clear" in sys.argv:
        print("Clearing existing memory and index...")
        memory.clear()
        print("Cleared.\n")

    ensure_gateway()

    papers = sorted(PAPERS_DIR.glob("*.md"))
    if not papers:
        print(f"No .md files found in {PAPERS_DIR}")
        return

    run_id = f"index-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    print(f"Indexing {len(papers)} papers  (run_id={run_id})\n")

    total_chunks = 0
    for path in papers:
        n = index_file(path, run_id)
        total_chunks += n
        print(f"  ✓  {path.name:<30}  {n} chunks")

    print(f"\nDone — {len(papers)} files, {total_chunks} chunks in FAISS index.")


if __name__ == "__main__":
    main()
