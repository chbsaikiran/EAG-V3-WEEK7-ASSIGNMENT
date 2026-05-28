"""rag_query.py — RAG query interface over indexed papers.

For each FAISS-matched chunk, the chunk immediately before and after it
(within the same source file) are also included in the LLM context.
This "context window expansion" prevents answers from being cut off at
chunk boundaries.

Usage:
    uv run rag_query.py "How can we fine-tune large models with less memory?"
    uv run rag_query.py "What is LoRA?" --no-index
    uv run rag_query.py "What is LoRA?" --compare
    uv run rag_query.py "What is LoRA?" --top-k 8
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from gateway import LLM, ensure_gateway
import memory
from schemas import MemoryItem

TOP_K = 5
MAX_CHUNK_CHARS = 600   # truncate each chunk before sending to keep prompt under ctx limit

SYSTEM = (
    "You are a research assistant. Answer the question using ONLY the provided context. "
    "Each context block shows a matched chunk plus its immediate neighbors from the same paper. "
    "Cite the source file for every claim (e.g. 'According to lora.md ...'). "
    "If the answer is not in the context, say so clearly. "
    "Be concise but complete."
)

NO_INDEX_SYSTEM = (
    "You are a research assistant. Answer the question from your general knowledge. "
    "Be concise but complete."
)


# ---------------------------------------------------------------------------
# Neighbor expansion
# ---------------------------------------------------------------------------

def _build_lookup(all_items: list[MemoryItem]) -> dict[tuple[str, int], MemoryItem]:
    """Index all fact items by (source, chunk_index) for O(1) neighbor lookup."""
    lookup: dict[tuple[str, int], MemoryItem] = {}
    for item in all_items:
        if item.kind != "fact":
            continue
        src = item.value.get("source") or item.source
        idx = item.value.get("chunk_index")
        if idx is not None:
            lookup[(src, int(idx))] = item
    return lookup


def _item_to_dict(item: MemoryItem, role: str) -> dict:
    return {
        "id":           item.id,
        "source":       item.value.get("source", item.source),
        "file":         item.value.get("file", item.source),
        "chunk_index":  item.value.get("chunk_index", 0),
        "total_chunks": item.value.get("total_chunks", "?"),
        "text":         (item.value.get("chunk", item.descriptor) or "")[:MAX_CHUNK_CHARS],
        "role":         role,   # "match" | "prev" | "next"
    }


def retrieve_with_neighbors(query: str, top_k: int) -> list[list[dict]]:
    """Return a list of groups. Each group is [prev?, match, next?] for one
    FAISS hit, deduplicated across all hits."""
    hits: list[MemoryItem] = memory.read(query, kinds=["fact"], top_k=top_k)
    if not hits:
        return []

    all_items   = memory._load()
    lookup      = _build_lookup(all_items)
    seen_ids: set[str] = set()
    groups: list[list[dict]] = []

    for hit in hits:
        if hit.id in seen_ids:
            continue

        src   = hit.value.get("source") or hit.source
        idx   = hit.value.get("chunk_index")
        total = hit.value.get("total_chunks", 0)

        group: list[dict] = []

        # prev chunk
        if idx is not None and idx > 0:
            prev = lookup.get((src, int(idx) - 1))
            if prev and prev.id not in seen_ids:
                seen_ids.add(prev.id)
                group.append(_item_to_dict(prev, "prev"))

        # matched chunk
        seen_ids.add(hit.id)
        group.append(_item_to_dict(hit, "match"))

        # next chunk
        if idx is not None and int(idx) + 1 < int(total):
            nxt = lookup.get((src, int(idx) + 1))
            if nxt and nxt.id not in seen_ids:
                seen_ids.add(nxt.id)
                group.append(_item_to_dict(nxt, "next"))

        groups.append(group)

    return groups


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

_ROLE_LABEL = {"prev": "↑ before", "match": "★ match ", "next": "↓ after "}


def build_context(groups: list[list[dict]]) -> str:
    parts = []
    for g_idx, group in enumerate(groups, 1):
        match_chunk = next(c for c in group if c["role"] == "match")
        header = (
            f"=== Result {g_idx}: {match_chunk['file']}  "
            f"chunk {match_chunk['chunk_index']+1}/{match_chunk['total_chunks']} ==="
        )
        lines = [header]
        for chunk in group:
            label = _ROLE_LABEL.get(chunk["role"], "       ")
            lines.append(f"\n[{label}]\n{chunk['text']}")
        parts.append("\n".join(lines))
    return "\n\n" + ("─" * 60 + "\n\n").join(parts)


# ---------------------------------------------------------------------------
# Query modes
# ---------------------------------------------------------------------------

def ask_with_index(query: str, top_k: int) -> None:
    print(f"\nQuery : {query}")
    print(f"Mode  : RAG with neighbor expansion  (top_k={top_k})\n")

    groups = retrieve_with_neighbors(query, top_k)
    if not groups:
        print("No chunks retrieved — index may be empty. Run index_corpus.py first.")
        return

    total_chunks = sum(len(g) for g in groups)
    print(f"Retrieved {len(groups)} matched chunks → expanded to {total_chunks} chunks (with neighbors):")
    for i, group in enumerate(groups, 1):
        match = next(c for c in group if c["role"] == "match")
        neighbor_note = f"  +{len(group)-1} neighbor(s)" if len(group) > 1 else ""
        preview = match["text"][:90].replace("\n", " ")
        print(f"  [{i}] {match['file']} chunk {match['chunk_index']+1}{neighbor_note} — {preview}…")
    print()

    context = build_context(groups)
    prompt  = f"Context:\n{context}\n\nQuestion: {query}"

    llm    = LLM()
    result = llm.chat(prompt=prompt, system=SYSTEM, max_tokens=1024, temperature=0.2,
                      provider="groq")

    print("Answer (with index + neighbors):")
    print(textwrap.fill(result["text"], width=90))
    print(f"\n[provider={result['provider']}  model={result['model']}  "
          f"in={result['input_tokens']} out={result['output_tokens']}]")


def ask_without_index(query: str) -> None:
    print(f"\nQuery : {query}")
    print("Mode  : NO INDEX (LLM only)\n")

    llm    = LLM()
    result = llm.chat(prompt=query, system=NO_INDEX_SYSTEM, max_tokens=1024,
                      temperature=0.2, provider="groq")

    print("Answer (no index):")
    print(textwrap.fill(result["text"], width=90))
    print(f"\n[provider={result['provider']}  model={result['model']}  "
          f"in={result['input_tokens']} out={result['output_tokens']}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG query over indexed papers")
    parser.add_argument("query",    nargs="?", help="Question to ask")
    parser.add_argument("--no-index",  action="store_true",
                        help="Skip retrieval — LLM answers from general knowledge only")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"Number of FAISS hits to retrieve (default: {TOP_K})")
    parser.add_argument("--compare", action="store_true",
                        help="Run both RAG and no-index modes for comparison")
    args = parser.parse_args()

    if not args.query:
        parser.print_help()
        sys.exit(1)

    ensure_gateway()

    if args.compare:
        ask_with_index(args.query, args.top_k)
        print("\n" + "=" * 90 + "\n")
        ask_without_index(args.query)
    elif args.no_index:
        ask_without_index(args.query)
    else:
        ask_with_index(args.query, args.top_k)


if __name__ == "__main__":
    main()
