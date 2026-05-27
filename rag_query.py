"""rag_query.py — RAG query interface over indexed papers.

Usage:
    uv run rag_query.py "How can we fine-tune large models with less memory?"
    uv run rag_query.py "What is chain of thought?" --no-index
    uv run rag_query.py "What is LoRA?" --top-k 8
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from gateway import LLM, ensure_gateway
import memory

TOP_K = 5

SYSTEM = (
    "You are a research assistant. Answer the question using ONLY the provided context chunks. "
    "If the answer is not in the context, say so clearly. "
    "Cite the source file for every claim (e.g. 'According to lora.md ...'). "
    "Be concise but complete."
)

NO_INDEX_SYSTEM = (
    "You are a research assistant. Answer the question from your general knowledge. "
    "Be concise but complete."
)


def retrieve(query: str, top_k: int) -> list[dict]:
    items = memory.read(query, kinds=["fact"], top_k=top_k)
    return [
        {
            "source": item.source,
            "file": item.value.get("file", item.source),
            "chunk_index": item.value.get("chunk_index", "?"),
            "total_chunks": item.value.get("total_chunks", "?"),
            "text": item.value.get("chunk", item.descriptor),
        }
        for item in items
    ]


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c['file']} (chunk {c['chunk_index']+1}/{c['total_chunks']})"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def ask_with_index(query: str, top_k: int) -> None:
    print(f"\nQuery : {query}")
    print(f"Mode  : RAG  (top_k={top_k})\n")

    chunks = retrieve(query, top_k)
    if not chunks:
        print("No chunks retrieved — index may be empty. Run index_corpus.py first.")
        return

    print(f"Retrieved {len(chunks)} chunks:")
    for i, c in enumerate(chunks, 1):
        preview = c["text"][:100].replace("\n", " ")
        print(f"  [{i}] {c['file']} chunk {c['chunk_index']+1} — {preview}...")
    print()

    context = build_context(chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}"

    llm = LLM()
    result = llm.chat(prompt=prompt, system=SYSTEM, max_tokens=1024, temperature=0.2,
                      provider="github")

    print("Answer (with index):")
    print(textwrap.fill(result["text"], width=90))
    print(f"\n[provider={result['provider']}  model={result['model']}  "
          f"in={result['input_tokens']} out={result['output_tokens']}]")


def ask_without_index(query: str) -> None:
    print(f"\nQuery : {query}")
    print("Mode  : NO INDEX (LLM only)\n")

    llm = LLM()
    result = llm.chat(prompt=query, system=NO_INDEX_SYSTEM, max_tokens=1024,
                      temperature=0.2, provider="github")

    print("Answer (no index):")
    print(textwrap.fill(result["text"], width=90))
    print(f"\n[provider={result['provider']}  model={result['model']}  "
          f"in={result['input_tokens']} out={result['output_tokens']}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG query over indexed papers")
    parser.add_argument("query", nargs="?", help="Question to ask")
    parser.add_argument("--no-index", action="store_true",
                        help="Skip retrieval — LLM answers from general knowledge only")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"Number of chunks to retrieve (default: {TOP_K})")
    parser.add_argument("--compare", action="store_true",
                        help="Run both modes side by side for comparison")
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
