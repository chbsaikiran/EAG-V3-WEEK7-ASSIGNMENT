"""fetch_papers.py — Fetch 45+ ML/AI papers from the arXiv API and save as
clean markdown files in sandbox/papers/.

Uses the arXiv Atom API (no key required). Each paper is saved as a tidy
markdown file: title, authors, date, abstract, and the arXiv link.
Existing files are skipped (idempotent — safe to re-run).

Usage:
    uv run fetch_papers.py
    uv run fetch_papers.py --dry-run     # print what would be fetched
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

PAPERS_DIR = Path(__file__).parent / "sandbox" / "papers"
ARXIV_API = "https://export.arxiv.org/api/query"
BATCH_SIZE = 20
DELAY_BETWEEN_BATCHES = 1.5  # seconds — be polite to arXiv

# ---------------------------------------------------------------------------
# Curated paper list — (arxiv_id, output_filename)
# Covers: Transformers, efficient fine-tuning, RLHF/alignment, reasoning,
# RAG/retrieval, agents/tools, multimodal. Excludes the 5 existing files.
# ---------------------------------------------------------------------------
PAPERS: list[tuple[str, str]] = [
    # ── Foundational language models ────────────────────────────────────────
    ("1810.04805", "bert.md"),                    # BERT
    ("2005.14165", "gpt3.md"),                    # GPT-3
    ("1910.10683", "t5.md"),                      # T5
    ("1907.11692", "roberta.md"),                 # RoBERTa
    ("2204.02311", "palm.md"),                    # PaLM
    ("2302.13971", "llama.md"),                   # LLaMA
    ("2307.09288", "llama2.md"),                  # LLaMA 2
    ("2310.06825", "mistral.md"),                 # Mistral 7B
    ("2401.04088", "mixtral.md"),                 # Mixtral MoE
    ("2312.00752", "mamba.md"),                   # Mamba (SSM)
    # ── Efficient training & fine-tuning ────────────────────────────────────
    ("2305.14314", "qlora.md"),                   # QLoRA
    ("1902.00751", "adapter.md"),                 # Adapter layers
    ("2101.00190", "prefix_tuning.md"),           # Prefix Tuning
    ("2205.14135", "flash_attention.md"),          # Flash Attention
    ("2307.08691", "flash_attention2.md"),         # Flash Attention 2
    ("2305.13245", "gqa.md"),                     # Grouped-Query Attention
    ("2309.12307", "longlora.md"),                # LongLoRA
    ("2104.09864", "rope.md"),                    # RoPE positional encoding
    ("2108.12409", "alibi.md"),                   # ALiBi positional encoding
    # ── RLHF & alignment ────────────────────────────────────────────────────
    ("2203.02155", "instructgpt.md"),             # InstructGPT / RLHF
    ("2009.01325", "learning_to_summarize.md"),   # Learning to Summarize (RLHF)
    ("2212.08073", "constitutional_ai.md"),       # Constitutional AI
    ("1707.06347", "ppo.md"),                     # PPO (RL algorithm used in RLHF)
    # ── Reasoning & prompting ───────────────────────────────────────────────
    ("2203.11171", "self_consistency.md"),        # Self-Consistency CoT
    ("2205.10625", "least_to_most.md"),           # Least-to-Most Prompting
    ("2305.10601", "tree_of_thoughts.md"),        # Tree of Thoughts
    ("2211.10435", "pal.md"),                     # PAL — Program-Aided Language Models
    ("2205.11916", "zero_shot_cot.md"),           # Zero-shot CoT ("Let's think step by step")
    ("2303.11366", "reflexion.md"),               # Reflexion
    ("2210.06726", "self_ask.md"),                # Self-Ask
    # ── RAG & retrieval ─────────────────────────────────────────────────────
    ("2005.11401", "rag.md"),                     # RAG (Lewis et al.)
    ("2002.08909", "realm.md"),                   # REALM
    ("2004.04906", "dpr.md"),                     # Dense Passage Retrieval
    ("2004.12832", "colbert.md"),                 # ColBERT
    ("2212.10496", "hyde.md"),                    # HyDE (hypothetical document embeddings)
    # ── Agents & tool use ───────────────────────────────────────────────────
    ("2302.04761", "toolformer.md"),              # Toolformer
    ("2303.17580", "hugginggpt.md"),              # HuggingGPT
    ("2305.16291", "voyager.md"),                 # Voyager (MC agent)
    ("2308.03688", "agentbench.md"),              # AgentBench
    ("2310.06770", "swe_bench.md"),               # SWE-bench
    ("2210.11610", "text2reward.md"),             # Generative Agents
    # ── Multimodal ──────────────────────────────────────────────────────────
    ("2103.00020", "clip.md"),                    # CLIP
    ("2204.06125", "dalle2.md"),                  # DALL-E 2
    ("2204.14198", "flamingo.md"),                # Flamingo
    ("2304.08485", "llava.md"),                   # LLaVA
    # ── Evaluation & benchmarks ─────────────────────────────────────────────
    ("2009.03300", "superglue.md"),               # SuperGLUE
    ("2110.14168", "truthfulqa.md"),              # TruthfulQA
    ("2212.09251", "gsm8k.md"),                   # GSM8K math benchmark
]


NS = {"atom": "http://www.w3.org/2005/Atom"}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60]


def _parse_entries(xml_text: str) -> dict[str, dict]:
    root = ET.fromstring(xml_text)
    out: dict[str, dict] = {}
    for entry in root.findall("atom:entry", NS):
        id_el = entry.find("atom:id", NS)
        if id_el is None:
            continue
        raw_id = id_el.text or ""
        arxiv_id = raw_id.split("/abs/")[-1].split("v")[0].strip()

        title = (entry.find("atom:title", NS).text or "").strip().replace("\n", " ")
        summary = (entry.find("atom:summary", NS).text or "").strip()
        published = (entry.find("atom:published", NS).text or "")[:10]
        authors = [
            (a.find("atom:name", NS).text or "").strip()
            for a in entry.findall("atom:author", NS)
        ]
        out[arxiv_id] = {
            "title": title,
            "authors": authors,
            "published": published,
            "abstract": summary,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
        }
    return out


def _to_markdown(paper: dict, arxiv_id: str) -> str:
    authors_str = ", ".join(paper["authors"][:6])
    if len(paper["authors"]) > 6:
        authors_str += f" et al. (+{len(paper['authors'])-6} more)"
    return (
        f"# {paper['title']}\n\n"
        f"**arXiv:** {arxiv_id}  \n"
        f"**Authors:** {authors_str}  \n"
        f"**Published:** {paper['published']}  \n"
        f"**URL:** {paper['url']}\n\n"
        f"## Abstract\n\n{paper['abstract']}\n"
    )


def fetch_batch(ids: list[str]) -> dict[str, dict]:
    url = f"{ARXIV_API}?id_list={','.join(ids)}&max_results={len(ids)}"
    r = httpx.get(url, timeout=30, follow_redirects=True)
    r.raise_for_status()
    return _parse_entries(r.text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch arXiv papers for RAG corpus")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be fetched without downloading")
    args = parser.parse_args()

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    to_fetch = [(aid, fname) for aid, fname in PAPERS
                if not (PAPERS_DIR / fname).exists()]

    if not to_fetch:
        print("All papers already downloaded.")
        return

    print(f"Papers to fetch : {len(to_fetch)}")
    print(f"Already present : {len(PAPERS) - len(to_fetch)}")

    if args.dry_run:
        for aid, fname in to_fetch:
            print(f"  would fetch  {aid}  →  {fname}")
        return

    saved = 0
    failed: list[str] = []

    for batch_start in range(0, len(to_fetch), BATCH_SIZE):
        batch = to_fetch[batch_start : batch_start + BATCH_SIZE]
        ids = [aid for aid, _ in batch]
        fname_map = {aid: fname for aid, fname in batch}

        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(to_fetch) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\nBatch {batch_num}/{total_batches}  ({len(batch)} papers) …")

        try:
            entries = fetch_batch(ids)
        except Exception as e:
            print(f"  [error] batch request failed: {e}")
            failed.extend(ids)
            continue

        for aid in ids:
            fname = fname_map[aid]
            dest = PAPERS_DIR / fname
            if aid not in entries:
                print(f"  [skip]  {aid} — not returned by arXiv API")
                failed.append(aid)
                continue
            paper = entries[aid]
            md = _to_markdown(paper, aid)
            dest.write_text(md, encoding="utf-8")
            saved += 1
            print(f"  ✓  {fname:<35}  {paper['title'][:55]}")

        if batch_start + BATCH_SIZE < len(to_fetch):
            time.sleep(DELAY_BETWEEN_BATCHES)

    print(f"\n{'─'*60}")
    print(f"Saved  : {saved} new papers")
    print(f"Skipped: {len(PAPERS) - len(to_fetch)} (already existed)")
    if failed:
        print(f"Failed : {len(failed)}  —  {failed}")
    total = len(list(PAPERS_DIR.glob('*.md')))
    print(f"Total in sandbox/papers/ : {total} files")


if __name__ == "__main__":
    main()
