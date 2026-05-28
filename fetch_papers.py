"""fetch_papers.py — Fetch 45+ ML/AI papers with FULL TEXT and save as
clean markdown files in sandbox/papers/.

Metadata (title, authors, abstract) comes from the arXiv Atom API.
Full body text comes from ar5iv (https://ar5iv.labs.arxiv.org/html/{id}),
which renders the LaTeX source as HTML. Falls back to arxiv's own HTML
renderer, then to abstract-only if both fail.

No extra dependencies — uses httpx (already in project) and stdlib html.parser.
Existing files are skipped (idempotent — safe to re-run).

Usage:
    uv run fetch_papers.py
    uv run fetch_papers.py --dry-run        # list what would be fetched
    uv run fetch_papers.py --refetch        # overwrite existing files too
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

import httpx

PAPERS_DIR = Path(__file__).parent / "sandbox" / "papers"
ARXIV_API   = "https://export.arxiv.org/api/query"
AR5IV_URL   = "https://ar5iv.labs.arxiv.org/html/{id}"
ARXIV_HTML  = "https://arxiv.org/html/{id}"

BATCH_SIZE             = 20
DELAY_BETWEEN_BATCHES  = 1.5   # seconds — polite to arXiv
FULL_TEXT_DELAY        = 0.5   # seconds between full-text fetches
MAX_FULL_TEXT_CHARS    = 120_000  # ~30k tokens; enough for any paper

# ---------------------------------------------------------------------------
# Paper list — (arxiv_id, output_filename)
# ---------------------------------------------------------------------------
PAPERS: list[tuple[str, str]] = [
    # ── Foundational language models ────────────────────────────────────────
    ("1810.04805", "bert.md"),
    ("2005.14165", "gpt3.md"),
    ("1910.10683", "t5.md"),
    ("1907.11692", "roberta.md"),
    ("2204.02311", "palm.md"),
    ("2302.13971", "llama.md"),
    ("2307.09288", "llama2.md"),
    ("2310.06825", "mistral.md"),
    ("2401.04088", "mixtral.md"),
    ("2312.00752", "mamba.md"),
    # ── Efficient training & fine-tuning ────────────────────────────────────
    ("2305.14314", "qlora.md"),
    ("1902.00751", "adapter.md"),
    ("2101.00190", "prefix_tuning.md"),
    ("2205.14135", "flash_attention.md"),
    ("2307.08691", "flash_attention2.md"),
    ("2305.13245", "gqa.md"),
    ("2309.12307", "longlora.md"),
    ("2104.09864", "rope.md"),
    ("2108.12409", "alibi.md"),
    # ── RLHF & alignment ────────────────────────────────────────────────────
    ("2203.02155", "instructgpt.md"),
    ("2009.01325", "learning_to_summarize.md"),
    ("2212.08073", "constitutional_ai.md"),
    ("1707.06347", "ppo.md"),
    # ── Reasoning & prompting ───────────────────────────────────────────────
    ("2203.11171", "self_consistency.md"),
    ("2205.10625", "least_to_most.md"),
    ("2305.10601", "tree_of_thoughts.md"),
    ("2211.10435", "pal.md"),
    ("2205.11916", "zero_shot_cot.md"),
    ("2303.11366", "reflexion.md"),
    ("2210.06726", "self_ask.md"),
    # ── RAG & retrieval ─────────────────────────────────────────────────────
    ("2005.11401", "rag.md"),
    ("2002.08909", "realm.md"),
    ("2004.04906", "dpr.md"),
    ("2004.12832", "colbert.md"),
    ("2212.10496", "hyde.md"),
    # ── Agents & tool use ───────────────────────────────────────────────────
    ("2302.04761", "toolformer.md"),
    ("2303.17580", "hugginggpt.md"),
    ("2305.16291", "voyager.md"),
    ("2308.03688", "agentbench.md"),
    ("2310.06770", "swe_bench.md"),
    ("2210.11610", "generative_agents.md"),
    # ── Multimodal ──────────────────────────────────────────────────────────
    ("2103.00020", "clip.md"),
    ("2204.06125", "dalle2.md"),
    ("2204.14198", "flamingo.md"),
    ("2304.08485", "llava.md"),
    # ── Evaluation & benchmarks ─────────────────────────────────────────────
    ("2009.03300", "superglue.md"),
    ("2110.14168", "truthfulqa.md"),
    ("2212.09251", "gsm8k.md"),
]


# ---------------------------------------------------------------------------
# HTML → plain text (stdlib only, no BeautifulSoup)
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Strip HTML tags; preserve paragraph breaks; skip boilerplate sections."""

    # Tags whose entire subtree we discard
    SKIP_TAGS = {
        "script", "style", "nav", "header", "footer", "aside",
        "figure", "figcaption", "math", "svg", "cite",
        "noscript", "button", "form", "input", "select",
    }
    # Tags that represent block boundaries (emit a newline)
    BLOCK_TAGS = {
        "p", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "br", "tr", "td", "th", "div", "section",
        "article", "blockquote", "pre", "code",
    }

    def __init__(self) -> None:
        super().__init__()
        self._buf: list[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, _attrs) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        if not self._skip_depth and tag in self.BLOCK_TAGS:
            self._buf.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        if not self._skip_depth and tag in self.BLOCK_TAGS:
            self._buf.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._buf.append(data)

    def get_text(self) -> str:
        text = "".join(self._buf)
        text = re.sub(r"[ \t]+", " ", text)           # collapse spaces
        text = re.sub(r"\n[ \t]+", "\n", text)         # trim leading space on lines
        text = re.sub(r"\n{3,}", "\n\n", text)         # max two consecutive newlines
        return text.strip()


def _html_to_text(html: str) -> str:
    p = _TextExtractor()
    p.feed(html)
    return p.get_text()


# ---------------------------------------------------------------------------
# arXiv Atom API — metadata
# ---------------------------------------------------------------------------

NS = {"atom": "http://www.w3.org/2005/Atom"}


def _parse_entries(xml_text: str) -> dict[str, dict]:
    root = ET.fromstring(xml_text)
    out: dict[str, dict] = {}
    for entry in root.findall("atom:entry", NS):
        id_el = entry.find("atom:id", NS)
        if id_el is None:
            continue
        raw_id = (id_el.text or "")
        arxiv_id = raw_id.split("/abs/")[-1].split("v")[0].strip()
        title    = (entry.find("atom:title",   NS).text or "").strip().replace("\n", " ")
        summary  = (entry.find("atom:summary", NS).text or "").strip()
        published = (entry.find("atom:published", NS).text or "")[:10]
        authors  = [(a.find("atom:name", NS).text or "").strip()
                    for a in entry.findall("atom:author", NS)]
        out[arxiv_id] = {
            "title": title, "authors": authors,
            "published": published, "abstract": summary,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
        }
    return out


def fetch_metadata_batch(ids: list[str]) -> dict[str, dict]:
    url = f"{ARXIV_API}?id_list={','.join(ids)}&max_results={len(ids)}"
    r = httpx.get(url, timeout=30, follow_redirects=True)
    r.raise_for_status()
    return _parse_entries(r.text)


# ---------------------------------------------------------------------------
# Full text — ar5iv → arxiv HTML → abstract-only fallback
# ---------------------------------------------------------------------------

def _fetch_html(arxiv_id: str) -> tuple[str, str]:
    """Return (plain_text, source_label). Tries ar5iv first, then arxiv HTML."""
    for url_tpl, label in [
        (AR5IV_URL,  "ar5iv"),
        (ARXIV_HTML, "arxiv-html"),
    ]:
        url = url_tpl.format(id=arxiv_id)
        try:
            r = httpx.get(url, timeout=40, follow_redirects=True)
            if r.status_code == 200 and len(r.text) > 2000:
                text = _html_to_text(r.text)
                # Trim to cap — keeps the file manageable for chunking
                if len(text) > MAX_FULL_TEXT_CHARS:
                    text = text[:MAX_FULL_TEXT_CHARS] + "\n\n[… truncated at 120 000 chars …]"
                return text, label
        except Exception:
            continue
    return "", "none"


# ---------------------------------------------------------------------------
# Markdown assembly
# ---------------------------------------------------------------------------

def _to_markdown(paper: dict, arxiv_id: str, full_text: str, source: str) -> str:
    authors_str = ", ".join(paper["authors"][:6])
    if len(paper["authors"]) > 6:
        authors_str += f" et al. (+{len(paper['authors'])-6} more)"

    header = (
        f"# {paper['title']}\n\n"
        f"**arXiv:** {arxiv_id}  \n"
        f"**Authors:** {authors_str}  \n"
        f"**Published:** {paper['published']}  \n"
        f"**URL:** {paper['url']}\n\n"
        f"## Abstract\n\n{paper['abstract']}\n\n"
    )

    if full_text:
        body = f"## Full Text  _(source: {source})_\n\n{full_text}\n"
    else:
        body = "_Full text unavailable — abstract only._\n"

    return header + body


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch arXiv papers (full text) for RAG corpus")
    parser.add_argument("--dry-run",  action="store_true", help="Print plan without downloading")
    parser.add_argument("--refetch",  action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    to_fetch = [
        (aid, fname) for aid, fname in PAPERS
        if args.refetch or not (PAPERS_DIR / fname).exists()
    ]

    if not to_fetch:
        print("All papers already downloaded. Use --refetch to overwrite.")
        return

    print(f"Papers to fetch : {len(to_fetch)}")
    print(f"Already present : {len(PAPERS) - len(to_fetch)}")
    print(f"Full-text sources tried: ar5iv → arxiv-html → abstract-only\n")

    if args.dry_run:
        for aid, fname in to_fetch:
            print(f"  {aid}  →  {fname}")
        return

    saved = 0
    abstract_only = 0
    failed: list[str] = []

    for batch_start in range(0, len(to_fetch), BATCH_SIZE):
        batch     = to_fetch[batch_start : batch_start + BATCH_SIZE]
        ids       = [aid for aid, _ in batch]
        fname_map = {aid: fname for aid, fname in batch}

        bn = batch_start // BATCH_SIZE + 1
        tb = (len(to_fetch) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Batch {bn}/{tb} — fetching metadata for {len(batch)} papers …")

        try:
            entries = fetch_metadata_batch(ids)
        except Exception as e:
            print(f"  [error] metadata batch failed: {e}")
            failed.extend(ids)
            continue

        for aid in ids:
            fname = fname_map[aid]
            dest  = PAPERS_DIR / fname

            if aid not in entries:
                print(f"  [skip]  {aid} — not in arXiv API response")
                failed.append(aid)
                continue

            paper = entries[aid]
            title_short = paper["title"][:50]

            # Full text fetch
            full_text, src = _fetch_html(aid)
            if full_text:
                words = len(full_text.split())
                flag  = f"({words:,} words  via {src})"
            else:
                flag  = "(abstract only)"
                abstract_only += 1

            md = _to_markdown(paper, aid, full_text, src)
            dest.write_text(md, encoding="utf-8")
            saved += 1
            print(f"  ✓  {fname:<36} {title_short:<50} {flag}")

            time.sleep(FULL_TEXT_DELAY)

        if batch_start + BATCH_SIZE < len(to_fetch):
            time.sleep(DELAY_BETWEEN_BATCHES)

    print(f"\n{'─'*70}")
    print(f"Saved         : {saved} papers")
    print(f"Abstract-only : {abstract_only}")
    print(f"Skipped       : {len(PAPERS) - len(to_fetch)} (already existed)")
    if failed:
        print(f"Failed        : {len(failed)} — {failed}")
    total = len(list(PAPERS_DIR.glob("*.md")))
    print(f"Total in sandbox/papers/ : {total} files")


if __name__ == "__main__":
    main()
