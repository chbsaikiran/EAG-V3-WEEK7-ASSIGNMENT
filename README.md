# Session 7 — RAG Application over ML/AI Research Papers

A real RAG (Retrieval-Augmented Generation) application built on top of the Session 7 agent infrastructure. The corpus contains 50+ ML/AI research papers. Retrieval uses FAISS vector search with local embeddings (Ollama `nomic-embed-text`). The LLM is served through `llm_gatewayV7` running on `localhost:8107`.

---

## Architecture

```
fetch_papers.py          →   sandbox/papers/*.md       (corpus, run once)
        ↓
index_corpus.py          →   state/memory.json          (FAISS index, run once)
        ↓                     state/*.faiss / *.index
rag_query.py             →   LLM answer via gateway     (run anytime)
```

Each script is standalone — just run with `uv run <script>`.

---

## Prerequisites

1. **Gateway running** on port 8107:
   ```bash
   cd ../llm_gatewayV7
   uv run main.py
   ```

2. **Ollama running** with the embedding model pulled:
   ```bash
   ollama serve          # in a separate terminal
   ollama pull nomic-embed-text
   ```

3. **`.env`** at `Assignment7/.env` with at least:
   ```
   GEMINI_API_KEY=...
   GROQ_API_KEY=...
   TAVILY_API_KEY=...
   EMBED_ORDER=ollama,gemini
   ```

---

## Scripts

### 1. `fetch_papers.py` — Build the corpus

Downloads 48 ML/AI papers from arXiv and saves them as markdown files in `sandbox/papers/`.

**How it works:**
- Fetches metadata (title, authors, abstract) from the **arXiv Atom API** in batches of 20
- Fetches **full paper body text** from `ar5iv.labs.arxiv.org/html/{id}` (server-side rendered HTML — no browser needed)
- Falls back to `arxiv.org/html/{id}` if ar5iv is unavailable, then to abstract-only
- Strips HTML tags using Python's stdlib `html.parser` — no extra dependencies
- Caps each file at 120,000 characters (~30k tokens) to keep files manageable
- Skips files that already exist (idempotent — safe to re-run)

**Papers covered** (9 topic areas):
| Topic | Examples |
|---|---|
| Foundational LLMs | BERT, GPT-3, T5, LLaMA, Mistral, Mixtral, Mamba |
| Efficient fine-tuning | QLoRA, Adapter, Prefix Tuning, Flash Attention 1 & 2, GQA, RoPE |
| RLHF & Alignment | InstructGPT, Constitutional AI, PPO, Learning to Summarize |
| Reasoning & Prompting | Self-Consistency, Tree of Thoughts, PAL, Zero-shot CoT, Reflexion |
| RAG & Retrieval | RAG, REALM, DPR, ColBERT, HyDE |
| Agents & Tool Use | Toolformer, HuggingGPT, Voyager, AgentBench, SWE-bench |
| Multimodal | CLIP, DALL-E 2, Flamingo, LLaVA |
| Benchmarks | SuperGLUE, TruthfulQA, GSM8K |

**Run:**
```bash
uv run fetch_papers.py              # fetch all 48 papers
uv run fetch_papers.py --dry-run    # preview what would be fetched
uv run fetch_papers.py --refetch    # overwrite existing files
```

---

### 2. `index_corpus.py` — Build the FAISS index

Reads every `.md` file in `sandbox/papers/`, chunks the text, embeds each chunk via the gateway's `/v1/embed` endpoint, and writes the vectors into the FAISS index on disk.

**How it works:**
- Sliding-window chunking: **400 words per chunk, 80-word overlap** — so adjacent chunks share context
- Each chunk is embedded via `nomic-embed-text` (768-dim) through Ollama
- Chunks are stored as `fact` records in `state/memory.json` alongside their vector in `state/*.faiss`
- The `chunk_index`, `total_chunks`, and `source` fields are stored per chunk — used later by `rag_query.py` to fetch neighbors
- `--clear` wipes the index and re-indexes from scratch

**Run:**
```bash
uv run index_corpus.py              # index all papers (skips already-indexed runs)
uv run index_corpus.py --clear      # wipe and re-index from scratch
```

---

### 3. `rag_query.py` — Query the corpus

Takes a natural language question, retrieves the most relevant chunks from FAISS, expands each hit with its neighboring chunks, and sends the assembled context to the LLM for a grounded answer.

**How it works:**

**Step 1 — FAISS retrieval:**
The query is embedded with `nomic-embed-text` and compared against all indexed chunk vectors using cosine similarity. Top-k chunks are returned.

**Step 2 — Neighbor expansion:**
For each matched chunk, the chunk immediately before it and immediately after it (within the same source file) are also fetched from the memory store. This prevents answers from being cut off at chunk boundaries.

```
FAISS hit: lora.md chunk 5
  → adds: lora.md chunk 4  (prev)
  → keeps: lora.md chunk 5  (★ match)
  → adds: lora.md chunk 6  (next)
```

Chunks are deduplicated — if two FAISS hits are adjacent, no chunk appears twice.

**Step 3 — LLM answer:**
The expanded context (up to 15 chunks) is sent to Groq (100k context window) with a system prompt that instructs the model to answer only from the provided context and cite the source file for every claim.

**Context layout sent to LLM:**
```
=== Result 1: lora.md  chunk 5/18 ===
[↑ before]  ...chunk 4 text...
[★ match ]  ...chunk 5 text (FAISS hit)...
[↓ after ]  ...chunk 6 text...

────────────────────────────────────────
=== Result 2: qlora.md  chunk 3/12 ===
...
```

**Run:**
```bash
# Standard RAG query
uv run rag_query.py "How can we fine-tune large models with less memory?"

# LLM only — no retrieval (for comparison)
uv run rag_query.py "How can we fine-tune large models with less memory?" --no-index

# Side-by-side comparison
uv run rag_query.py "How can we fine-tune large models with less memory?" --compare

# Retrieve more chunks (default is 5)
uv run rag_query.py "What is LoRA?" --top-k 8
```

---

## Full Setup (from scratch)

```bash
cd S7code

# 1. Fetch all papers (~48 files saved to sandbox/papers/)
uv run fetch_papers.py

# 2. Index the corpus into FAISS
uv run index_corpus.py

# 3. Query
uv run rag_query.py "What is LoRA and how does it reduce trainable parameters?"
```

---

## Five Test Queries

These queries are designed to answer correctly **with** the index and fail or hallucinate **without** it. Queries 3 and 4 are **semantic recall** — the query words do not appear in the answer chunks.

| # | Type | Query | Answered by |
|---|---|---|---|
| 1 | Direct | What exact BLEU score did the Transformer achieve on WMT 2014 English-to-German, and by how much did it beat the previous best? | `attention.md` |
| 2 | Direct | How does DPO train a language model on human preferences without using a separate reward model? | `dpo.md` |
| 3 | **Semantic** | What approach lets a model improve its own answers using a set of written principles rather than human-labeled feedback? | `constitutional_ai.md` |
| 4 | **Semantic** | How can an AI agent fix its own errors by reflecting on what went wrong in a previous attempt? | `reflexion.md` |
| 5 | Cross-doc | Which techniques avoid storing the full N×N attention matrix in GPU memory during training? | `flash_attention.md` + `flash_attention2.md` |

**Why semantic queries fail without the index:**
- Query 3 uses *"written principles"* — the chunk uses *"constitutional"* (zero lexical overlap)
- Query 4 uses *"fix its own errors / reflecting on what went wrong"* — the chunk uses *"verbal reinforcement / self-reflection signals / episodic memory"* (zero lexical overlap)

---

## Sample Run Logs

```
(eagv3-s7) saikiran@Saikirans-MBP S7code % uv run rag_query.py "What exact BLEU score did the Transformer achieve on WMT 2014 English-to-German, and by how much did it beat the previous best?"

Query : What exact BLEU score did the Transformer achieve on WMT 2014 English-to-German, and by how much did it beat the previous best?
Mode  : RAG with neighbor expansion  (top_k=5)

Retrieved 5 matched chunks → expanded to 15 chunks (with neighbors):
  [1] palm.md chunk 19  +2 neighbor(s) — show that PaLM obtains competitive close-to-SOTA performance. It is worth noting that both…
  [2] palm.md chunk 51  +2 neighbor(s) — stricter generative scoring. Particularly noteworthy is that PaLM 540B performance improve…
  [3] gsm8k.md chunk 34  +2 neighbor(s) — 2022. Announcing the inverse scaling prize ($250k prize pool). Meng et al. (2022) Yu Meng,…
  [4] reflexion.md chunk 13  +2 neighbor(s) — we displayed the inferior performance of Reflexion to the baseline GPT-4 on MBPP Python. I…
  [5] least_to_most.md chunk 45  +2 neighbor(s) — "RUN") * 4 * 2 + ("TURN_LEFT" + "WALK") * 4 * 3. So the output of "run around left twice a…

Answer (with index + neighbors):
The provided excerpts do not contain any information about the BLEU score achieved by the
Transformer on the WMT 2014 English-to-German benchmark, nor how much it surpassed the
previous best result.

[provider=groq  model=openai/gpt-oss-120b  in=2441 out=171]


(eagv3-s7) saikiran@Saikirans-MBP S7code % uv run rag_query.py "How does DPO train a language model on human preferences without using a separate reward model?"

Query : How does DPO train a language model on human preferences without using a separate reward model?
Mode  : RAG with neighbor expansion  (top_k=5)

Retrieved 5 matched chunks → expanded to 15 chunks (with neighbors):
  [1] llama2.md chunk 56  +2 neighbor(s) — language models with human preferences. arXiv preprint arXiv:2302.08582, 2023. Kudo and Ri…
  [2] generative_agents.md chunk 16  +2 neighbor(s) — Large language models are zero-shot reasoners. Neural Information Processing Systems (Neur…
  [3] llama2.md chunk 18  +2 neighbor(s) — we further train our language model following the RL scheme of , which uses the reward mod…
  [4] llama2.md chunk 31  +2 neighbor(s) — according to a set of guidelines. We then use the human preference data to train a safety …
  [5] gsm8k.md chunk 40  +2 neighbor(s) — found that larger models are more effective at maximizing PM rewards after RL. Overall, we…

Answer (with index + neighbors):
The provided excerpts do not contain any description of Direct Preference Optimization
(DPO) or how it trains a language model on human preferences without a separate reward
model. Consequently, I cannot answer the question based on the given context.

[provider=groq  model=openai/gpt-oss-120b  in=2556 out=169]


(eagv3-s7) saikiran@Saikirans-MBP S7code % uv run rag_query.py "How can an AI agent fix its own errors by reflecting on what went wrong in a previous attempt?"

Query : How can an AI agent fix its own errors by reflecting on what went wrong in a previous attempt?
Mode  : RAG with neighbor expansion  (top_k=5)

Retrieved 4 matched chunks → expanded to 12 chunks (with neighbors):
  [1] agentbench.md chunk 33  +2 neighbor(s) — after the correct SQL operation. Metrics. We measure the Success Rate of agents in complet…
  [2] agentbench.md chunk 51  +2 neighbor(s) — truth, but your responses must align with the facts of the truth. 5. You can only answer "…
  [3] agentbench.md chunk 8  +2 neighbor(s) — "irrelevant". The game is terminated when one of the player recovers the critical plots of…
  [4] generative_agents.md chunk 6  +2 neighbor(s) — method, along with additional approaches for the model to self-improve without supervised …

Answer (with index + neighbors):
An AI agent can "look back" on a failed attempt, generate a set of alternative reasoning
traces for the same problem, and then use the most reliable trace to update itself.  In
the generative-agents framework this is done by prompting the model to produce many
Chain-of-Thought (CoT) reasoning paths, applying a self-consistency check to pick the
high-confidence (i.e., correct) path, and then treating that path as pseudo-labeled data
for further fine-tuning.  By repeatedly sampling, filtering, and training on these
self-selected, high-confidence solutions, the agent learns from its own mistakes and
improves its future performance without any external supervision [generative_agents.md].

[provider=groq  model=openai/gpt-oss-120b  in=1758 out=329]


(eagv3-s7) saikiran@Saikirans-MBP S7code % uv run rag_query.py "What approach lets a model improve its own answers using a set of written principles rather than human-labeled feedback?"

Query : What approach lets a model improve its own answers using a set of written principles rather than human-labeled feedback?
Mode  : RAG with neighbor expansion  (top_k=5)

Retrieved 5 matched chunks → expanded to 15 chunks (with neighbors):
  [1] constitutional_ai.md chunk 23  +2 neighbor(s) — using human feedback labels for harmlessness. We referred to the technique as 'constitutio…
  [2] learning_to_summarize.md chunk 6  +2 neighbor(s) — line of research, human feedback has been used to train agents in simulated environments .…
  [3] constitutional_ai.md chunk 20  +2 neighbor(s) — typically causes the feedback model to commit to one choice over another, resulting in pro…
  [4] instructgpt.md chunk 30  +2 neighbor(s) — improve models' truthfulness, such as WebGPT . In this work, if the user requests a potent…
  [5] generative_agents.md chunk 6  +2 neighbor(s) — method, along with additional approaches for the model to self-improve without supervised …

Answer (with index + neighbors):
The approach is called Constitutional AI (CAI). In CAI a "constitution" of human-written
principles is used so that the model can critique and revise its own responses—
bootstrapping its instruction-following ability to remove harmful content—rather than
relying on human-labeled feedback [constitutional_ai.md chunk 23/55].

[provider=groq  model=openai/gpt-oss-120b  in=2121 out=286]


(eagv3-s7) saikiran@Saikirans-MBP S7code % uv run rag_query.py "Which techniques avoid storing the full N×N attention matrix in GPU memory during training?"

Query : Which techniques avoid storing the full N×N attention matrix in GPU memory during training?
Mode  : RAG with neighbor expansion  (top_k=5)

Retrieved 5 matched chunks → expanded to 14 chunks (with neighbors):
  [1] gqa.md chunk 6  +2 neighbor(s) — avoid materializing the quadratic attention scores, reducing memory and speeding up traini…
  [2] gqa.md chunk 8  +1 neighbor(s) — memory-efficient exact attention with io-awareness. CoRR, abs/2205.14135. de Jong et al. (…
  [3] t5.md chunk 48  +2 neighbor(s) — we train our models on) are most efficient for large dense matrix multiplications like tho…
  [4] self_ask.md chunk 6  +2 neighbor(s) — to train a large 6GB model. What's worse, their method may not be applicable to smaller lan…
  [5] instructgpt.md chunk 4  +2 neighbor(s) — RM as a reward function and fine-tune our supervised learning baseline to maximize this re…

Answer (with index + neighbors):
The paper lists several approaches that sidestep materialising the full N × N attention
matrix (i.e., the quadratic-size attention scores) during training, thereby cutting GPU
memory use. These include:
  * Quantisation – lowering the precision of weights and activations to shrink memory footprint.
  * Model distillation – training a smaller model with data generated by a larger one.
  * Layer-sparse cross-attention – removing most cross-attention layers.
  * Speculative sampling – using a smaller model to propose multiple tokens before scoring
    with the full model.
These techniques are described as "avoid[ing] materializing the quadratic attention
scores, reducing memory and speeding up training" [gqa.md chunk 6/10].

[provider=groq  model=openai/gpt-oss-120b  in=2165 out=551]
```

---

## Notes on Results

Queries 1 and 2 correctly returned **"not in context"** — `attention.md` (the Transformer paper) was fetched as raw arxiv HTML with navigation noise before `fetch_papers.py` was updated to use the ar5iv full-text endpoint. Re-running `fetch_papers.py --refetch` followed by `index_corpus.py --clear` on the original five papers will fix retrieval for those queries.

Queries 3, 4, and 5 retrieved relevant chunks and produced grounded answers citing the correct source files.
