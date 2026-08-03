# Agent Pipeline Design

## State object (what flows through the graph)

One shared state dict is passed node to node. Each node reads what it needs
and adds its own output — nothing gets deleted, so you always have the full
trail for debugging.

| Key | Set by | Type | Description |
|---|---|---|---|
| `raw_text` | (input) | str | The uploaded document's extracted text |
| `chunks` | Ingest | list[str] | Document split into paragraph-level chunks |
| `retrieved` | Retrieve | list[dict] | Per chunk: top-3 candidate ISO/DPDP matches from ChromaDB |
| `mappings` | Map | list[ControlMapping] | LLM's claimed control/clause matches, per chunk |
| `validated_mappings` | Validate | list[ControlMapping] | Only mappings that passed the evidence check |
| `rejected_mappings` | Validate | list[dict] | Mappings the gate rejected, with the reason — kept for transparency, not hidden |
| `report` | Report | str | Final human-readable gap report |

## Node-by-node design

### 1. Ingest
- **Input:** raw uploaded file (PDF or text).
- **Logic:** extract text (`pypdf` for PDFs), split into paragraph-level chunks (roughly 2-4 sentences each — small enough that one chunk usually maps to one control, large enough to keep context).
- **No LLM call.** Pure Python. Fast, free, deterministic.
- **Output:** `chunks`.

### 2. Retrieve
- **Input:** `chunks`.
- **Logic:** for each chunk, query the ChromaDB collection built in Step 3 for the top-3 nearest ISO/DPDP entries.
- **No LLM call.** Just embedding similarity search.
- **Output:** `retrieved` — a list of `{chunk, candidates: [...]}`.

### 3. Map (LLM)
- **Input:** `retrieved`.
- **Logic:** for each `{chunk, candidates}` pair, call Groq's small model (`llama-3.1-8b-instant`) with a prompt that says, in effect: "Given this document excerpt and these candidate controls, does the excerpt actually satisfy any of them? If yes, quote the exact sentence that proves it." Force the response into the `ControlMapping` schema (see below) via structured output.
- **Why the small model here:** this step runs once per chunk, potentially dozens of times per document — cost and latency matter, and the task (matching + quoting) doesn't need heavy reasoning.
- **Output:** `mappings` (unvalidated — may contain hallucinated evidence).

### 4. Hard-gate validate
- **Input:** `mappings`.
- **Logic:** pure Python, no LLM call. For every mapping claiming `satisfied: true`, check that `evidence_sentence` actually appears (near-exact substring match, allowing minor whitespace/punctuation differences) inside the original chunk text. If it doesn't match, the mapping is downgraded to rejected — regardless of how confident the model sounded.
- **This is the single most important node in the whole pipeline** — it's what makes the tool trustworthy rather than just plausible-sounding.
- **Output:** `validated_mappings`, `rejected_mappings`.

### 5. Report (LLM)
- **Input:** `validated_mappings` (only — the report never sees rejected/unverified claims).
- **Logic:** one call to Groq's larger model (`llama-3.3-70b-versatile`) with the full validated mapping list, asked to write a structured gap report: coverage by theme, top 3 gaps, plain-English summary.
- **Why the large model here:** this runs once per document (not once per chunk), and needs better synthesis/writing quality — the cost tradeoff favors quality over speed for this single call.
- **Output:** `report`.

## Why this order, specifically
Retrieval happens **before** the LLM call (not after) so the model is only ever asked to judge a short, pre-filtered candidate list — not to search the entire 108-entry knowledge base itself from memory, which is exactly the kind of task that invites hallucination. Validation happens **after** mapping and **before** reporting, so a bad claim never reaches the user-facing output at all — it's caught at the earliest possible point after it's created.

## Control flow (LangGraph)
Linear: `ingest → retrieve → map → validate → report → END`. No branching needed for v1 — every chunk goes through every stage. (A v2 extension could add a loop: if a chunk's mapping gets rejected, re-run Map once with a stricter prompt before giving up — worth noting as a future improvement, not needed for v1.)
