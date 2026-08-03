"""
nodes_map.py
Step 5, part 2: the Map node — the first node that needs an API key.

For each chunk + its candidate controls (from Retrieve), asks Groq's small,
fast model whether the chunk actually satisfies any candidate, and forces the
answer into the ControlMapping schema from schemas.py.
"""

import os
import json
from groq import Groq
from schemas import ControlMapping

MODEL_SMALL = "llama-3.1-8b-instant"  # cheap, fast — runs once per chunk

SYSTEM_PROMPT = """You are a compliance analyst. You will be given a short
excerpt from a company policy document, and a list of candidate ISO 27001 or
DPDP controls/clauses that might be relevant.

For EACH candidate, decide: does the excerpt actually satisfy this control?
Be strict — only mark satisfied=true if the excerpt clearly and specifically
addresses what the control requires, not just if it's loosely related.

If satisfied=true, you MUST quote the exact sentence from the excerpt (word
for word, no paraphrasing) that proves it in evidence_sentence.
If satisfied=false, leave evidence_sentence as null.

Respond with ONLY a JSON array, no other text, one object per candidate, each
matching this shape:
{"source": "...", "control_id": "...", "title": "...", "satisfied": true/false,
 "evidence_sentence": "..." or null, "confidence": 0.0-1.0}
"""


def map_chunk(client: Groq, chunk_text: str, candidates: list[dict], chunk_id: int) -> list[ControlMapping]:
    candidate_list = "\n".join(
        f"- {c['id']} ({c['metadata'].get('title') or c['metadata'].get('topic')}): {c['text']}"
        for c in candidates
    )
    user_prompt = f"""Excerpt:
\"\"\"{chunk_text}\"\"\"

Candidate controls:
{candidate_list}

Return the JSON array now."""

    response = client.chat.completions.create(
        model=MODEL_SMALL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,  # deterministic — we want consistent compliance judgments, not creativity
    )

    raw = response.choices[0].message.content.strip()
    # The model sometimes wraps its JSON array in markdown code fences even
    # when told not to — strip those defensively before parsing.
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[Map] WARNING: chunk {chunk_id} returned unparseable JSON, skipping.")
        return []

    mappings = []
    for item in parsed:
        try:
            item["chunk_id"] = chunk_id
            mappings.append(ControlMapping(**item))
        except Exception as e:
            print(f"[Map] WARNING: skipped malformed mapping for chunk {chunk_id}: {e}")

    return mappings


def map_node(state: dict) -> dict:
    """LangGraph node: state must contain 'retrieved'. Adds 'mappings'."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Create a .env file with GROQ_API_KEY=your_key_here "
            "(see the .env.example file in this project)."
        )
    client = Groq(api_key=api_key)

    all_mappings = []
    for r in state["retrieved"]:
        mappings = map_chunk(client, r["chunk_text"], r["candidates"], r["chunk_id"])
        all_mappings.extend(mappings)

    print(f"[Map] LLM produced {len(all_mappings)} candidate mappings "
          f"({sum(1 for m in all_mappings if m.satisfied)} claimed satisfied).")
    return {**state, "mappings": all_mappings}


if __name__ == "__main__":
    from dotenv import load_dotenv
    from nodes_ingest_retrieve import ingest_node, retrieve_node

    load_dotenv()

    state = {"file_path": "sample_policy.txt"}
    state = ingest_node(state)
    state = retrieve_node(state)
    state = map_node(state)

    for m in state["mappings"]:
        status = "SATISFIED" if m.satisfied else "not satisfied"
        print(f"\n[{m.control_id}] {m.title} — {status} (confidence {m.confidence})")
        if m.evidence_sentence:
            print(f"   Evidence: \"{m.evidence_sentence}\"")
