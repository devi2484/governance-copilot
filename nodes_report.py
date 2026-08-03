"""
nodes_report.py
Step 5, part 4: the Report node — one Groq call (the larger model, since this
runs once per document and quality matters more than speed here) that writes
the final human-readable gap report from ONLY the validated mappings.

Rejected mappings never reach this node's prompt — that's the whole point of
the hard gate running before this step.
"""

import os
from groq import Groq

MODEL_LARGE = "llama-3.3-70b-versatile"  # runs once per document — quality over speed

SYSTEM_PROMPT = """You are a compliance analyst writing a gap-readiness
report for a company policy document. You will be given a list of controls
that were VERIFIED as satisfied (each with real quoted evidence), and the
full list of controls that were checked but NOT found to be satisfied.

Write a clear, structured report with three sections:
1. Summary (2-3 sentences: overall coverage picture)
2. Covered controls (list each, with its evidence quote)
3. Top gaps (the most important missing controls, with a one-line
   explanation of why each matters)

Be factual and specific. Do not invent any coverage beyond what's given to
you. If very few controls were checked, say so honestly rather than
overstating confidence."""


def report_node(state: dict, all_control_ids: list[str] | None = None) -> dict:
    """LangGraph node: state must contain 'validated_mappings'. Adds 'report'.

    all_control_ids (optional): the full knowledge-base ID list, so the
    report can say "X was never even checked" vs "X was checked and failed" —
    a v2 refinement. For v1, gaps are simply "everything not in validated_mappings"."""

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found — see .env.example.")
    client = Groq(api_key=api_key)

    validated = state.get("validated_mappings", [])
    rejected = state.get("rejected_mappings", [])

    covered_text = "\n".join(
        f"- {m.control_id} ({m.title}): \"{m.evidence_sentence}\" [confidence {m.confidence}]"
        for m in validated
    ) or "None."

    rejected_note = (
        f"\n\nNote: {len(rejected)} additional claim(s) were made by the initial "
        f"analysis but could not be verified against the source text, so they "
        f"are excluded from this report entirely."
        if rejected else ""
    )

    user_prompt = f"""Document: {state.get('file_path', 'uploaded document')}
Total chunks analyzed: {len(state.get('chunks', []))}

VERIFIED covered controls:
{covered_text}
{rejected_note}

Write the gap report now."""

    response = client.chat.completions.create(
        model=MODEL_LARGE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,  # a little more room than Map's temperature=0, since this is writing, not judging
    )

    report_text = response.choices[0].message.content.strip()
    print("[Report] Generated final gap report.")
    return {**state, "report": report_text}


if __name__ == "__main__":
    from dotenv import load_dotenv
    from nodes_ingest_retrieve import ingest_node, retrieve_node
    from nodes_map import map_node
    from nodes_validate import validate_node

    load_dotenv()

    state = {"file_path": "sample_policy.txt"}
    state = ingest_node(state)
    state = retrieve_node(state)
    state = map_node(state)
    state = validate_node(state)
    state = report_node(state)

    print("\n" + "=" * 60)
    print(state["report"])
    print("=" * 60)
