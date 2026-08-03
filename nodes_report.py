"""
nodes_report.py
Step 5, part 4: the Report node — one Groq call (the larger model, since this
runs once per document and quality matters more than speed here) that writes
the final human-readable gap report from ONLY the validated mappings.

Rejected mappings never reach this node's prompt — that's the whole point of
the hard gate running before this step.
"""

import os
import json
from groq import Groq

MODEL_LARGE = "llama-3.3-70b-versatile"  # runs once per document — quality over speed

SYSTEM_PROMPT = """You are a compliance analyst writing a gap-readiness
report for a company policy document. You will be given:
1. A list of controls VERIFIED as satisfied (each with real quoted evidence)
2. Coverage statistics broken down by theme (e.g. "Organizational: 4 of 37 covered")
3. A sample of specific controls that were NOT found to be covered

Write a clear, structured report with three sections:
1. Summary (2-3 sentences: overall coverage picture, referencing the
   coverage percentages given to you)
2. Covered controls (list each, with its evidence quote)
3. Top gaps (pick the most important-sounding controls from the "not
   covered" sample and explain in one line each why they matter — do NOT
   say you lack the information to identify gaps, you have been given a
   real sample to work from)

Be factual and specific. Do not invent any coverage beyond what's given to
you. If overall coverage is low, say so honestly rather than overstating
confidence.

Important nuance for the Summary: a single policy document is often scoped
to one part of an organization's controls on purpose — a privacy notice,
for instance, is not expected to cover physical security or employee
screening just because those categories exist in the wider control set.
A theme sitting at 0 of a large total is not automatically evidence of a
security failure; it may simply be outside what this specific document was
ever meant to address. When you write the Summary, lead with coverage in
the theme(s) this document actually engages with (based on what's in the
VERIFIED covered controls and how many themes have at least one control
covered), and only characterize a theme as a concerning "gap" if the
document's own content suggests it should plausibly have addressed it.
Avoid a single blanket "coverage is low across the board" framing when
the truth is closer to "this document is narrowly scoped and does what
it covers reasonably well, but a full ISMS policy set would need
additional documents for the other themes." Never present a
single-document scope limitation as a compliance failure of the company
as a whole."""


def load_knowledge_base_ids() -> dict:
    """Loads the full 93 ISO + 15 DPDP entries so the report can measure
    real coverage against the whole knowledge base, not just what happened
    to be retrieved as a candidate for this particular document."""
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "knowledge_base", "iso27001_controls.json")) as f:
        iso = json.load(f)
    with open(os.path.join(here, "knowledge_base", "dpdp_clauses.json")) as f:
        dpdp = json.load(f)

    entries = []
    for c in iso["controls"]:
        entries.append({"id": f"ISO-{c['control_id']}", "title": c["title"], "theme": c["theme"]})
    for c in dpdp["clauses"]:
        entries.append({"id": f"DPDP-{c['clause_id']}", "title": c["topic"], "theme": "DPDP"})
    return entries


def compute_coverage_stats(validated_mappings, all_entries):
    covered_ids = {m.control_id for m in validated_mappings}

    themes = {}
    for e in all_entries:
        themes.setdefault(e["theme"], {"total": 0, "covered": 0})
        themes[e["theme"]]["total"] += 1
        if e["id"] in covered_ids:
            themes[e["theme"]]["covered"] += 1

    uncovered = [e for e in all_entries if e["id"] not in covered_ids]

    return themes, uncovered


def report_node(state: dict) -> dict:
    """LangGraph node: state must contain 'validated_mappings'. Adds 'report'."""

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found — see .env.example.")
    client = Groq(api_key=api_key)

    validated = state.get("validated_mappings", [])
    rejected = state.get("rejected_mappings", [])

    all_entries = load_knowledge_base_ids()
    theme_stats, uncovered = compute_coverage_stats(validated, all_entries)

    # Group by control_id before rendering. Widening Retrieve's top_k means a
    # broad control (e.g. "Privacy and protection of PII") can legitimately
    # get claimed against several different sentences in the same document.
    # That's real signal (stronger evidence for that control) but it should
    # not show up as 3 separate line items for what is actually 1 control —
    # that reads as inflated coverage even though compute_coverage_stats()
    # itself already dedupes correctly by control_id.
    from collections import defaultdict
    grouped = defaultdict(list)
    for m in validated:
        grouped[m.control_id].append(m)

    covered_lines = []
    for control_id, ms in sorted(grouped.items()):
        best = max(ms, key=lambda m: m.confidence)
        line = f"- {control_id} ({best.title}): \"{best.evidence_sentence}\" [confidence {best.confidence}]"
        if len(ms) > 1:
            line += f" (also matched {len(ms) - 1} other excerpt{'s' if len(ms) > 2 else ''} in the document)"
        covered_lines.append(line)
    covered_text = "\n".join(covered_lines) or "None."

    theme_text = "\n".join(
        f"- {theme}: {stats['covered']} of {stats['total']} covered"
        for theme, stats in theme_stats.items()
    )

    # Sample up to 3 uncovered controls per theme, so the LLM has concrete
    # material for the Top Gaps section instead of just a percentage.
    sample_by_theme = {}
    for e in uncovered:
        sample_by_theme.setdefault(e["theme"], []).append(e)
    uncovered_sample_text = "\n".join(
        f"- {theme}: " + "; ".join(f"{e['id']} ({e['title']})" for e in items[:3])
        for theme, items in sample_by_theme.items()
    )

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

Coverage by theme (out of the full 93 ISO controls + 15 DPDP clauses):
{theme_text}

Sample of controls NOT found to be covered (use these for Top Gaps):
{uncovered_sample_text}

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

