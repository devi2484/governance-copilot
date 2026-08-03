"""
nodes_validate.py
Step 5, part 3: the Hard-gate Validate node — the most important node in the
whole project. No API key, no LLM call. Pure Python, fully deterministic.

Its only job: for every mapping the LLM claimed was "satisfied", check that
the evidence_sentence it quoted actually exists in the real source chunk.
If it doesn't, the claim is rejected — no matter how confident the model
sounded. This is what stops the tool from hallucinating compliance.
"""

import re
from schemas import ControlMapping, RejectedMapping


def normalize(text: str) -> str:
    """Lowercase + collapse whitespace/punctuation differences so we're
    comparing meaning-equivalent text, not being overly strict about a
    stray comma or double space."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def evidence_exists_in_source(evidence_sentence: str, source_chunk: str, min_overlap: float = 0.85) -> bool:
    """Checks the claimed evidence sentence is a genuine near-exact match
    inside the real source chunk — not a paraphrase, not an invention.

    Method: normalize both strings, then check what fraction of the
    evidence's words appear, in order, as a contiguous run inside the
    source. A simple substring check would be too strict (misses tiny
    whitespace diffs); pure word-overlap (no order) would be too loose
    (misses invented sentences reusing the same vocabulary). Requiring a
    contiguous, mostly-matching run is the middle ground."""
    if not evidence_sentence:
        return False

    norm_evidence = normalize(evidence_sentence)
    norm_source = normalize(source_chunk)

    # Fast path: exact substring match after normalization
    if norm_evidence in norm_source:
        return True

    # Fallback: check if a high fraction of the evidence appears as a
    # contiguous substring, to tolerate very minor differences (e.g. the
    # model dropping a trailing word).
    evidence_len = len(norm_evidence)
    if evidence_len == 0:
        return False

    # Try trimming a few characters off each end and re-check
    for trim in range(0, min(20, evidence_len // 4)):
        trimmed = norm_evidence[trim: evidence_len - trim] if trim > 0 else norm_evidence
        if len(trimmed) / evidence_len >= min_overlap and trimmed in norm_source:
            return True

    return False


def validate_node(state: dict) -> dict:
    """LangGraph node: state must contain 'mappings' and 'retrieved'
    (to look up each chunk's real source text). Adds 'validated_mappings'
    and 'rejected_mappings'."""

    # Build a lookup from chunk_id -> real source text
    chunk_lookup = {r["chunk_id"]: r["chunk_text"] for r in state["retrieved"]}

    validated = []
    rejected = []

    for mapping in state["mappings"]:
        source_chunk = chunk_lookup.get(mapping.chunk_id, "")

        if not mapping.satisfied:
            # Nothing to validate — the model already said "not satisfied",
            # so there's no compliance claim being made that needs checking.
            continue

        if evidence_exists_in_source(mapping.evidence_sentence, source_chunk):
            validated.append(mapping)
        else:
            rejected.append(RejectedMapping(
                mapping=mapping,
                rejection_reason=(
                    "Claimed evidence sentence could not be verified against "
                    "the real source text — likely a hallucinated or "
                    "paraphrased quote rather than an exact excerpt."
                ),
            ))

    print(f"[Validate] {len(validated)} mappings passed the hard gate, "
          f"{len(rejected)} were rejected as unverifiable.")

    return {**state, "validated_mappings": validated, "rejected_mappings": rejected}


# --- Self-test with fabricated data, no API key or LLM call needed ---
if __name__ == "__main__":
    test_state = {
        "retrieved": [
            {"chunk_id": 0, "chunk_text": "We conduct background verification checks on all new hires."},
        ],
        "mappings": [
            # This one should PASS — evidence is a real quote from the chunk
            ControlMapping(
                source="ISO27001", control_id="ISO-6.1", title="Screening",
                satisfied=True,
                evidence_sentence="We conduct background verification checks on all new hires.",
                confidence=0.9, chunk_id=0,
            ),
            # This one should FAIL — evidence is invented, not actually in the chunk
            ControlMapping(
                source="ISO27001", control_id="ISO-8.7", title="Protection against malware",
                satisfied=True,
                evidence_sentence="All systems run up-to-date antivirus software.",
                confidence=0.7, chunk_id=0,
            ),
        ],
    }

    result = validate_node(test_state)
    print(f"\nValidated: {len(result['validated_mappings'])} (expected 1)")
    print(f"Rejected: {len(result['rejected_mappings'])} (expected 1)")
    assert len(result["validated_mappings"]) == 1, "Expected exactly 1 validated mapping"
    assert len(result["rejected_mappings"]) == 1, "Expected exactly 1 rejected mapping"
    print("\nSelf-test PASSED — the hard gate correctly separated real evidence from invented evidence.")
