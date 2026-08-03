"""
schemas.py
Shared data structures for the Governance Copilot pipeline (Step 4 design,
used starting in Step 5's node implementations).

Keeping these in one file means every node imports the same contract — the
Map node's LLM output, the Validate node's checks, and the Report node's input
are all guaranteed to be shaped identically. This is the schema-enforcement
pattern carried over from Lumen.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal


class ControlMapping(BaseModel):
    """One claimed match between a document chunk and a knowledge-base entry.
    This is the exact shape the Map node's LLM call must return."""

    source: Literal["ISO27001", "DPDP"] = Field(
        description="Which knowledge base the matched entry came from"
    )
    control_id: str = Field(
        description="e.g. 'ISO-8.28' or 'DPDP-DPDP-6' — matches the id used in ChromaDB"
    )
    title: str = Field(
        description="The control/clause title, for display in the report"
    )
    satisfied: bool = Field(
        description="Does the LLM believe this chunk satisfies this control/clause?"
    )
    evidence_sentence: Optional[str] = Field(
        default=None,
        description="The exact sentence from the chunk that proves satisfaction. "
                    "Required if satisfied=True. This is what the hard-gate checks."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model's self-reported confidence, 0-1"
    )
    chunk_id: int = Field(
        description="Index of the source chunk this mapping came from"
    )


class RejectedMapping(BaseModel):
    """A mapping the hard-gate validator downgraded — kept for transparency
    rather than silently discarded, so the report can (optionally) note
    'the model claimed X but we couldn't verify it' during debugging."""

    mapping: ControlMapping
    rejection_reason: str


class GapReportInput(BaseModel):
    """What the Report node receives — only validated mappings ever reach here."""

    validated_mappings: list[ControlMapping]
    document_name: str
    total_chunks: int


# --- Quick self-test: run `python3 schemas.py` to confirm the schemas import
# and validate correctly before wiring them into the actual LangGraph nodes.
if __name__ == "__main__":
    sample = ControlMapping(
        source="ISO27001",
        control_id="ISO-6.1",
        title="Screening",
        satisfied=True,
        evidence_sentence="We conduct background verification checks on all new hires.",
        confidence=0.91,
        chunk_id=2,
    )
    print("Schema OK:", sample.model_dump_json(indent=2))
