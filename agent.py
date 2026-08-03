"""
agent.py
Step 5, final part: wires all five nodes into one LangGraph StateGraph, exactly
matching the linear flow from DESIGN.md — ingest -> retrieve -> map -> validate
-> report -> END. This is the single entry point Step 6 (the Streamlit UI)
will call.
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from nodes_ingest_retrieve import ingest_node, retrieve_node
from nodes_map import map_node
from nodes_validate import validate_node
from nodes_report import report_node


class PipelineState(TypedDict, total=False):
    file_path: str
    raw_text: str
    chunks: list
    retrieved: list
    mappings: list
    validated_mappings: list
    rejected_mappings: list
    report: str


def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("ingest", ingest_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("map", map_node)
    graph.add_node("validate", validate_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "retrieve")
    graph.add_edge("retrieve", "map")
    graph.add_edge("map", "validate")
    graph.add_edge("validate", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_pipeline(file_path: str) -> dict:
    """The one function the Streamlit UI (Step 6) will call."""
    app = build_graph()
    result = app.invoke({"file_path": file_path})
    return result


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("Running the full Governance Copilot pipeline on sample_policy.txt...\n")
    result = run_pipeline("sample_policy.txt")

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(result["report"])
