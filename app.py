"""
app.py
Step 6: the Streamlit UI for Verity — Evidence-gated compliance mapping.
Run with: streamlit run app.py
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from nodes_ingest_retrieve import ingest_node, retrieve_node
from nodes_map import map_node
from nodes_validate import validate_node
from nodes_report import report_node
from build_retriever import build_collection

load_dotenv()

st.set_page_config(
    page_title="Verity — Governance Copilot",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling — dark charcoal + amber accent, matching a consistent visual
# identity across this whole placement-prep project.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
:root {
    --accent: #e8d94a;
    --accent-dim: #b8ab2f;
}
.stApp { background-color: #17181a; }
h1, h2, h3 { font-family: 'Georgia', serif; }
.verity-header {
    padding: 8px 0 20px;
    border-bottom: 1px solid #38393d;
    margin-bottom: 24px;
}
.verity-tag {
    font-family: monospace;
    font-size: 11px;
    letter-spacing: 2px;
    color: var(--accent);
    text-transform: uppercase;
}
.verity-sub {
    color: #a9a9a2;
    font-size: 14px;
    margin-top: 4px;
}
.badge {
    display: inline-block;
    font-family: monospace;
    font-size: 10.5px;
    padding: 3px 10px;
    border: 1px solid #38393d;
    border-radius: 20px;
    color: #a9a9a2;
    margin-right: 6px;
}
.result-card {
    background: #232427;
    border: 1px solid #38393d;
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.result-card .cid {
    font-family: monospace;
    font-size: 11px;
    color: var(--accent);
    letter-spacing: 1px;
}
.result-card .quote {
    color: #c8c7be;
    font-style: italic;
    margin-top: 6px;
    border-left: 2px solid #38393d;
    padding-left: 10px;
}
.rejected-card {
    background: #232427;
    border: 1px solid #38393d;
    border-left: 3px solid #77776f;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
    opacity: 0.85;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="verity-header">
    <div class="verity-tag">Evidence-gated compliance mapping</div>
    <h1 style="margin:4px 0 0;">🔎 Verity</h1>
    <div class="verity-sub">
        Upload a company policy document to see which ISO 27001 and DPDP Act
        controls it actually covers — every claim backed by a real quoted
        sentence, checked in code, not just trusted from the model.
    </div>
    <div style="margin-top:12px;">
        <span class="badge">LangGraph</span>
        <span class="badge">Groq</span>
        <span class="badge">ChromaDB</span>
        <span class="badge">Hard-gate validated</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "Verity is a portfolio project — a study/demo tool, **not** a "
        "certified compliance product or legal advice."
    )
    st.markdown("### How it works")
    st.markdown(
        "1. **Ingest** — parse and chunk the document\n"
        "2. **Retrieve** — semantic search against ISO 27001 + DPDP\n"
        "3. **Map** — LLM judges each candidate match\n"
        "4. **Validate** — every claim checked against the real text\n"
        "5. **Report** — final gap report, evidence-cited only"
    )
    st.markdown("### Knowledge base")
    st.caption("93 ISO/IEC 27001:2022 Annex A controls")
    st.caption("16 DPDP Act 2023 key obligations")

if not os.environ.get("GROQ_API_KEY"):
    st.error(
        "No Groq API key found. Create a `.env` file in this project folder "
        "(copy `.env.example` and paste in your key from console.groq.com), "
        "then restart this app."
    )
    st.stop()


@st.cache_resource
def ensure_index_is_built():
    """Builds the ChromaDB search index from knowledge_base/*.json the first
    time this app instance starts, and caches it for the life of that
    instance. This removes the old dependency on manually running
    `python build_retriever.py` before deploying — chroma_store/ is git-
    ignored on purpose (it's a rebuildable artifact, not source), which
    means a deployed app with no auto-build step would silently search
    against a stale or missing index. st.cache_resource ensures this only
    runs once per running instance, not once per file upload."""
    with st.spinner("First-time setup: building the ISO/DPDP search index..."):
        build_collection()
    return True


ensure_index_is_built()

# ---------------------------------------------------------------------------
# Upload + run
# ---------------------------------------------------------------------------
col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader(
        "Upload a policy document", type=["pdf", "txt"], label_visibility="collapsed"
    )
with col2:
    run_clicked = st.button("Run analysis", type="primary", use_container_width=True)

if uploaded_file is not None and run_clicked:
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    state = {"file_path": tmp_path}

    with st.status("Running pipeline...", expanded=True) as status:
        st.write("**1 · Ingest** — parsing and chunking the document...")
        state = ingest_node(state)
        st.write(f"Found {len(state['chunks'])} chunks.")

        st.write("**2 · Retrieve** — searching the ISO 27001 / DPDP knowledge base...")
        state = retrieve_node(state)
        st.write("Candidate matches found for each chunk.")
        st.write("**3 · Map** — asking the model to judge each candidate match...")
        state = map_node(state)
        claimed = sum(1 for m in state["mappings"] if m.satisfied)
        st.write(f"Model claimed {claimed} controls were satisfied.")
        if state.get("map_warnings"):
            st.warning(
                f"{len(state['map_warnings'])} chunk(s) had a parsing issue during "
                f"Map — see details below. This can cause a document to look like "
                f"it has less coverage than it really does."
            )

        st.write("**4 · Hard-gate validate** — checking every claim against the real text...")
        state = validate_node(state)
        st.write(
            f"{len(state['validated_mappings'])} claims verified, "
            f"{len(state['rejected_mappings'])} rejected as unverifiable."
        )

        st.write("**5 · Report** — writing the final gap report...")
        state = report_node(state)

        status.update(label="Analysis complete", state="complete", expanded=False)

    st.session_state["result"] = state
    os.unlink(tmp_path)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if "result" in st.session_state:
    state = st.session_state["result"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Chunks analyzed", len(state.get("chunks", [])))
    m2.metric("Verified matches", len(state.get("validated_mappings", [])))
    m3.metric("Rejected claims", len(state.get("rejected_mappings", [])))

    if state.get("map_warnings"):
        with st.expander(f"🐛 Debug: {len(state['map_warnings'])} Map parsing warning(s)"):
            st.caption(
                "These chunks had an issue during the Map step (usually the model's "
                "reply wasn't in the expected format) and were skipped — meaning "
                "the report below may be missing coverage that's actually there. "
                "If you see these often, share this list to debug the prompt further."
            )
            for w in state["map_warnings"]:
                st.code(w, language=None)

    with st.expander("🔍 Debug: retrieval candidates per chunk (what the search actually found)"):
        st.caption(
            "This shows the raw candidate list from ChromaDB before the AI model "
            "judges anything — useful for checking whether a control you expected "
            "to see (e.g. a specific DPDP clause) was even offered as a candidate "
            "for a given chunk. If it's missing here, the model never had the "
            "chance to consider it. If it's present here but missing from the "
            "final report, the model considered it and judged it not satisfied, "
            "or the hard gate rejected the evidence — both are working as intended, "
            "not a bug."
        )
        for r in state.get("retrieved", []):
            st.markdown(f"**Chunk {r['chunk_id']}:** _{r['chunk_text'][:100]}..._")
            for c in r["candidates"]:
                label = c["metadata"].get("title") or c["metadata"].get("topic")
                st.caption(f"  · {c['id']} — {label}")

    tab1, tab2, tab3 = st.tabs(["📄 Gap report", "✅ Verified evidence", "⚠️ Rejected claims"])

    with tab1:
        st.markdown(state["report"])
        st.download_button(
            "Download report (markdown)",
            data=state["report"],
            file_name="verity_gap_report.md",
            mime="text/markdown",
        )
    with tab2:
        if not state["validated_mappings"]:
            st.info("No controls were verified as covered in this document.")
        for m in state["validated_mappings"]:
            st.markdown(f"""
            <div class="result-card">
                <span class="cid">{m.control_id}</span> — <strong>{m.title}</strong>
                <div class="quote">"{m.evidence_sentence}"</div>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        if not state["rejected_mappings"]:
            st.info("No claims were rejected — everything the model proposed checked out.")
        else:
            st.caption(
                "These were claimed by the model but could not be verified "
                "against the real document text, so they're excluded from "
                "the report."
            )
        for r in state["rejected_mappings"]:
            st.markdown(f"""
            <div class="rejected-card">
                <span class="cid">{r.mapping.control_id}</span> — <strong>{r.mapping.title}</strong>
                <div class="quote">{r.rejection_reason}</div>
            </div>
            """, unsafe_allow_html=True)

