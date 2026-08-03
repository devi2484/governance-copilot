"""
app.py
Step 6: the Streamlit UI. Run with: streamlit run app.py

Lets a user upload a policy document, watch the pipeline stages run, and see
the final gap report with evidence citations. This is what a portfolio
reviewer will actually click through — the visible stage-by-stage progress
is what shows the multi-agent architecture, not just the final text.
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from nodes_ingest_retrieve import ingest_node, retrieve_node
from nodes_map import map_node
from nodes_validate import validate_node
from nodes_report import report_node

load_dotenv()

st.set_page_config(page_title="Enterprise Governance Copilot", page_icon="📋", layout="centered")

st.title("Enterprise Governance Copilot")
st.caption(
    "Upload a company policy document to see which ISO 27001 / DPDP Act "
    "controls it covers — with every claim backed by a real quoted sentence, "
    "not a guess. This is a study/portfolio project, not a certified "
    "compliance tool or legal advice."
)

if not os.environ.get("GROQ_API_KEY"):
    st.error(
        "No Groq API key found. Create a `.env` file in this project folder "
        "(copy `.env.example` and paste in your key from console.groq.com), "
        "then restart this app."
    )
    st.stop()

uploaded_file = st.file_uploader("Upload a policy document", type=["pdf", "txt"])

if uploaded_file is not None:
    if st.button("Run analysis", type="primary"):
        # Save the uploaded file to a temp path so our existing ingest_node
        # (which expects a file path, not an in-memory object) can read it.
        suffix = "." + uploaded_file.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        state = {"file_path": tmp_path}

        with st.status("Running pipeline...", expanded=True) as status:
            st.write("**Stage 1 — Ingest:** parsing and chunking the document...")
            state = ingest_node(state)
            st.write(f"Found {len(state['chunks'])} chunks.")

            st.write("**Stage 2 — Retrieve:** searching the ISO 27001 / DPDP knowledge base...")
            state = retrieve_node(state)
            st.write("Candidate matches found for each chunk.")

            st.write("**Stage 3 — Map:** asking the model to judge each candidate match...")
            state = map_node(state)
            claimed = sum(1 for m in state["mappings"] if m.satisfied)
            st.write(f"Model claimed {claimed} controls were satisfied.")

            st.write("**Stage 4 — Hard-gate validate:** checking every claim against the real text...")
            state = validate_node(state)
            st.write(
                f"{len(state['validated_mappings'])} claims verified, "
                f"{len(state['rejected_mappings'])} rejected as unverifiable."
            )

            st.write("**Stage 5 — Report:** writing the final gap report...")
            state = report_node(state)

            status.update(label="Done", state="complete", expanded=False)

        st.subheader("Gap report")
        st.markdown(state["report"])

        with st.expander("Verified control matches (evidence-cited)"):
            for m in state["validated_mappings"]:
                st.markdown(f"**{m.control_id} — {m.title}**")
                st.markdown(f"> {m.evidence_sentence}")
                st.caption(f"Confidence: {m.confidence}")

        if state["rejected_mappings"]:
            with st.expander(f"Rejected claims ({len(state['rejected_mappings'])}) — for transparency"):
                st.caption(
                    "These were claimed by the model but could not be verified "
                    "against the real document text, so they're excluded from "
                    "the report above."
                )
                for r in state["rejected_mappings"]:
                    st.markdown(f"**{r.mapping.control_id} — {r.mapping.title}**")
                    st.caption(r.rejection_reason)

        os.unlink(tmp_path)
