"""
nodes_ingest_retrieve.py
Step 5, part 1: the two nodes that don't need any API key at all —
Ingest (read + chunk the document) and Retrieve (search the knowledge base).

These can be tested completely on their own before touching Groq.
"""

import re
import chromadb
from pypdf import PdfReader
from build_retriever import USE_LOCAL_FALLBACK, TfidfEmbeddingFunction, load_knowledge_base


def extract_text(file_path: str) -> str:
    """Pull raw text out of a PDF or plain text file."""
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


def chunk_text(raw_text: str, min_chunk_len: int = 40) -> list[str]:
    """Split into paragraph-ish chunks. Simple rule: split on blank lines and
    on sentence groups, drop anything too short to be meaningful (headers,
    stray whitespace)."""
    # First split on blank lines (paragraphs)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]

    chunks = []
    for para in paragraphs:
        # If a paragraph is long, break it into 2-4 sentence groups
        sentences = re.split(r"(?<=[.!?])\s+", para)
        buffer = []
        for s in sentences:
            buffer.append(s)
            if len(buffer) >= 3:
                chunk = " ".join(buffer).strip()
                if len(chunk) >= min_chunk_len:
                    chunks.append(chunk)
                buffer = []
        if buffer:
            chunk = " ".join(buffer).strip()
            if len(chunk) >= min_chunk_len:
                chunks.append(chunk)

    return chunks


def ingest_node(state: dict) -> dict:
    """LangGraph node: state must contain 'file_path'. Adds 'chunks'."""
    raw_text = extract_text(state["file_path"])
    chunks = chunk_text(raw_text)
    print(f"[Ingest] Extracted {len(raw_text)} characters, split into {len(chunks)} chunks.")
    return {**state, "raw_text": raw_text, "chunks": chunks}


def retrieve_node(state: dict, db_path: str = "chroma_store", top_k: int = 6) -> dict:
    """LangGraph node: state must contain 'chunks'. Adds 'retrieved'.

    Important: this must use the exact same embedding function build_retriever.py
    used when it created the collection — a semantic-model collection can't be
    queried with a TF-IDF vectorizer and vice versa. We import the same flag and
    class from build_retriever.py so the two files can never drift out of sync.

    top_k default raised from 3 -> 6. With only 3 candidates per chunk, a
    control that's a strong-but-not-top-ranked semantic match (e.g. a precise
    DPDP clause sitting just behind a more generic ISO control) never even
    reaches the Map step, so the LLM has no chance to judge it. Widening the
    net costs a bit more Map-step tokens per chunk, but Map still judges each
    candidate independently and the hard-gate Validate step downstream throws
    out anything unverifiable — so casting wider is low-risk, it mainly adds
    recall, not noise that survives to the final report."""
    client = chromadb.PersistentClient(path=db_path)

    if USE_LOCAL_FALLBACK:
        # Re-fit the TF-IDF vectorizer on the same knowledge-base corpus used
        # at build time, so query vectors land in the same vector space.
        kb_texts = [r["text"] for r in load_knowledge_base()]
        embedding_fn = TfidfEmbeddingFunction(corpus=kb_texts)
        collection = client.get_collection("governance_kb", embedding_function=embedding_fn)
    else:
        collection = client.get_collection("governance_kb")

    retrieved = []
    for i, chunk in enumerate(state["chunks"]):
        results = collection.query(query_texts=[chunk], n_results=top_k)
        candidates = [
            {
                "id": doc_id,
                "text": doc,
                "metadata": meta,
            }
            for doc_id, doc, meta in zip(
                results["ids"][0], results["documents"][0], results["metadatas"][0]
            )
        ]
        retrieved.append({"chunk_id": i, "chunk_text": chunk, "candidates": candidates})

    print(f"[Retrieve] Found candidate matches for {len(retrieved)} chunks.")
    return {**state, "retrieved": retrieved}


# --- Self-test: run this file directly to test Ingest + Retrieve on a sample
# document, with zero API key needed.
if __name__ == "__main__":
    import os

    # Create a tiny sample policy document to test against, if one doesn't exist
    sample_path = "sample_policy.txt"
    if not os.path.exists(sample_path):
        with open(sample_path, "w") as f:
            f.write(
                "Access Control Policy\n\n"
                "All employee laptops must be encrypted using full-disk encryption "
                "and must automatically lock after 10 minutes of inactivity. "
                "This applies to all company-owned devices.\n\n"
                "We conduct background verification checks on all new hires before "
                "their start date, in line with role sensitivity. Verification records "
                "are retained by HR for the duration of employment.\n\n"
                "Users may submit a request to access a copy of the personal data "
                "we hold about them at any time, and we will respond within 30 days."
            )
        print(f"Created a sample test document at {sample_path}")

    state = {"file_path": sample_path}
    state = ingest_node(state)
    state = retrieve_node(state)

    for r in state["retrieved"]:
        print(f"\nChunk {r['chunk_id']}: \"{r['chunk_text'][:70]}...\"")
        for c in r["candidates"][:2]:
            label = c["metadata"].get("title") or c["metadata"].get("topic")
            print(f"   -> {c['id']}: {label}")
