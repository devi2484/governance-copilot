"""
build_retriever.py
Step 3 of the Governance Copilot build: embeds the ISO 27001 + DPDP knowledge
base into a local ChromaDB collection so the agent can do semantic search
instead of relying on hardcoded keyword matches.

Run once to build the DB: python3 build_retriever.py
The resulting DB is saved to ./chroma_store and is reused on every future run
(no need to re-embed unless the knowledge base JSON changes).

EMBEDDING MODEL NOTE:
By default this uses ChromaDB's bundled all-MiniLM-L6-v2 (ONNX) model, which
downloads automatically from S3 the first time it runs. That download needs
normal internet access — if you're testing inside a restricted/offline sandbox
and the download fails, set USE_LOCAL_FALLBACK = True below to use a TF-IDF
embedding instead. TF-IDF is weaker at true semantic matching (it matches on
shared vocabulary, not meaning) but needs zero network access and is a fine way
to sanity-check the retrieval pipeline works end-to-end before you have real
internet/API access.
"""

import json
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings

KB_DIR = "knowledge_base"
DB_DIR = "chroma_store"

# Flip to True if the default MiniLM model download is blocked in your environment.
USE_LOCAL_FALLBACK = False


class TfidfEmbeddingFunction(EmbeddingFunction):
    """Zero-network fallback embedding — fits a TF-IDF vectorizer on the
    knowledge base itself. Good enough to prove the pipeline works; swap back
    to the default MiniLM embedding (or a real API-based embedding) for the
    actual portfolio demo, since TF-IDF misses synonyms/paraphrases that a real
    semantic embedding model would catch."""

    def __init__(self, corpus):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=384)
        self.vectorizer.fit(corpus)

    def __call__(self, input: Documents) -> Embeddings:
        return self.vectorizer.transform(input).toarray().tolist()


def load_knowledge_base():
    """Flatten both KB files into one list of {id, text, metadata} records."""
    records = []

    with open(f"{KB_DIR}/iso27001_controls.json") as f:
        iso = json.load(f)
    for c in iso["controls"]:
        records.append({
            "id": f"ISO-{c['control_id']}",
            # what actually gets embedded — title + description gives the
            # embedding model enough semantic signal to match against real
            # policy-document sentences later
            "text": f"{c['title']}. {c['description']}",
            "metadata": {
                "source": "ISO27001",
                "control_id": c["control_id"],
                "theme": c["theme"],
                "title": c["title"],
            },
        })

    with open(f"{KB_DIR}/dpdp_clauses.json") as f:
        dpdp = json.load(f)
    for c in dpdp["clauses"]:
        records.append({
            "id": f"DPDP-{c['clause_id']}",
            "text": f"{c['topic']}. {c['summary']}",
            "metadata": {
                "source": "DPDP",
                "clause_id": c["clause_id"],
                "topic": c["topic"],
            },
        })

    return records


def build_collection():
    client = chromadb.PersistentClient(path=DB_DIR)

    # Wipe and rebuild each time this script runs, so the DB always matches
    # the current knowledge_base/*.json files exactly.
    try:
        client.delete_collection("governance_kb")
    except Exception:
        pass

    records = load_knowledge_base()
    texts = [r["text"] for r in records]

    if USE_LOCAL_FALLBACK:
        embedding_fn = TfidfEmbeddingFunction(corpus=texts)
        print("Using local TF-IDF fallback embedding (no network required).")
    else:
        embedding_fn = None  # let Chroma use its default MiniLM model
        print("Using ChromaDB's default MiniLM embedding (requires network access "
              "to download the model on first run).")

    collection = client.create_collection(
        name="governance_kb",
        metadata={"description": "ISO 27001 Annex A + DPDP Act 2023 reference"},
        embedding_function=embedding_fn,
    )

    collection.add(
        ids=[r["id"] for r in records],
        documents=texts,
        metadatas=[r["metadata"] for r in records],
    )

    print(f"Indexed {len(records)} knowledge-base entries into '{DB_DIR}'.")
    print(f"  - ISO 27001 controls: {sum(1 for r in records if r['metadata']['source']=='ISO27001')}")
    print(f"  - DPDP clauses: {sum(1 for r in records if r['metadata']['source']=='DPDP')}")
    return collection


def demo_query(collection, query_text, n_results=3):
    """Sanity-check: given a sample policy sentence, what controls does it match?"""
    results = collection.query(query_texts=[query_text], n_results=n_results)
    print(f"\nQuery: \"{query_text}\"")
    for i, (doc_id, meta, dist) in enumerate(zip(
        results["ids"][0], results["metadatas"][0], results["distances"][0]
    )):
        label = meta.get("title") or meta.get("topic")
        print(f"  {i+1}. [{doc_id}] {label}  (distance={dist:.3f})")


if __name__ == "__main__":
    coll = build_collection()

    # A few sample policy sentences to sanity-check retrieval quality
    demo_query(coll, "All employee laptops must be encrypted and screen-locked after 10 minutes of inactivity.")
    demo_query(coll, "We conduct background verification checks on all new hires before their start date.")
    demo_query(coll, "Users may request a copy of the personal data we hold about them at any time.")

