"""Streamlit UI for the custom RAG system."""

import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.chunker import ensure_processed_chunks
from modules.evidence import has_sufficient_evidence
from modules.answering import extract_answer
from modules.retrieval_engine import RetrievalEngine


@st.cache_resource
def load_rag_engine():
    """Initialize the retrieval engine once and reuse it for the session."""
    chunks = ensure_processed_chunks(
        output_path="processed_chunks.json",
        csv_path="Ghana_Election_Result.csv",
        budget_path="cleaned_budget_text.txt",
        chunk_size=800,
        force=True,
    )

    engine = RetrievalEngine()
    engine.index(chunks)
    return engine


def build_answer_context(results, max_chars: int = 1600) -> str:
    """Return a compact grounded context block for the UI."""
    blocks = []
    for idx, doc in enumerate(results, start=1):
        snippet = doc.text.strip().replace("\n", " ")
        if len(snippet) > max_chars:
            snippet = snippet[: max_chars - 3] + "..."
        blocks.append(f"[{idx}] {doc.source}: {snippet}")
    return "\n\n".join(blocks)


def main():
    st.set_page_config(page_title="Custom RAG System", page_icon="🔎", layout="wide")
    st.title("RAG System")
    st.caption(" With PyTorch, ChromaDB, and custom retrieval logic.")

    engine = load_rag_engine()

    query = st.text_input(
        "Ask a question",
        placeholder="e.g. What was allocated to education in the 2025 budget?",
    )

    if query:
        results = engine.retrieve(query, k=5, use_expansion=True)
        supported = has_sufficient_evidence(query, results)

        if supported:
            answer = extract_answer(query, results)
            if answer:
                st.subheader("Answer")
                st.success(answer)

            st.subheader("Retrieved evidence")
            for idx, doc in enumerate(results, start=1):
                with st.expander(f"Result {idx} · {doc.source} · score {doc.combined_score:.4f}"):
                    st.write(doc.text)
                    st.caption(
                        f"Vector: {doc.vector_score:.4f} | "
                        f"Keyword: {doc.keyword_score:.4f} | "
                        f"Combined: {doc.combined_score:.4f}"
                    )

            st.subheader("Grounding context")
            st.code(build_answer_context(results), language="text")
            st.info("These passages are retrieved evidence; the app does not generate a synthesized answer.")
        else:
            st.warning(
                "This question is outside the scope of this system, or the indexed files do not provide "
                "enough evidence. Its focus is the Ghana budget and election data in the available files. "
                "Try asking about a budget item or an election result."
            )
            if results:
                with st.expander("Closest matches (not sufficient evidence)"):
                    for doc in results:
                        st.write(f"{doc.source}: {doc.text}")


if __name__ == "__main__":
    main()
