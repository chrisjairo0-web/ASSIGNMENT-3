from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import streamlit as st

from src.doc_ingestion import ingest_pdf
from src.hybrid_retriever import HybridRetriever
from src.response_generator import (
    generate_llm_only_answer,
    generate_rag_answer,
    retrieval_only_answer,
    validate_gemini_provider,
)
from src.ui_styles import apply_styles

DEFAULT_PDF_PATH = "data/document.pdf"
DEFAULT_MODEL = "gemini-2.5-flash"


def _execution_case(selected_mode: str, provider_ok: bool, temperature: float) -> str:
    if not provider_ok:
        return "RAG-puro (retrieval-only, sin Gemini)"

    if selected_mode == "LLM-only":
        return f"Gemini-puro (LLM-only, T={temperature:.2f})"

    if temperature > 0.0:
        return f"RAG+Gemini (T={temperature:.2f})"

    return "RAG+Gemini determinista (T=0.00)"


def _resolve_api_key(manual_value: Optional[str]) -> tuple[Optional[str], str]:
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key.strip(), "environment variable"

    try:
        secret_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        secret_key = None

    if secret_key:
        return str(secret_key).strip(), "streamlit secrets"

    if manual_value:
        return manual_value.strip(), "sidebar session input"

    return None, "none"


def _show_pipeline() -> None:
    with st.sidebar.expander("RAG Pipeline", expanded=True):
        st.markdown(
            """
1. Document ingestion  
2. Chunking  
3. Embedding  
4. Retrieval (BM25 + semantic)  
5. Reranking  
6. Answer generation
            """
        )


def _example_queries() -> st.container | None:
    """Display example queries as interactive buttons that pre-fill the input field."""
    examples = [
        "What is C-MAPSS?",
        "What assumptions does this model make?",
        "How are engine degradation effects represented?",
        "What does the closed-loop analysis section cover?",
    ]

    with st.expander("Example Queries", expanded=True):
        st.caption("Click any question to load it into the field below:")
        cols = st.columns(2)
        for idx, query_text in enumerate(examples):
            col = cols[idx % 2]
            with col:
                if st.button(
                    query_text,
                    key=f"example_btn_{idx}",
                    use_container_width=True,
                ):
                    st.session_state.query_input = query_text
                    st.rerun()

    return None


def _render_retrieved_chunks(chunks: list[dict]) -> None:
    st.subheader("Retrieved Chunks")
    if not chunks:
        st.info("No chunks were retrieved for this query.")
        return

    for chunk in chunks:
        meta = chunk.get("metadata", {})
        st.markdown(
            f"""
<div class=\"demo-card\">
  <div><strong>Rank {chunk['rank']}</strong> | Score: {chunk['score']} | Confidence: {chunk['confidence']}</div>
  <div class=\"demo-meta\">Page {chunk.get('page_number', 'N/A')} | Section {chunk.get('section', 'N/A')} | Chunk #{meta.get('chunk_index', 'N/A')}</div>
  <div>{chunk.get('snippet', '')}</div>
</div>
            """,
            unsafe_allow_html=True,
        )


def _render_answer(answer_result: dict, mode_label: str) -> None:
    st.subheader("Answer")
    st.markdown(f"<div class='mode-badge'>{mode_label}</div>", unsafe_allow_html=True)

    grounded = answer_result.get("grounded", False)
    grounded_cls = "grounded-yes" if grounded else "grounded-no"
    grounded_text = "Grounded in retrieved context" if grounded else "Not grounded / fallback"

    st.markdown(
        f"<div class='{grounded_cls}'>{grounded_text}</div>",
        unsafe_allow_html=True,
    )
    st.write(answer_result.get("answer", ""))

    if answer_result.get("error_message"):
        st.caption(f"Provider note: {answer_result['error_message']}")


def _render_provider_status(status: str, source: str) -> None:
    st.sidebar.markdown(f"**Gemini provider status:** {status}")
    st.sidebar.markdown(f"**API key source:** {source}")


def main() -> None:
    st.set_page_config(page_title="Minimal RAG PDF Demo", layout="wide")
    apply_styles()

    st.title("Minimal Python RAG Demo")
    st.caption("Grounded QA over technical PDFs with RAG vs LLM-only comparison.")

    _show_pipeline()

    st.sidebar.header("Inputs")
    pdf_input_mode = st.sidebar.radio("PDF source", ["File path", "Upload PDF"], index=0)
    pdf_path = st.sidebar.text_input("PDF path", value=DEFAULT_PDF_PATH)
    uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

    st.sidebar.header("Gemini")
    if "manual_api_key" not in st.session_state:
        st.session_state.manual_api_key = ""

    st.session_state.manual_api_key = st.sidebar.text_input(
        "Temporary API key (session only)",
        type="password",
        value=st.session_state.manual_api_key,
    )

    api_key, key_source = _resolve_api_key(st.session_state.manual_api_key)
    provider_ok, provider_status = validate_gemini_provider(api_key, DEFAULT_MODEL)
    _render_provider_status(provider_status, key_source)

    if provider_ok:
        available_modes = ["RAG", "LLM-only"]
    else:
        available_modes = ["RAG"]
        st.sidebar.info("LLM-only mode is disabled until a valid Gemini API key is configured.")

    selected_mode = st.sidebar.radio(
        "Mode",
        available_modes,
        help="RAG Mode (grounded answers) or LLM-only Mode (may hallucinate)",
    )
    show_comparison = st.sidebar.checkbox(
        "Show optional RAG vs LLM-only comparison",
        value=False,
        disabled=not provider_ok,
    )

    if "gemini_temperature" not in st.session_state:
        st.session_state.gemini_temperature = 0.0

    st.sidebar.markdown("**Gemini temperature**")
    temp_cols = st.sidebar.columns(3)
    if temp_cols[0].button("T=0.0", use_container_width=True):
        st.session_state.gemini_temperature = 0.0
    if temp_cols[1].button("T=0.2", use_container_width=True):
        st.session_state.gemini_temperature = 0.2
    if temp_cols[2].button("T=0.7", use_container_width=True):
        st.session_state.gemini_temperature = 0.7

    st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.5,
        step=0.1,
        key="gemini_temperature",
        help=(
            "Controls randomness in Gemini responses. T=0 is more deterministic; "
            "higher values are more variable."
        ),
    )

    current_case = _execution_case(
        selected_mode=selected_mode,
        provider_ok=provider_ok,
        temperature=float(st.session_state.gemini_temperature),
    )
    st.sidebar.markdown(f"**Current execution case:** {current_case}")

    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = None
    if "source_id" not in st.session_state:
        st.session_state.source_id = None
    if "index_timestamp" not in st.session_state:
        st.session_state.index_timestamp = None
    if "document_name" not in st.session_state:
        st.session_state.document_name = None
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""

    st.markdown("---")
    st.subheader("📑 Document Indexing")
    
    col_idx_a, col_idx_b = st.columns([2, 1])
    with col_idx_a:
        st.write("Step 1️⃣: Index your document (one-time setup)")
    with col_idx_b:
        index_button = st.button("Index Document", use_container_width=True, type="primary")
    
    if index_button:
        with st.spinner("Indexing document..."):
            progress = st.progress(0)
            status = st.empty()

            uploaded_bytes = uploaded_pdf.getvalue() if uploaded_pdf is not None else None
            chosen_path = None if pdf_input_mode == "Upload PDF" else pdf_path

            status.write("1/3 Ingesting document...")
            progress.progress(20)
            chunks, source_id, resolved_pdf_path = ingest_pdf(
                pdf_path=chosen_path,
                uploaded_pdf_bytes=uploaded_bytes,
            )

            status.write("2/3 Building retriever (BM25 + embeddings + reranking)...")
            progress.progress(55)
            retriever = HybridRetriever(chunks=chunks, source_id=source_id)
            st.session_state.retriever = retriever
            st.session_state.chunks = chunks
            st.session_state.source_id = source_id
            st.session_state.document_name = str(resolved_pdf_path)
            st.session_state.chunk_count = len(chunks)
            
            from datetime import datetime
            st.session_state.index_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            progress.progress(100)
            status.write("✓ Document indexed successfully!")
        
        st.success(f"✓ Ready to query! Indexed {len(chunks)} chunks from {Path(resolved_pdf_path).name}")

    if st.session_state.retriever is not None:
        st.info(
            f"✓ Document indexed: **{st.session_state.document_name}** "
            f"({st.session_state.chunk_count} chunks, indexed at {st.session_state.index_timestamp})"
        )
    else:
        st.warning("⚠️ No document indexed yet. Click 'Index Document' to begin.")

    st.markdown("---")
    st.subheader("❓ Ask Your Question")
    
    st.write("**Example queries** — Click any button to load it into the field:")
    _example_queries()
    
    col_q_a, col_q_b = st.columns([2, 1])
    with col_q_a:
        query = st.text_input(
            "Step 2️⃣: Enter your question",
            value=st.session_state.query_input,
            placeholder="Example: What assumptions does this model make?",
            key="query_field",
        )
        st.session_state.query_input = query
    with col_q_b:
        run_query = st.button("Query", use_container_width=True)

    if st.session_state.retriever is None:
        st.info("📌 Index a document first before asking questions.")
    elif run_query or (query and st.session_state.get("trigger_query", False)):
        if not query.strip():
            st.warning("⚠️ Please enter a question to continue.")
        else:
            st.session_state.trigger_query = False
            
            progress = st.progress(0)
            status = st.empty()

            status.write("Retrieving evidence...")
            progress.progress(50)
            retrieved = st.session_state.retriever.retrieve(query=query, top_n=5)
            progress.progress(100)
            status.empty()

            if selected_mode == "LLM-only":
                answer_result = generate_llm_only_answer(
                    question=query,
                    api_key=api_key,
                    model_name=DEFAULT_MODEL,
                    temperature=float(st.session_state.gemini_temperature),
                )
                _render_answer(
                    answer_result,
                    f"Gemini-puro (LLM-only, T={float(st.session_state.gemini_temperature):.2f})",
                )
                st.info("ℹ️ No retrieval context used in LLM-only mode. Results may be unreliable.")
            else:
                if provider_ok:
                    answer_result = generate_rag_answer(
                        question=query,
                        retrieved_chunks=retrieved,
                        api_key=api_key,
                        model_name=DEFAULT_MODEL,
                        temperature=float(st.session_state.gemini_temperature),
                    )
                    if float(st.session_state.gemini_temperature) > 0.0:
                        mode_label = (
                            f"RAG+Gemini (grounded answers, T={float(st.session_state.gemini_temperature):.2f})"
                        )
                    else:
                        mode_label = "RAG+Gemini determinista (grounded answers, T=0.00)"
                else:
                    answer_result = retrieval_only_answer(query, retrieved)
                    answer_result["provider_status"] = "fallback mode active"
                    mode_label = "RAG-puro (retrieval-only, sin Gemini)"

                _render_answer(answer_result, mode_label)
                _render_retrieved_chunks(retrieved)

            if show_comparison and provider_ok:
                st.divider()
                st.subheader("📊 Optional Mode Comparison")
                rag_answer = generate_rag_answer(
                    question=query,
                    retrieved_chunks=retrieved,
                    api_key=api_key,
                    model_name=DEFAULT_MODEL,
                    temperature=float(st.session_state.gemini_temperature),
                )
                llm_only_answer = generate_llm_only_answer(
                    question=query,
                    api_key=api_key,
                    model_name=DEFAULT_MODEL,
                    temperature=float(st.session_state.gemini_temperature),
                )

                c1, c2 = st.columns(2)
                with c1:
                    if float(st.session_state.gemini_temperature) > 0.0:
                        st.write("**RAG+Gemini (Grounded)**")
                        rag_label = (
                            f"RAG+Gemini (grounded answers, T={float(st.session_state.gemini_temperature):.2f})"
                        )
                    else:
                        st.write("**RAG+Gemini determinista (Grounded)**")
                        rag_label = "RAG+Gemini determinista (grounded answers, T=0.00)"

                    _render_answer(rag_answer, rag_label)
                with c2:
                    st.write("**Gemini-puro (Ungrounded)**")
                    _render_answer(
                        llm_only_answer,
                        f"Gemini-puro (LLM-only, T={float(st.session_state.gemini_temperature):.2f})",
                    )


if __name__ == "__main__":
    main()
