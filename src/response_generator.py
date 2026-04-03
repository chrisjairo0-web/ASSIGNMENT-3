from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import google.generativeai as genai

STRICT_GROUNDING_POLICY = (
    "Answer ONLY using the provided retrieved context.\n"
    "If the answer is not contained in the context, say: "
    "'I cannot find this information in the document.'\n"
    "Do NOT fabricate or infer missing information."
)

FALLBACK_MESSAGE = "I cannot find this information in the document."


def _classify_provider_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "api key" in text or "permission" in text or "invalid" in text or "unauthorized" in text:
        return "invalid key"
    if "quota" in text or "rate" in text or "429" in text:
        return "quota issue"
    if "network" in text or "timeout" in text or "connection" in text:
        return "network error"
    return "fallback mode active"


def validate_gemini_provider(api_key: Optional[str], model_name: str) -> Tuple[bool, str]:
    if not api_key:
        return False, "fallback mode active"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Reply with the single word OK.")
        if getattr(response, "text", "").strip():
            return True, "working"
        return False, "fallback mode active"
    except Exception as exc:
        return False, _classify_provider_error(exc)


def build_context(chunks: List[Dict], max_chars: int = 6500) -> str:
    parts: List[str] = []
    total = 0
    for chunk in chunks:
        page = chunk.get("page_number")
        section = chunk.get("section")
        snippet = chunk.get("text", "")
        header = f"[Page {page} | Section {section}]\n"
        block = f"{header}{snippet}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)


def retrieval_only_answer(question: str, retrieved_chunks: List[Dict]) -> Dict:
    if not retrieved_chunks:
        return {
            "answer": FALLBACK_MESSAGE,
            "grounded": False,
            "used_fallback": True,
            "mode": "retrieval-only",
        }

    top = retrieved_chunks[0]
    confidence = float(top.get("confidence", 0.0))
    if confidence < 0.25:
        answer = FALLBACK_MESSAGE
        grounded = False
    else:
        section = top.get("section", "N/A")
        page = top.get("page_number", "N/A")
        snippet = top.get("snippet", "")
        answer = (
            f"Based on the top retrieved evidence (page {page}, section {section}), "
            f"the document states: {snippet}"
        )
        grounded = True

    return {
        "answer": answer,
        "grounded": grounded,
        "used_fallback": True,
        "mode": "retrieval-only",
    }


def generate_rag_answer(
    question: str,
    retrieved_chunks: List[Dict],
    api_key: Optional[str],
    model_name: str,
    temperature: float = 0.0,
) -> Dict:
    if not api_key:
        result = retrieval_only_answer(question, retrieved_chunks)
        result["provider_status"] = "fallback mode active"
        return result

    context = build_context(retrieved_chunks)
    prompt = (
        f"{STRICT_GROUNDING_POLICY}\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}\n"
    )

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": float(temperature)},
        )
        answer = (getattr(response, "text", "") or "").strip()
        if not answer:
            answer = FALLBACK_MESSAGE

        grounded = FALLBACK_MESSAGE.lower() not in answer.lower()
        return {
            "answer": answer,
            "grounded": grounded,
            "used_fallback": False,
            "mode": "rag",
            "provider_status": "working",
        }
    except Exception as exc:
        result = retrieval_only_answer(question, retrieved_chunks)
        result["provider_status"] = _classify_provider_error(exc)
        result["error_message"] = str(exc)
        return result


def generate_llm_only_answer(
    question: str,
    api_key: Optional[str],
    model_name: str,
    temperature: float = 0.0,
) -> Dict:
    if not api_key:
        return {
            "answer": "LLM-only mode requires a valid Gemini API key.",
            "grounded": False,
            "used_fallback": True,
            "mode": "llm-only-blocked",
            "provider_status": "fallback mode active",
        }

    prompt = (
        "You are a helpful assistant. Answer the following question using your "
        "general knowledge. No document context is provided — this is intentional. "
        "Answer as completely and clearly as you can.\n\n"
        f"Question: {question}\n"
    )

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": float(temperature)},
        )
        answer = (getattr(response, "text", "") or "").strip() or FALLBACK_MESSAGE
        return {
            "answer": answer,
            "grounded": False,
            "used_fallback": False,
            "mode": "llm-only",
            "provider_status": "working",
        }
    except Exception as exc:
        return {
            "answer": "LLM-only generation failed. Switch to RAG/retrieval-only mode.",
            "grounded": False,
            "used_fallback": True,
            "mode": "llm-only-failed",
            "provider_status": _classify_provider_error(exc),
            "error_message": str(exc),
        }
