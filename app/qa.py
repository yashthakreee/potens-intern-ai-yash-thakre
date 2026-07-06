"""
Core question-answering logic: retrieve -> hallucination guard -> generate -> cite.
Kept separate from main.py (the FastAPI layer) so it can be tested directly,
without spinning up a server, and so main.py stays a thin routing layer.
"""
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # makes langdetect deterministic across runs

from app import config, retrieval, llm

NOT_COVERED_MESSAGE = "The provided documents do not cover this question."


def _detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"  # default to English if detection fails on very short input


def _translate(text: str, target_lang: str) -> str:
    """
    Translation-at-the-boundary, done via the same LLM call rather than a
    separate translation API. The brief explicitly allows this for the 24h
    version. Used both directions: question -> English (for retrieval) and
    English answer -> original question language (for the response).
    """
    if target_lang == "en":
        return text
    prompt = (
        f"Translate the following text to language code '{target_lang}'. "
        f"Return ONLY the translated text, nothing else, no quotes.\n\nText: {text}"
    )
    return llm.generate(prompt)


def answer_question(question: str) -> dict:
    # Step 1: detect the question's language, translate to English for retrieval
    # (our TF-IDF vocabulary was built from English-language source documents).
    query_lang = _detect_language(question)
    english_question = question if query_lang == "en" else _translate(question, "en")

    # Step 2: retrieve evidence.
    hits = retrieval.retrieve(english_question)
    top_confidence = hits[0]["confidence"] if hits else 0.0

    # Step 3: hallucination guard. If the best match is weak, refuse instead
    # of letting the LLM improvise an answer from its own training data.
    if not hits or top_confidence < config.CONFIDENCE_THRESHOLD:
        answer_text = NOT_COVERED_MESSAGE
        try:
            if query_lang != "en":
                answer_text = _translate(answer_text, query_lang)
        except RuntimeError:
            pass  # no LLM key configured -- still return the English refusal, not a crash
        return {
            "answer": answer_text,
            "citations": [],
            "confidence": top_confidence,
            "covered_by_documents": False,
        }

    # Step 4: build a strict, cite-by-index prompt from the retrieved chunks.
    # The LLM only ever sees numbered snippets -- it cannot cite a source we
    # didn't actually retrieve, because the citation list below is built from
    # OUR metadata, not parsed out of the model's free-text response.
    context_block = "\n\n".join(
        f"[{i + 1}] (source: {h['source']}, page: {h['page']})\n{h['text']}"
        for i, h in enumerate(hits)
    )
    prompt = (
        "Answer the question using ONLY the numbered context chunks below. "
        "Reference chunks by their number in brackets, e.g. [1]. "
        "If the chunks do not actually answer the question, reply with EXACTLY "
        f"this sentence and nothing else: \"{NOT_COVERED_MESSAGE}\"\n\n"
        f"Context:\n{context_block}\n\nQuestion: {english_question}\n\nAnswer:"
    )

    citations = [
        {"source": h["source"], "page": h["page"], "snippet": h["text"][:200]}
        for h in hits
    ]

    try:
        answer_text = llm.generate(prompt)
    except RuntimeError as e:
        # No LLM key yet -- still return the real retrieved evidence so the
        # retrieval half of the system is demonstrably working on its own.
        return {
            "answer": None,
            "citations": citations,
            "confidence": top_confidence,
            "covered_by_documents": True,
            "error": str(e),
        }

    refused = NOT_COVERED_MESSAGE in answer_text
    if query_lang != "en":
        answer_text = _translate(answer_text, query_lang)

    return {
        "answer": answer_text,
        "citations": [] if refused else citations,
        "confidence": top_confidence,
        "covered_by_documents": not refused,
    }


def contradict(doc_a: str, doc_b: str, topic: str) -> dict:
    """
    Retrieves what each document says about `topic` independently, then asks
    the LLM to compare the two evidence sets. doc_a / doc_b are source
    filenames (the same strings shown in /ask citations), used as a metadata
    filter so we only look at chunks from that one file.
    """
    hits_a = retrieval.retrieve(topic, source_filter=doc_a, top_k=3)
    hits_b = retrieval.retrieve(topic, source_filter=doc_b, top_k=3)

    if not hits_a or not hits_b:
        return {
            "conflict": None,
            "reasoning": "Could not find enough content about this topic in one or both documents.",
            "doc_a_chunks_found": len(hits_a),
            "doc_b_chunks_found": len(hits_b),
        }

    block_a = "\n".join(f"- {h['text']}" for h in hits_a)
    block_b = "\n".join(f"- {h['text']}" for h in hits_b)
    prompt = (
        f"Document A ({doc_a}) says about '{topic}':\n{block_a}\n\n"
        f"Document B ({doc_b}) says about '{topic}':\n{block_b}\n\n"
        "Do these two documents conflict on this topic? Reply in this exact format:\n"
        "CONFLICT: yes/no\nREASONING: <one paragraph>"
    )

    try:
        raw = llm.generate(prompt)
    except RuntimeError as e:
        return {
            "conflict": None,
            "reasoning": f"LLM not configured: {e}",
            "doc_a_evidence": hits_a,
            "doc_b_evidence": hits_b,
        }

    return {
        "raw_model_output": raw,
        "doc_a_evidence": hits_a,
        "doc_b_evidence": hits_b,
    }
