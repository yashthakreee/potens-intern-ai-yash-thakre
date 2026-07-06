"""
Ingestion pipeline: PDF -> per-page text -> overlapping character chunks -> embed -> Chroma.

Two design decisions worth calling out (both explained fully in README):

1. Chunking PER PAGE rather than across the whole document in one continuous
   stream. Costs a little cross-page context, but every chunk can then cite an
   exact, verifiable page number -- which the assignment explicitly requires
   ("chunk or page reference").

2. Embeddings via scikit-learn TF-IDF, not a downloaded neural embedding model.
   This keeps the whole pipeline offline-reproducible: no model download that
   can fail/rate-limit right before a demo, no API key needed just to search.
   The honest tradeoff: TF-IDF matches on shared words, not paraphrases, so a
   query using totally different wording than the source text can miss. Good
   enough for keyword-rich policy/product docs in a 24h build; a next step
   would be swapping in a sentence-transformer or API embedding for better
   semantic recall (see README "What I'd build next").
"""
import os
import glob
import pickle
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
import chromadb

from app import config


def _chunk_text(text: str, size: int, overlap: int):
    """Simple sliding-window chunker over raw characters."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap  # step forward, re-sharing the last `overlap` chars
    return chunks


def load_and_chunk_pdf(path: str):
    """Returns a list of dicts: {text, source, page, chunk_index}."""
    reader = PdfReader(path)
    source_name = os.path.basename(path)
    records = []
    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for i, chunk in enumerate(_chunk_text(page_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            records.append({
                "text": chunk,
                "source": source_name,
                "page": page_num,
                "chunk_index": i,
            })
    return records


def ingest_documents(docs_dir: str = "docs"):
    """
    Walks docs_dir, ingests every PDF found, fits a fresh TF-IDF vectorizer over
    ALL chunks, and rebuilds the Chroma collection from scratch.

    Rebuilding from scratch (not upserting) on every run is intentional: the
    vectorizer's vocabulary depends on the full document set, so if you add a
    new PDF later, old vectors would no longer be comparable to new ones unless
    everything is re-embedded together.
    """
    pdf_paths = sorted(glob.glob(os.path.join(docs_dir, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {docs_dir}/ -- add your source documents first.")

    all_ids, all_texts, all_metadatas = [], [], []
    for path in pdf_paths:
        records = load_and_chunk_pdf(path)
        for r in records:
            uid = f"{r['source']}::p{r['page']}::c{r['chunk_index']}"
            all_ids.append(uid)
            all_texts.append(r["text"])
            all_metadatas.append({
                "source": r["source"],
                "page": r["page"],
                "chunk_index": r["chunk_index"],
            })

    # Fit TF-IDF over every chunk from every document, then persist the fitted
    # vectorizer so retrieval.py can encode future queries the exact same way.
    vectorizer = TfidfVectorizer(max_features=config.TFIDF_MAX_FEATURES, stop_words="english")
    vectors = vectorizer.fit_transform(all_texts).toarray().tolist()

    os.makedirs(os.path.dirname(config.VECTORIZER_PATH), exist_ok=True)
    with open(config.VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    # Drop and recreate so the collection's vector space always matches the
    # vectorizer we just fit (see docstring above).
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    # hnsw:space="cosine" is important here: with the default (squared L2),
    # a totally irrelevant query (zero vocabulary overlap -> zero vector) can
    # score BETWEEN a weak-but-real match and a strong match, because a zero
    # vector sits at a fixed "medium" distance from everything. Cosine space
    # fixes this: distance = 1 - cosine_similarity, so zero overlap always
    # lands at distance 1 (the true minimum of relevance), never in the middle.
    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(ids=all_ids, embeddings=vectors, documents=all_texts, metadatas=all_metadatas)

    return {
        "documents_ingested": len(pdf_paths),
        "chunks_stored": len(all_ids),
        "files": [os.path.basename(p) for p in pdf_paths],
        "vocabulary_size": len(vectorizer.vocabulary_),
    }


if __name__ == "__main__":
    result = ingest_documents()
    print(result)
