"""Retrieval: encode the query with the SAME fitted TF-IDF vectorizer used at
ingestion time, then fetch the top-k most similar chunks from Chroma."""
import pickle
import chromadb
from app import config

_vectorizer_cache = None


def _get_vectorizer():
    global _vectorizer_cache
    if _vectorizer_cache is None:
        with open(config.VECTORIZER_PATH, "rb") as f:
            _vectorizer_cache = pickle.load(f)
    return _vectorizer_cache


def _get_collection():
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    return client.get_collection(name=config.COLLECTION_NAME)


def retrieve(query: str, top_k: int = None, source_filter: str = None):
    vectorizer = _get_vectorizer()
    collection = _get_collection()
    top_k = top_k or config.TOP_K

    query_vector = vectorizer.transform([query]).toarray()[0].tolist()
    where = {"source": source_filter} if source_filter else None
    results = collection.query(query_embeddings=[query_vector], n_results=top_k, where=where)

    hits = []
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for i in range(len(ids)):
        distance = dists[i]
        # Collection was created with hnsw:space="cosine", so distance is
        # exactly (1 - cosine_similarity). TF-IDF entries are non-negative,
        # so cosine similarity is always >= 0 -- confidence is a clean 0-1
        # scale where 0 truly means "no shared vocabulary at all".
        confidence = max(0.0, 1 - distance)
        hits.append({
            "id": ids[i],
            "text": docs[i],
            "source": metas[i]["source"],
            "page": metas[i]["page"],
            "confidence": round(confidence, 3),
        })
    return hits
