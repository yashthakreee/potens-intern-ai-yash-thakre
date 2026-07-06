import os
from dotenv import load_dotenv

load_dotenv()

# --- Storage ---
CHROMA_PATH = os.getenv("CHROMA_PATH", "storage/chroma_db")
COLLECTION_NAME = "potens_docs"
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "storage/tfidf_vectorizer.pkl")

# --- Embeddings ---
# Design decision: TF-IDF (scikit-learn) instead of a downloaded neural embedding
# model. This keeps the whole pipeline offline-reproducible -- no dependency on
# a model download succeeding at demo time, no API key needed for embeddings at
# all. See README "Design Decisions" for the honest tradeoff (misses paraphrase
# matches a neural embedder would catch; fine for keyword-rich policy/product docs).
TFIDF_MAX_FEATURES = 4096

# --- Chunking ---
# Character-based (not token-based) chunking -- chosen for simplicity in a 24h build,
# see README "Design Decisions" for the full reasoning.
# Chunking happens PER PAGE, so every chunk can cite an exact page number.
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # characters shared between consecutive chunks on the same page

# --- Retrieval ---
TOP_K = 5

# Below this similarity score, we refuse to answer instead of guessing.
# confidence = 1 - cosine_distance (0 = zero vocabulary overlap, 1 = identical).
# Calibrated from a real manual test on the seed documents: a genuinely
# relevant match landed around 0.26-0.34, a clearly irrelevant query landed
# flat at 0.0. 0.15 is a deliberately conservative starting cutoff -- it is
# meant to be re-tuned once the eval set (stretch goal) exists with real
# right/wrong labels instead of a single manual spot-check.
CONFIDENCE_THRESHOLD = 0.15

# --- LLM generation (used for /ask, /contradict, and translation) ---
GEN_PROVIDER = os.getenv("GEN_PROVIDER", "groq")   # "groq" or "gemini"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
