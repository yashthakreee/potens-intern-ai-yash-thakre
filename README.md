# Document Q&A with Citations

A RAG system over 5+ real documents about AI-driven order-risk scoring for India's
social-commerce sellers (my own MSME hackathon project docs, plus supporting policy
documents). Retrieval-backed answers with citations, a refusal mechanism when the
docs don't cover a question, a document-vs-document contradiction check, and a
multilingual query flow.

## How to run it

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your source documents (PDFs) to /docs
#    Two are already included: Msme-idea.pdf and AI-Social-Commerce-Complete-Project-Document.pdf
#    Add 3+ more of your own to reach the 5-document minimum.

# 3. Set up your LLM key (needed for /ask answer generation, /contradict, and translation --
#    NOT needed for ingestion/retrieval, which run fully offline)
cp .env.example .env
# then edit .env and paste in a free-tier Groq or Gemini API key

# 4. Start the API
uvicorn app.main:app --reload --port 8000

# 5. Run ingestion once (re-run any time you add/change a document)
curl -X POST http://localhost:8000/ingest

# 6. Ask a question
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "why do so many COD orders get returned?"}'

# 7. Or use the UI instead of curl
streamlit run ui/streamlit_app.py
```

## Design decisions

**Chunking strategy: per-page, character-based, with overlap.** Each PDF page is
extracted separately and split into ~800-character windows with 150 characters of
overlap between consecutive windows on the *same page*. Chunking per page (rather
than treating the whole document as one continuous stream) costs a little
cross-page context, but it means every chunk can cite an exact, verifiable page
number, which the assignment requires. Character-based (not token-based) chunking
was chosen for simplicity given the time box; a token-aware chunker (e.g. via
`tiktoken`) would pack context more precisely and is a natural next step.

**Embeddings: TF-IDF (scikit-learn), not a downloaded neural embedding model.**
This was a deliberate call, not a shortcut: it keeps the whole retrieval pipeline
offline-reproducible, with no dependency on a model download succeeding right
before a demo, and no API key needed just to search. The honest tradeoff: TF-IDF
matches on shared vocabulary, not paraphrases -- a query using completely
different wording than the source text can under-match. This is a reasonable
fit for keyword-rich policy/product documents, but a real next step (see below)
would be a sentence-transformer or hosted embedding API for better semantic
recall on paraphrased queries.

**Vector store: Chroma, in cosine-distance mode.** Cosine distance was chosen
deliberately over Chroma's default (squared L2): with L2 distance, a query with
*zero* vocabulary overlap can score numerically *between* a strong match and a
weak-but-real match, because a zero vector sits at a fixed "medium" distance
from every stored vector. This actually happened during testing -- an
unrelated test query ("what is the capital of France") scored a higher
apparent "confidence" than a genuinely relevant query, under the L2 metric.
Switching the collection to cosine space fixed it: confidence is exactly
`1 - cosine_distance`, so zero overlap correctly lands at 0, the true bottom
of the relevance scale, and a real (if imperfect) match to correctly land
above it, every time.

**No hallucination, two layers deep.** (1) A confidence threshold on the top
retrieved chunk -- if it's below `CONFIDENCE_THRESHOLD` (see `app/config.py`),
the system returns "The provided documents do not cover this question"
*without even calling the LLM*. (2) A strict generation prompt instructing the
model to answer only from the numbered context it's given, and to output that
exact refusal sentence if the context doesn't actually answer the question.
Citations are never parsed out of the model's free text -- they're built
directly from our own retrieval metadata (source file + page + snippet), so
the model can't accidentally cite a source it wasn't actually given.

**Multilingual flow: translate at the boundary, via the same LLM call.** The
question's language is detected (`langdetect`), translated to English before
retrieval (since the TF-IDF vocabulary is built from English-language source
docs), and the final answer is translated back to the original language. The
brief explicitly allows this simplification for the 24-hour version rather
than requiring true cross-lingual embeddings.

**Contradiction check works by comparing independent retrievals, not a joint
one.** `/contradict` retrieves the top chunks from Document A and Document B
*separately*, filtered by a `source` metadata field, then asks the LLM to
compare the two evidence sets side by side and explain any conflict. This
means the contradiction check is only as good as whether the topic string
actually surfaces relevant chunks in both documents -- worth keeping topic
strings specific.

## What's broken or unfinished right now

- Only 2 of the 5+ required source documents are in `/docs` so far (my own
  project docs). Still need to add the rest.
- Generation, translation, and `/contradict` are wired up and structurally
  tested (they degrade gracefully with a clear error when no LLM key is
  present), but not yet tested with a real LLM key/response.
- `CONFIDENCE_THRESHOLD` (currently 0.15) is calibrated from a single manual
  spot-check, not a real eval set. It needs proper tuning.
- No automated tests yet (unit tests, eval set).
- Stretch goals not started: reranker, human-in-the-loop escalation UI,
  10-question eval set scored on retrieval@k.

## What I'd build next

1. Finish adding source documents to reach 5+, re-run ingestion, and do a real
   pass of manual questions to sanity-check citations and refusals.
2. Build the eval set (10 Q&A pairs with known correct chunks) and use it to
   properly tune `CONFIDENCE_THRESHOLD` instead of guessing.
3. Add a lightweight reranker (e.g. a small cross-encoder) on top of the
   top-10 TF-IDF results before picking the final top-5 -- would help recover
   from TF-IDF's weakness on paraphrased queries without needing a full
   switch to neural embeddings.
4. Swap TF-IDF for a real embedding model (sentence-transformers or a hosted
   embedding API) once I'm not developing inside a network-restricted
   environment, and compare retrieval quality against the eval set.
5. Confidence-gated human-in-the-loop path in the UI (flag low-but-not-zero
   confidence answers for manual review instead of a flat refuse/answer split).

## AI Use Log

- **Claude (Sonnet 5), via Cowork** -- approx. [fill in message/token count].
  Used for: pair-programming the entire backend (ingestion, retrieval,
  FastAPI routes, Streamlit UI), debugging the cosine-vs-L2 distance issue
  found during testing, and drafting this README's design-decisions section
  based on decisions made together during the build.
- [Add any other tool you use going forward -- Cursor, ChatGPT, Copilot,
  etc. -- with an approximate count and a one-line description of what it
  was used for.]
