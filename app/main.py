"""FastAPI app exposing /ask, /contradict, /ingest, /health."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import ingestion, qa

app = FastAPI(title="Potens RAG - Document Q&A with Citations")


class AskRequest(BaseModel):
    question: str


class ContradictRequest(BaseModel):
    doc_a: str
    doc_b: str
    topic: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest():
    """Re-runs ingestion over everything in /docs. Run this once before asking questions,
    and again any time you add or change a source document."""
    try:
        return ingestion.ingest_documents()
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ask")
def ask(req: AskRequest):
    return qa.answer_question(req.question)


@app.post("/contradict")
def contradict(req: ContradictRequest):
    return qa.contradict(req.doc_a, req.doc_b, req.topic)
