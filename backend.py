from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from RAG import hybrid_answer

app = FastAPI(title="Spotify Architecture Hybrid RAG API", version="1.0.0")

class Question(BaseModel):
    question: str
    k: int = 5

@app.get("/")
def health():
    return {"status": "running", "project": "Spotify Architecture Hybrid RAG"}

@app.post("/ask")
def ask(payload: Question):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        return hybrid_answer(payload.question, payload.k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
