import json

from fastapi import FastAPI
from pydantic import BaseModel

from modules.evidence import has_sufficient_evidence
from modules.retrieval_engine import RetrievalEngine

app = FastAPI(title="Custom RAG API")

with open("processed_chunks.json", "r", encoding="utf-8") as handle:
    chunks = json.load(handle)

engine = RetrievalEngine()
engine.index(chunks)


class QueryRequest(BaseModel):
    query: str
    k: int = 5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query_documents(payload: QueryRequest):
    results = engine.retrieve(payload.query, k=payload.k, use_expansion=True)
    supported = has_sufficient_evidence(payload.query, results)
    return {
        "query": payload.query,
        "supported": supported,
        "message": None if supported else (
            "This question is outside the scope of this system, or the indexed files do not provide "
            "enough evidence. Its focus is the Ghana budget and election data in the available files."
        ),
        "results": [
            {
                "source": item.source,
                "text": item.text,
                "vector_score": round(item.vector_score, 4),
                "keyword_score": round(item.keyword_score, 4),
                "combined_score": round(item.combined_score, 4),
            }
            for item in results
        ],
    }
