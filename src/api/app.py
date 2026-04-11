from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from src.api.schemas import HealthResponse, IndexRequest, IndexResponse, QueryRequest, QueryResponse
from src.api.service import TeleRAGService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(title="TeleRAG API", version="1.0.0")
service = TeleRAGService()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(**service.health())


@app.post("/index", response_model=IndexResponse)
def index_document(payload: IndexRequest) -> IndexResponse:
    try:
        result = service.index_document(
            file_path=payload.file_path,
            chunk_size=payload.chunk_size,
            overlap=payload.overlap,
            persist=payload.persist,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Index build failed: {exc}") from exc
    return IndexResponse(**result)


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    try:
        result = service.ask(
            query=payload.query,
            top_k=payload.top_k,
            enable_rerank=payload.enable_rerank,
            max_new_tokens=payload.max_new_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return QueryResponse(**result)
