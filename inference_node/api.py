from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from shared.schemas import QueryRequest, QueryResponse
from shared.utils import setup_logger
from inference_node.rag_pipeline import RAGPipeline
from inference_node.vector_store import VectorStore


logger = setup_logger("InferenceAPI")

# These are set by main.py before the app starts
_pipeline: RAGPipeline | None = None
_vector_store: VectorStore | None = None


def create_app(pipeline: RAGPipeline, vector_store: VectorStore) -> FastAPI:
    global _pipeline, _vector_store
    _pipeline = pipeline
    _vector_store = vector_store

    app = FastAPI(
        title="CCTV Inference Node",
        description="Text-to-timestamp CCTV footage search using VLM + RAG",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok", "model_loaded": _pipeline is not None}

    @app.get("/stats")
    async def stats():
        count = _vector_store.get_event_count() if _vector_store else 0
        return {
            "event_count": count,
            "status": "ready" if _pipeline else "initializing",
        }

    @app.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest):
        """Search CCTV footage by natural language description.

        Returns ranked timestamps with VLM-scored confidence and thumbnails.
        """
        if _pipeline is None:
            return QueryResponse(query=request.query, results=[])

        logger.info(f"Query: '{request.query}' (top_k={request.top_k})")

        results = _pipeline.query(
            query_text=request.query,
            top_k=request.top_k,
            camera_id=request.camera_id,
        )

        logger.info(f"Returning {len(results)} results")
        return QueryResponse(query=request.query, results=results)

    return app
