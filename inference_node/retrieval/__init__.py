"""Adapter-based visual retrieval — no VLM dependencies."""

from inference_node.retrieval.encoder import (
    BaseRetrievalEncoder,
    SigLIP2Encoder,
    OpenCLIPEncoder,
    get_retrieval_encoder,
)
from inference_node.retrieval.query_parser import ParsedQuery, parse_query
from inference_node.retrieval.search import RetrievalEngine, RetrievalResult
from inference_node.retrieval.vector_store import VectorStore

__all__ = [
    "BaseRetrievalEncoder",
    "SigLIP2Encoder",
    "OpenCLIPEncoder",
    "get_retrieval_encoder",
    "ParsedQuery",
    "parse_query",
    "RetrievalEngine",
    "RetrievalResult",
    "VectorStore",
]
