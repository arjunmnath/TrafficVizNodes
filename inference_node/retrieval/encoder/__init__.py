"""Retrieval encoder implementations and factory."""

from inference_node.retrieval.encoder.base import BaseRetrievalEncoder
from inference_node.retrieval.encoder.siglip2 import SigLIP2Encoder
from inference_node.retrieval.encoder.openclip import OpenCLIPEncoder
from inference_node.retrieval.encoder.evaclip import EVACLIPEncoder
from inference_node.retrieval.encoder.factory import get_retrieval_encoder

__all__ = [
    "BaseRetrievalEncoder",
    "SigLIP2Encoder",
    "OpenCLIPEncoder",
    "EVACLIPEncoder",
    "get_retrieval_encoder",
]
