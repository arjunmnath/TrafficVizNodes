"""VQA reasoning module with adapter pattern."""

from inference_node.vqa.base import BaseVQAReasoner
from inference_node.vqa.factory import get_vqa_reasoner
from inference_node.vqa.florence import FlorenceReasoner
from inference_node.vqa.qwen import QwenVLMReasoner
from inference_node.vqa.types import CandidateImage, RankedResult


def answer(
    reasoner: BaseVQAReasoner,
    query: str,
    candidate_images: list[CandidateImage],
    top_k: int = 5,
) -> list[RankedResult]:
    """Public interface: score retrieved candidates with a VQA reasoner."""
    return reasoner.answer(query=query, candidates=candidate_images, top_k=top_k)


__all__ = [
    "BaseVQAReasoner",
    "get_vqa_reasoner",
    "FlorenceReasoner",
    "QwenVLMReasoner",
    "CandidateImage",
    "RankedResult",
    "answer",
]
