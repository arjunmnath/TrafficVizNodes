"""Base class for Visual Question Answering (VQA) reasoning engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch

from inference_node.vqa.types import CandidateImage, RankedResult


class BaseVQAReasoner(ABC):
    """Abstract base class for VQA reasoning engines (adapters)."""

    @abstractmethod
    def answer(
        self,
        query: str,
        candidates: List[CandidateImage],
        top_k: int = 5,
    ) -> List[RankedResult]:
        """Score each candidate image against the query and return top-K results."""
        pass

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _resolve_dtype(device: str) -> torch.dtype:
        if device in ("cuda", "mps"):
            return torch.bfloat16 if device == "cuda" else torch.float16
        return torch.float32
