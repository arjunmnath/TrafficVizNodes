"""Factory for instantiating VQA reasoning adapters."""

from __future__ import annotations

from inference_node.vqa.base import BaseVQAReasoner


def get_vqa_reasoner(model_name: str, device: str = "auto") -> BaseVQAReasoner:
    """Factory to resolve a model name to its respective VQA Reasoner adapter.

    Args:
        model_name: Name/path of the model (e.g. microsoft/Florence-2-base-ft,
          Qwen/Qwen2-VL-2B-Instruct)
        device: Device to load the model on ("auto", "cuda", "mps", "cpu")

    Returns:
        An instance of BaseVQAReasoner
    """
    model_name_lower = model_name.lower()
    if "florence" in model_name_lower:
        from inference_node.vqa.florence import FlorenceReasoner
        return FlorenceReasoner(model_name=model_name, device=device)
    elif "qwen" in model_name_lower:
        from inference_node.vqa.qwen import QwenVLMReasoner
        return QwenVLMReasoner(model_name=model_name, device=device)
    else:
        raise ValueError(
            f"Unsupported VQA reasoning model: '{model_name}'. "
            "Supported models include Florence-2 and Qwen2-VL family models."
        )
