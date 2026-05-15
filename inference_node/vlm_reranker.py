import torch
import json
import re
from PIL import Image
from typing import List
from dataclasses import dataclass
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from shared.utils import setup_logger


@dataclass
class CandidateFrame:
    """A candidate event with its extracted frame for VLM re-ranking."""
    camera_id: str
    timestamp: float
    video_pos_ms: float
    global_id: int
    class_label: str
    color: str
    type: str | None
    bbox: list | None
    frame: Image.Image  # Full frame or crop
    distance: float  # ChromaDB distance (lower = more similar)


@dataclass
class RankedResult:
    """A VLM-scored result."""
    camera_id: str
    timestamp: float
    video_pos_ms: float
    global_id: int
    class_label: str
    color: str
    type: str | None
    vlm_score: float
    vlm_explanation: str
    frame: Image.Image


class VLMReranker:
    """Re-ranks candidate frames using Qwen2.5-VL-7B-Instruct."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "auto"):
        self.logger = setup_logger("VLMReranker")
        self.logger.info(f"Loading VLM: {model_name} ...")

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.logger.info("VLM loaded successfully")

    def _build_prompt(self, query: str) -> str:
        return (
            "You are analyzing CCTV surveillance footage. "
            f'Does this image match the following description: "{query}"?\n'
            "Rate the match from 0 to 10 where 0 means no match and 10 means perfect match.\n"
            "Respond ONLY with valid JSON: {\"score\": <int>, \"explanation\": \"<brief reason>\"}"
        )

    def _parse_response(self, text: str) -> tuple[float, str]:
        """Extract score and explanation from VLM output."""
        # Try to parse as JSON first
        try:
            # Find JSON object in the response
            match = re.search(r'\{[^}]+\}', text)
            if match:
                data = json.loads(match.group())
                score = float(data.get("score", 0))
                explanation = data.get("explanation", "")
                return min(max(score, 0), 10), explanation
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: try to extract a number
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
        if numbers:
            score = float(numbers[0])
            return min(max(score, 0), 10), text.strip()

        return 0.0, text.strip()

    def rerank(
        self,
        query: str,
        candidates: List[CandidateFrame],
        top_k: int = 5,
    ) -> List[RankedResult]:
        """Score each candidate frame against the query using the VLM.

        Args:
            query: Natural language description to match.
            candidates: List of candidate frames from RAG retrieval.
            top_k: Number of top results to return.

        Returns:
            Sorted list of RankedResults (highest score first).
        """
        self.logger.info(f"Re-ranking {len(candidates)} candidates for query: '{query}'")
        results = []

        prompt_text = self._build_prompt(query)

        for candidate in candidates:
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": candidate.frame},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]

                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.model.device)

                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                    )

                # Decode only the generated tokens
                generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
                response = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

                score, explanation = self._parse_response(response)

                results.append(RankedResult(
                    camera_id=candidate.camera_id,
                    timestamp=candidate.timestamp,
                    video_pos_ms=candidate.video_pos_ms,
                    global_id=candidate.global_id,
                    class_label=candidate.class_label,
                    color=candidate.color,
                    type=candidate.type,
                    vlm_score=score,
                    vlm_explanation=explanation,
                    frame=candidate.frame,
                ))

            except Exception as e:
                self.logger.error(f"VLM inference failed for candidate: {e}")
                continue

        # Sort by score descending
        results.sort(key=lambda r: r.vlm_score, reverse=True)
        return results[:top_k]
