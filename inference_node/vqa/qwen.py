"""Qwen2-VL reasoning module — reranks retrieved candidate images using Qwen2-VL."""

from __future__ import annotations

import torch
from typing import List
from PIL import Image

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from inference_node.vqa.base import BaseVQAReasoner
from inference_node.vqa.types import CandidateImage, RankedResult
from shared.utils import setup_logger


class QwenVLMReasoner(BaseVQAReasoner):
    """Scores retrieved candidates against a query using Qwen2-VL."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "auto",
    ) -> None:
        self.logger = setup_logger("QwenVLMReasoner")
        self.logger.info(f"Loading Qwen2-VL: {model_name}")

        self.device = self._resolve_device(device)
        dtype = self._resolve_dtype(self.device)

        if self.device == "cuda":
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
            )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
            ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.logger.info("Qwen2-VL loaded")

    def answer(
        self,
        query: str,
        candidates: List[CandidateImage],
        top_k: int = 5,
    ) -> List[RankedResult]:
        """Score each candidate image against the query and return top-K results."""
        self.logger.info(
            f"Qwen2-VL reasoning on {len(candidates)} candidates for: '{query}'"
        )
        results: List[RankedResult] = []

        for candidate in candidates:
            try:
                score, explanation = self._score_candidate(query, candidate.frame)
                results.append(
                    RankedResult(
                        camera_id=candidate.camera_id,
                        camera_timestamp=candidate.camera_timestamp,
                        video_pos_ms=candidate.video_pos_ms,
                        track_id=candidate.track_id,
                        vlm_score=score,
                        vlm_explanation=explanation,
                        frame=candidate.frame,
                    )
                )
            except Exception as exc:
                self.logger.error(f"Qwen2-VL inference failed: {exc}")
                continue

        results.sort(key=lambda item: item.vlm_score, reverse=True)
        return results[:top_k]

    def _score_candidate(self, query: str, frame: Image.Image) -> tuple[float, str]:
        img = frame.convert("RGB")

        # Query structure to answer yes/no and describe the image
        prompt = (
            f"Analyze this image to check if it matches the search query: '{query}'.\n"
            "Format your response exactly as follows:\n"
            "Answer: <Yes or No>\n"
            "Explanation: <brief description explaining why or why not>"
        )

        response = self._run_task(prompt, img).strip()

        # Parse structured output
        score = 0.0
        explanation = response

        answer_line = None
        explanation_line = None
        for line in response.split("\n"):
            line = line.strip()
            if line.lower().startswith("answer:"):
                answer_line = line[len("answer:"):].strip().lower()
            elif line.lower().startswith("explanation:"):
                explanation_line = line[len("explanation:"):].strip()

        if answer_line:
            score = 10.0 if "yes" in answer_line else 0.0
            if explanation_line:
                explanation = f"Matched: {answer_line.capitalize()}. Caption: {explanation_line}"
            else:
                explanation = f"Matched: {answer_line.capitalize()}. Explanation: {response}"
        else:
            # Fallback
            is_yes = any(word in response[:25].lower() for word in ["yes", "yeah", "yep", "contains"])
            score = 10.0 if is_yes else 0.0
            explanation = response

        return score, explanation

    def _run_task(self, prompt: str, image: Image.Image) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
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
        )
        inputs = inputs.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        return output_text
