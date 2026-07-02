"""Florence-2 reasoning module — reranks retrieved candidate images."""

from __future__ import annotations

import torch
from typing import List

from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from inference_node.vqa.base import BaseVQAReasoner
from inference_node.vqa.types import CandidateImage, RankedResult
from shared.utils import setup_logger


class FlorenceReasoner(BaseVQAReasoner):
    """Scores retrieved candidates against a query using Florence-2."""

    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-base-ft",
        device: str = "auto",
    ) -> None:
        self.logger = setup_logger("FlorenceReasoner")
        self.logger.info(f"Loading Florence-2: {model_name}")

        self.device = self._resolve_device(device)
        dtype = self._resolve_dtype(self.device)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(self.device).eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.logger.info("Florence-2 loaded")

    def answer(
        self,
        query: str,
        candidates: List[CandidateImage],
        top_k: int = 5,
    ) -> List[RankedResult]:
        """Score each candidate image against the query and return top-K results."""
        self.logger.info(
            f"Florence reasoning on {len(candidates)} candidates for: '{query}'"
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
                self.logger.error(f"Florence inference failed: {exc}")
                continue

        results.sort(key=lambda item: item.vlm_score, reverse=True)
        return results[:top_k]

    def _score_candidate(self, query: str, frame: Image.Image) -> tuple[float, str]:
        # Resize to square to avoid AssertionError: only support square feature maps for now
        img = frame.convert("RGB").resize((768, 768))

        caption_prompt = "<CAPTION>"
        caption = self._run_task(caption_prompt, img).get(
            "<CAPTION>", "No description available."
        ).strip()

        vqa_prompt = f"<VQA>Does the image contain: {query}? Answer yes or no."
        answer = self._run_task(vqa_prompt, img, task_key="<VQA>").get(
            "<VQA>", "no"
        ).strip().lower()

        score = 10.0 if "yes" in answer else 0.0
        explanation = f"Matched: {answer.capitalize()}. Caption: {caption}"
        return score, explanation

    def _run_task(
        self,
        prompt: str,
        image: Image.Image,
        task_key: str | None = None,
    ) -> dict:
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.device
        )
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128,
                num_beams=3,
                use_cache=False,
            )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        return self.processor.post_process_generation(
            generated_text,
            task=task_key or prompt,
            image_size=image.size,
        )


def answer(
    reasoner: FlorenceReasoner,
    query: str,
    candidate_images: List[CandidateImage],
    top_k: int = 5,
) -> List[RankedResult]:
    """Public interface: score retrieved candidates with Florence-2."""
    return reasoner.answer(query=query, candidates=candidate_images, top_k=top_k)
