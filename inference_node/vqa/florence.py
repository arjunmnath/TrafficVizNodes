"""Florence-2 reasoning module — reranks retrieved candidate images."""

from __future__ import annotations

import logging
import warnings

# Suppress transformers warnings and load reports
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from typing import List

from PIL import Image
import transformers
from transformers import AutoModelForCausalLM, AutoProcessor

transformers.utils.logging.set_verbosity_error()

from inference_node.vqa.base import BaseVQAReasoner
from inference_node.vqa.types import CandidateImage, RankedResult
from shared.utils import setup_logger


def _patch_huggingface_florence2(model_name: str):
    """Patches local Hugging Face cache files for Florence-2 to fix compatibility
    issues with newer versions of the transformers library (such as cache indexing,
    forced_bos_token_id, flash_attn, and sdpa).
    """
    import glob
    import os
    import re
    from transformers import AutoConfig, AutoProcessor

    # Try loading the configuration to ensure the model files are downloaded first
    try:
        AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        pass

    # Try loading the processor to ensure processor files are downloaded first
    try:
        AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        pass

    cache_dir = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
    if not os.path.exists(cache_dir):
        return

    config_files = glob.glob(
        os.path.join(cache_dir, "**", "configuration_florence2.py"), recursive=True
    )
    model_files = glob.glob(os.path.join(cache_dir, "**", "modeling_florence2.py"), recursive=True)
    processing_files = glob.glob(
        os.path.join(cache_dir, "**", "processing_florence2.py"), recursive=True
    )

    for cf in config_files:
        try:
            with open(cf, "r") as f:
                content = f.read()
            target = "if self.forced_bos_token_id is None"
            replacement = 'if getattr(self, "forced_bos_token_id", None) is None'
            if target in content:
                content = content.replace(target, replacement)
                with open(cf, "w") as f:
                    f.write(content)
        except Exception:
            pass

    for pf in processing_files:
        try:
            with open(pf, "r") as f:
                content = f.read()
            target = "tokenizer.additional_special_tokens"
            replacement = 'getattr(tokenizer, "additional_special_tokens", [])'
            if target in content:
                content = content.replace(target, replacement)
                with open(pf, "w") as f:
                    f.write(content)
        except Exception:
            pass

    for mf in model_files:
        try:
            with open(mf, "r") as f:
                content = f.read()

            patched = False

            # Patch _supports_flash_attn_2
            sub1, count1 = re.subn(
                r"def _supports_flash_attn_2\(self\):\s+.*?return self\.language_model\._supports_flash_attn_2",
                "def _supports_flash_attn_2(self):\n        if not hasattr(self, 'language_model'):\n            return False\n        return self.language_model._supports_flash_attn_2",
                content,
                flags=re.DOTALL,
            )
            if count1 > 0:
                content = sub1
                patched = True

            # Patch _supports_sdpa
            sub2, count2 = re.subn(
                r"def _supports_sdpa\(self\):\s+.*?return self\.language_model\._supports_sdpa",
                "def _supports_sdpa(self):\n        if not hasattr(self, 'language_model'):\n            return False\n        return self.language_model._supports_sdpa",
                content,
                flags=re.DOTALL,
            )
            if count2 > 0:
                content = sub2
                patched = True

            # Patch past_key_values[0][0].shape[2] cache subscript issue (compatibility with EncoderDecoderCache in transformers >= 4.50)
            target_cache = "past_key_values[0][0].shape[2]"
            replacement_cache = '(past_key_values.get_seq_length() if hasattr(past_key_values, "get_seq_length") else past_key_values[0][0].shape[2])'
            if target_cache in content:
                content = content.replace(target_cache, replacement_cache)
                patched = True

            if patched:
                with open(mf, "w") as f:
                    f.write(content)
        except Exception:
            pass


class FlorenceReasoner(BaseVQAReasoner):
    """Scores retrieved candidates against a query using Florence-2."""

    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-large",
        device: str = "auto",
    ) -> None:
        self.logger = setup_logger("FlorenceReasoner")

        # Self-heal Hugging Face cache files for Florence-2 compatibility with transformers v4.50+
        _patch_huggingface_florence2(model_name)

        self.logger.info(f"Loading Florence-2: {model_name}")

        self.device = self._resolve_device(device)
        dtype = self._resolve_dtype(self.device)

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                trust_remote_code=True,
                attn_implementation="eager",
            )
            .to(self.device)
            .eval()
        )

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
        self.logger.info(f"Florence reasoning on {len(candidates)} candidates for: '{query}'")
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
        caption = (
            self._run_task(caption_prompt, img)
            .get("<CAPTION>", "No description available.")
            .strip()
        )

        vqa_prompt = f"<VQA>Does the image contain: {query}? Answer yes or no."
        answer = (
            self._run_task(vqa_prompt, img, task_key="<VQA>").get("<VQA>", "no").strip().lower()
        )

        score = 10.0 if "yes" in answer else 0.0
        explanation = f"Matched: {answer.capitalize()}. Caption: {caption}"
        return score, explanation

    def _run_task(
        self,
        prompt: str,
        image: Image.Image,
        task_key: str | None = None,
    ) -> dict:
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
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

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
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
