#!/usr/bin/env python3
"""Single script to dry-run the inference node pipeline in memory.

It loads the cropped images from reid_crops_cleaned, encodes them with SigLIP2,
stores all embeddings in memory, and enters an interactive loop to wait for
user query inputs. For each query, it filters/searches in memory and reranks
candidates using Florence-2 VQA.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add workspace root to python path to import app modules
workspace_root = Path(__file__).resolve().parent.parent
sys.path.append(str(workspace_root))

from inference_node.retrieval.encoder import get_retrieval_encoder, BaseRetrievalEncoder
from inference_node.vqa import get_vqa_reasoner, BaseVQAReasoner, CandidateImage, RankedResult
from inference_node.retrieval.query_parser import parse_query
from shared.utils import setup_logger, compute_cosine_similarity

logger = setup_logger("DryRunInference")

# ANSI colors for beautiful terminal output
_BOLD    = "\033[1m"
_CYAN    = "\033[96m"
_GREEN   = "\033[92m"
_YELLOW  = "\033[93m"
_RED     = "\033[91m"
_MAGENTA = "\033[95m"
_DIM     = "\033[2m"
_RESET   = "\033[0m"

# Filename pattern expected inside each global-ID subdirectory:
#   clip1_f000001_t1_s0.04.jpg
FILENAME_REGEX = re.compile(
    r"^(clip\d+)_f(\d+)_t(\d+)_s([\d.]+)\.(jpg|jpeg|png)$", re.IGNORECASE
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dry-run script to test SigLIP2 retrieval and Florence-2 VQA in memory."
    )
    parser.add_argument(
        "--crops_dir",
        type=str,
        default=str(workspace_root / "reid_crops_cleaned"),
        help="Root directory of ReID crops.",
    )
    parser.add_argument(
        "--reid_json",
        type=str,
        default=str(workspace_root / "reid-dmt-backbone" / "cleaned_reid.json"),
        help="Path to cleaned_reid.json for bbox / class label lookup.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device to use for PyTorch models.",
    )
    parser.add_argument(
        "--retrieval_model",
        type=str,
        default="google/siglip2-base-patch16-224",
        help="Retrieval encoder model (e.g. google/siglip2-base-patch16-224, openclip-vit, openclip:ViT-B-32/laion2b_s34b_b79k)",
    )
    parser.add_argument(
        "--retrieval_top_k",
        type=int,
        default=10,
        help="Number of candidates to retrieve using SigLIP2 (passed to Florence-2).",
    )
    parser.add_argument(
        "--rerank_top_k",
        type=int,
        default=5,
        help="Number of final candidates to return after Florence-2 reranking.",
    )
    parser.add_argument(
        "--reasoning_model",
        type=str,
        default="microsoft/Florence-2-base-ft",
        help="VQA reasoning model to use (e.g. microsoft/Florence-2-base-ft or Qwen/Qwen2-VL-2B-Instruct).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single query and exit. If not provided, enters interactive mode.",
    )
    return parser.parse_args()


def camera_id_from_clip(clip_name: str) -> str:
    """Derive a camera identifier from the clip stem (e.g. 'clip1' -> 'cam_1')."""
    if clip_name.startswith("clip"):
        num = clip_name[4:]
        if num.isdigit():
            return f"cam_{num}"
    return clip_name


def collect_crop_files(crops_dir: Path) -> list:
    """Recursively collect all valid ReID crop images under crops_dir."""
    crop_files = []
    for p in crops_dir.glob("**/*"):
        if not (p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
            continue
        match = FILENAME_REGEX.match(p.name)
        if not match:
            continue
        try:
            global_id = int(p.parent.name)
        except ValueError:
            continue
        crop_files.append((p, global_id, match))
    return crop_files


def load_reid_metadata(reid_json_path: str) -> dict:
    """Load metadata lookup dictionary from cleaned_reid.json."""
    if not os.path.exists(reid_json_path):
        logger.warning(
            f"ReID JSON not found at {reid_json_path}. "
            "Proceeding without bbox/class_label enrichment."
        )
        return {}

    logger.info(f"Loading ReID metadata from {reid_json_path}...")
    with open(reid_json_path, "r") as f:
        data = json.load(f)

    lookup = {}
    for global_id_str, entries in data.items():
        try:
            global_id = int(global_id_str)
        except ValueError:
            continue
        for entry in entries:
            video = entry.get("video", "")
            frame = entry.get("frame")
            if video and frame is not None:
                lookup[(global_id, video, int(frame))] = entry

    logger.info(f"Loaded {len(lookup)} ReID metadata entries.")
    return lookup


def filter_candidates(candidates: list[dict], filters: dict) -> list[dict]:
    """Filter candidates using metadata filters parsed from the query."""
    filtered = []
    for c in candidates:
        keep = True
        meta = c["metadata"]

        if "camera_id" in filters:
            # We support flexible matching (e.g. 'cam_1' vs 'cam1' vs '1')
            q_cam = str(filters["camera_id"]).lower().replace("_", "").replace("cam", "")
            c_cam = str(meta["camera_id"]).lower().replace("_", "").replace("cam", "")
            if q_cam != c_cam:
                keep = False

        if "camera_timestamp_gte" in filters:
            if meta["camera_timestamp"] < float(filters["camera_timestamp_gte"]):
                keep = False

        if "camera_timestamp_lt" in filters:
            if meta["camera_timestamp"] >= float(filters["camera_timestamp_lt"]):
                keep = False

        if keep:
            filtered.append(c)

    return filtered


def run_search_pipeline(
    query_str: str,
    candidates_db: list[dict],
    encoder: BaseRetrievalEncoder,
    reasoner: BaseVQAReasoner,
    retrieval_k: int,
    rerank_k: int,
):
    """Executes search and VQA reranking for a single query."""
    # 1. Parse query for filters
    parsed_query = parse_query(query_str)
    semantic_text = parsed_query.semantic_text or query_str

    print(f"\n{_BOLD}{'─' * 70}{_RESET}")
    print(f"{_BOLD}  Query         :{_RESET} {_CYAN}{query_str}{_RESET}")
    print(f"{_BOLD}  Semantic Query:{_RESET} {semantic_text}")
    if parsed_query.metadata_filters:
        print(f"{_BOLD}  Filters       :{_RESET} {parsed_query.metadata_filters}")
    print(f"{_BOLD}{'─' * 70}{_RESET}\n")

    # 2. Encode text query
    try:
        query_embedding = encoder.encode_text(semantic_text)
    except Exception as e:
        logger.error(f"Failed to encode query text: {e}")
        return

    # 3. Compute cosine similarity & filter candidates
    scored_candidates = []
    for c in candidates_db:
        sim = compute_cosine_similarity(query_embedding, c["embedding"])
        scored_candidates.append({**c, "similarity": sim})

    # temp remove
    sorted_candidates = sorted(scored_candidates, key=lambda x: x["similarity"], reverse=True)
    for idx, c in enumerate(sorted_candidates[:5], start=1):
        print(f"    {idx}. {c['filepath'].name} (Score: {c['similarity']:.4f})")
    sys.exit(0)

    # Apply parsed metadata filters
    if parsed_query.metadata_filters:
        filtered_scored = filter_candidates(scored_candidates, parsed_query.metadata_filters)
        print(f"  Applied filters: {len(scored_candidates)} -> {len(filtered_scored)} candidates remaining.")
        scored_candidates = filtered_scored

    # Sort and take top retrieval_k
    scored_candidates.sort(key=lambda x: x["similarity"], reverse=True)
    top_retrieved = scored_candidates[:retrieval_k]

    if not top_retrieved:
        print(f"  {_YELLOW}No candidates matched the query or filters.{_RESET}")
        return

    print(f"{_BOLD}  Top {len(top_retrieved)} SigLIP2 Retrieval Results (Similarity):{_RESET}")
    for idx, c in enumerate(top_retrieved, start=1):
        print(
            f"    #{idx:<2} Similarity: {c['similarity'] * 100:.1f}% | "
            f"Cam: {c['metadata']['camera_id']} | "
            f"Track ID: {c['metadata']['track_id']} | "
            f"Time: {c['metadata']['camera_timestamp']:.2f}s | "
            f"File: {c['filepath'].name}"
        )

    # 4. Prep and run VQA reasoning
    print(f"\n  Running {reasoner.__class__.__name__} VQA reasoning on top retrieved candidates...")
    reasoning_candidates = []
    for c in top_retrieved:
        try:
            # Load the frame
            frame = Image.open(c["filepath"]).convert("RGB")
            bbox = c["metadata"].get("bbox")
            bbox_list = [float(v) for v in bbox.split(",")] if bbox else None

            reasoning_candidates.append(
                CandidateImage(
                    camera_id=c["metadata"]["camera_id"],
                    camera_timestamp=c["metadata"]["camera_timestamp"],
                    video_pos_ms=c["metadata"]["video_pos_ms"],
                    track_id=c["metadata"]["track_id"],
                    bbox=bbox_list,
                    frame=frame,
                    retrieval_distance=1.0 - c["similarity"],
                )
            )
        except Exception as e:
            logger.error(f"Failed to load image candidate {c['filepath']}: {e}")

    if not reasoning_candidates:
        print(f"  {_RED}Failed to prepare any candidate images for Florence-2.{_RESET}")
        return

    # Run VQA Reranking
    ranked_results = reasoner.answer(
        query=semantic_text,
        candidates=reasoning_candidates,
        top_k=rerank_k,
    )

    # Print final Ranked Results
    print(f"\n{_BOLD}{'─' * 70}{_RESET}")
    print(f"{_BOLD}  Final Ranked Results ({reasoner.__class__.__name__}):{_RESET}")
    print(f"{_BOLD}{'─' * 70}{_RESET}")
    if not ranked_results:
        print(f"  {_YELLOW}No candidate verified by {reasoner.__class__.__name__}.{_RESET}")
    else:
        for idx, res in enumerate(ranked_results, start=1):
            color = _GREEN if res.vlm_score > 0 else _YELLOW
            status = "YES" if res.vlm_score > 0 else "NO"
            print(
                f"\n  {_BOLD}#{idx:<2}{_RESET} [{color}{status}{_RESET}] (Score: {res.vlm_score}) "
                f"Cam: {res.camera_id} | Track: {res.track_id} | Time: {res.camera_timestamp:.2f}s"
            )
            print(f"       {_DIM}Explanation: {res.vlm_explanation}{_RESET}")
    print(f"\n{_BOLD}{'─' * 70}{_RESET}\n")


def patch_huggingface_florence2():
    """Patches local Hugging Face cache files for Florence-2 to fix compatibility
    issues with newer versions of the transformers library.
    """
    import glob
    import os
    import re
    from transformers import AutoConfig

    # Try loading the configuration to ensure the model files are downloaded first
    logger.info("Checking Florence-2 cache and applying compatibility patches if needed...")
    try:
        AutoConfig.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)
    except Exception:
        # Ignore errors during initial load, as we will patch the downloaded files
        pass

    cache_dir = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
    if not os.path.exists(cache_dir):
        return

    config_files = glob.glob(os.path.join(cache_dir, "**", "configuration_florence2.py"), recursive=True)
    model_files = glob.glob(os.path.join(cache_dir, "**", "modeling_florence2.py"), recursive=True)
    processing_files = glob.glob(os.path.join(cache_dir, "**", "processing_florence2.py"), recursive=True)

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
                logger.info(f"Patched compatibility bug in: {cf}")
        except Exception as e:
            logger.warning(f"Failed to patch {cf}: {e}")

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
                logger.info(f"Patched compatibility bug in: {pf}")
        except Exception as e:
            logger.warning(f"Failed to patch {pf}: {e}")

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
                flags=re.DOTALL
            )
            if count1 > 0:
                content = sub1
                patched = True
                
            # Patch _supports_sdpa
            sub2, count2 = re.subn(
                r"def _supports_sdpa\(self\):\s+.*?return self\.language_model\._supports_sdpa",
                "def _supports_sdpa(self):\n        if not hasattr(self, 'language_model'):\n            return False\n        return self.language_model._supports_sdpa",
                content,
                flags=re.DOTALL
            )
            if count2 > 0:
                content = sub2
                patched = True

            if patched:
                with open(mf, "w") as f:
                    f.write(content)
                logger.info(f"Patched compatibility bug in: {mf}")
        except Exception as e:
            logger.warning(f"Failed to patch {mf}: {e}")


def main():
    args = parse_args()

    # Self-heal Hugging Face cache files for Florence-2 before initialization
    patch_huggingface_florence2()

    # Load ReID metadata lookup
    metadata_lookup = load_reid_metadata(args.reid_json)

    # 1. Initialize retrieval encoder
    logger.info(f"Initializing retrieval encoder '{args.retrieval_model}'...")
    encoder = get_retrieval_encoder(model_name=args.retrieval_model, device=args.device)

    # 2. Collect crop files
    crops_dir = Path(args.crops_dir)
    if not crops_dir.exists():
        logger.error(f"Crops directory does not exist: {crops_dir}")
        sys.exit(1)

    logger.info(f"Scanning crops directory: {crops_dir}")
    crop_files = collect_crop_files(crops_dir)
    logger.info(f"Found {len(crop_files)} crop images under {crops_dir}.")

    if not crop_files:
        logger.error("No valid crops found. Exiting.")
        sys.exit(1)

    # 3. Load & Encode all crops in-memory
    logger.info("Encoding all crop images to memory embeddings...")
    candidates_db = []
    for filepath, global_id, match in tqdm(crop_files, desc="Encoding crops"):
        clip_name = match.group(1)
        frame_idx = int(match.group(2))
        timestamp_seconds = float(match.group(4))
        camera_id = camera_id_from_clip(clip_name)

        # Build metadata
        metadata = {
            "camera_id": camera_id,
            "track_id": global_id,
            "camera_timestamp": timestamp_seconds,
            "video_pos_ms": timestamp_seconds * 1000.0,
        }

        # Enrich with cleaned_reid.json
        video_filename = f"{clip_name}.mp4"
        entry = metadata_lookup.get((global_id, video_filename, frame_idx), {})
        if entry.get("bbox"):
            metadata["bbox"] = ",".join(str(v) for v in entry["bbox"])
        if entry.get("class_label"):
            metadata["class_label"] = entry["class_label"]

        # Encode image
        try:
            with Image.open(filepath) as img:
                embedding = encoder.encode_image(img)
        except Exception as e:
            logger.error(f"Failed to encode {filepath}: {e}")
            continue

        candidates_db.append({
            "filepath": filepath,
            "embedding": embedding,
            "metadata": metadata
        })

    logger.info(f"Successfully encoded {len(candidates_db)} crops into memory.")

    # 4. Initialize VQA Reasoner
    logger.info(f"Initializing VQA Reasoner: {args.reasoning_model}...")
    reasoner = get_vqa_reasoner(model_name=args.reasoning_model, device=args.device)

    # 5. Handle Query input
    if args.query:
        # Run single query mode
        run_search_pipeline(
            query_str=args.query,
            candidates_db=candidates_db,
            encoder=encoder,
            reasoner=reasoner,
            retrieval_k=args.retrieval_top_k,
            rerank_k=args.rerank_top_k,
        )
    else:
        # Interactive mode
        print(f"\n{_BOLD}{_GREEN}=== In-Memory Inference Pipeline Ready ==={_RESET}")
        print("Type your search query and press Enter. Type 'exit' or 'quit' to close.")
        while True:
            try:
                query_str = input(f"\n{_BOLD}Query > {_RESET}").strip()
                if not query_str:
                    continue
                if query_str.lower() in ("exit", "quit"):
                    print("Exiting dry-run inference pipeline. Goodbye!")
                    break
                run_search_pipeline(
                    query_str=query_str,
                    candidates_db=candidates_db,
                    encoder=encoder,
                    reasoner=reasoner,
                    retrieval_k=args.retrieval_top_k,
                    rerank_k=args.rerank_top_k,
                )
            except KeyboardInterrupt:
                print("\nExiting dry-run inference pipeline. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error during query execution: {e}")


if __name__ == "__main__":
    main()
