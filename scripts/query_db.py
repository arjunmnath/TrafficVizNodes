#!/usr/bin/env python3
"""Dry-run inference script: query the prepopulated ChromaDB and display top-K results.

Usage examples:
    uv run --python 3.11 python scripts/query_db.py "person in red jacket"
    uv run --python 3.11 python scripts/query_db.py "man with backpack near camera 1" --top_k 5
    uv run --python 3.11 python scripts/query_db.py "blue shirt" --top_k 10 --no-filters
"""

import argparse
import sys
from pathlib import Path

workspace_root = Path(__file__).resolve().parent.parent
sys.path.append(str(workspace_root))

from inference_node.retrieval.encoder import get_retrieval_encoder
from inference_node.retrieval.search import RetrievalEngine, RetrievalResult
from inference_node.retrieval.vector_store import VectorStore
from shared.utils import setup_logger

logger = setup_logger("QueryDB")

# ANSI colours for terminal output
_BOLD   = "\033[1m"
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a text query against the prepopulated ChromaDB and show top-K results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "query",
        type=str,
        help="Natural-language search query (e.g. 'person in red jacket near cam_1')",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of results to retrieve (default: 10)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="track_events",
        help="ChromaDB collection name (default: track_events)",
    )
    parser.add_argument(
        "--no-filters",
        dest="filters",
        action="store_false",
        default=True,
        help="Disable metadata filtering (camera/time constraints are ignored)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Force-filter results to a specific camera ID (e.g. cam_1)",
    )
    parser.add_argument(
        "--retrieval_model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Retrieval encoder model to use (default: google/siglip2-base-patch16-224)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run encoder on (default: auto)",
    )
    return parser.parse_args()


def _similarity_bar(distance: float, width: int = 20) -> str:
    """Render a visual similarity bar. ChromaDB cosine distance: 0=identical, 1=orthogonal."""
    similarity = max(0.0, 1.0 - distance)
    filled = round(similarity * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} {similarity * 100:.1f}%"


def _fmt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h:
        return f"{h:02d}:{m:02d}:{s:06.3f}"
    return f"{m:02d}:{s:06.3f}"


def print_results(query: str, parsed_query, results: list[RetrievalResult]) -> None:
    print()
    print(f"{_BOLD}{'─' * 70}{_RESET}")
    print(f"{_BOLD}  Query     :{_RESET} {_CYAN}{query}{_RESET}")
    print(f"{_BOLD}  Semantic  :{_RESET} {parsed_query.semantic_text}")
    if parsed_query.metadata_filters:
        print(f"{_BOLD}  Filters   :{_RESET} {parsed_query.metadata_filters}")
    print(f"{_BOLD}  Results   :{_RESET} {len(results)} returned")
    print(f"{_BOLD}{'─' * 70}{_RESET}")

    if not results:
        print(f"\n  {_YELLOW}No results found.{_RESET}\n")
        return

    for rank, r in enumerate(results, start=1):
        bar = _similarity_bar(r.distance)
        ts = _fmt_timestamp(r.camera_timestamp)
        bbox_str = f"  bbox={r.bbox}" if r.bbox else ""

        print(
            f"\n  {_BOLD}#{rank:<3}{_RESET}"
            f"  {_GREEN}ID:{_RESET} {r.id}"
        )
        print(
            f"       {_GREEN}Camera:{_RESET} {r.camera_id:<10}"
            f"  {_GREEN}Track:{_RESET} {r.track_id:<6}"
            f"  {_GREEN}Time:{_RESET} {ts}"
            f"{_DIM}{bbox_str}{_RESET}"
        )
        print(f"       {_GREEN}Similarity:{_RESET} {bar}")

    print(f"\n{_BOLD}{'─' * 70}{_RESET}\n")


def main():
    args = parse_args()

    logger.info(f"Initializing retrieval encoder '{args.retrieval_model}'...")
    encoder = get_retrieval_encoder(model_name=args.retrieval_model, device=args.device)

    logger.info(f"Connecting to ChromaDB collection '{args.collection}'...")
    vector_store = VectorStore(collection_name=args.collection)

    logger.info(
        f"Collection has {vector_store.get_event_count()} indexed embeddings."
    )

    engine = RetrievalEngine(
        encoder=encoder,
        vector_store=vector_store,
        metadata_filter_enabled=args.filters,
    )

    logger.info(f"Running query: '{args.query}' (top_k={args.top_k})")
    parsed, results = engine.search(
        query=args.query,
        top_k=args.top_k,
        camera_id=args.camera,
    )

    print_results(args.query, parsed, results)


if __name__ == "__main__":
    main()
