import argparse
import json
import uvicorn

from inference_node.config import InferenceConfig
from inference_node.retrieval.encoder import get_retrieval_encoder
from inference_node.retrieval.search import RetrievalEngine
from inference_node.retrieval.vector_store import VectorStore
from inference_node.frame_extractor import FrameExtractor
from inference_node.vqa import get_vqa_reasoner
from inference_node.rag_pipeline import RAGPipeline
from inference_node.api import create_app
from shared.utils import setup_logger


def run_inference_node(config: InferenceConfig) -> None:
    logger = setup_logger("InferenceNode")
    logger.info("Starting Inference Node")
    logger.info(f"Retrieval model: {config.retrieval_model}")
    logger.info(f"Reasoning model: {config.reasoning_model}")
    logger.info(f"ChromaDB: {config.chroma_host}:{config.chroma_port}")
    logger.info(f"Video sources: {config.video_sources}")

    logger.info("Connecting to ChromaDB...")
    vector_store = VectorStore(
        host=config.chroma_host,
        port=config.chroma_port,
        collection_name=config.chroma_collection,
    )

    logger.info(f"Loading retrieval encoder '{config.retrieval_model}'...")
    encoder = get_retrieval_encoder(
        model_name=config.retrieval_model,
        device=config.device,
    )

    retrieval_engine = RetrievalEngine(
        encoder=encoder,
        vector_store=vector_store,
        metadata_filter_enabled=config.metadata_filter_enabled,
    )

    logger.info("Initializing frame extractor...")
    frame_extractor = FrameExtractor(video_sources=config.video_sources)

    logger.info(f"Loading VQA reasoner '{config.reasoning_model}' (this may take a few minutes)...")
    reasoner = get_vqa_reasoner(
        model_name=config.reasoning_model,
        device=config.device,
    )

    logger.info("Building RAG pipeline...")
    pipeline = RAGPipeline(
        retrieval_engine=retrieval_engine,
        frame_extractor=frame_extractor,
        reasoner=reasoner,
        retrieval_top_k=config.retrieval_top_k,
        rerank_top_k=config.rerank_top_k,
    )

    logger.info("Starting API server...")
    app = create_app(pipeline=pipeline, vector_store=vector_store)
    uvicorn.run(app, host="0.0.0.0", port=config.api_port, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCTV Inference Node")
    parser.add_argument("--chroma_host", type=str, default="chromadb")
    parser.add_argument("--chroma_port", type=int, default=8000)
    parser.add_argument("--chroma_collection", type=str, default="track_events")
    parser.add_argument(
        "--retrieval_model",
        type=str,
        default="google/siglip2-base-patch16-224",
    )
    parser.add_argument(
        "--reasoning_model",
        type=str,
        default="microsoft/Florence-2-large",
    )
    parser.add_argument(
        "--vlm_model",
        type=str,
        default="microsoft/Florence-2-large",
        help="Deprecated alias for --reasoning_model",
    )
    parser.add_argument("--api_port", type=int, default=8100)
    parser.add_argument("--retrieval_top_k", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=20, help="Deprecated alias for --retrieval_top_k")
    parser.add_argument("--rerank_top_k", type=int, default=5)
    parser.add_argument(
        "--metadata_filter_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--video_sources",
        type=str,
        default="{}",
        help='JSON dict mapping camera_id to video path, e.g. \'{"cam_1": "/app/dataset/video.avi"}\'',
    )
    args = parser.parse_args()

    reasoning_model = args.reasoning_model
    if args.vlm_model != "microsoft/Florence-2-large":
        reasoning_model = args.vlm_model

    retrieval_top_k = args.retrieval_top_k
    if args.top_k != 20 and args.retrieval_top_k == 20:
        retrieval_top_k = args.top_k

    config = InferenceConfig(
        chroma_host=args.chroma_host,
        chroma_port=args.chroma_port,
        chroma_collection=args.chroma_collection,
        retrieval_model=args.retrieval_model,
        reasoning_model=reasoning_model,
        vlm_model=args.vlm_model,
        video_sources=json.loads(args.video_sources),
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=args.rerank_top_k,
        metadata_filter_enabled=args.metadata_filter_enabled,
        top_k=args.top_k,
        api_port=args.api_port,
        device=args.device,
    )
    run_inference_node(config)
