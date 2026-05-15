import argparse
import json
import uvicorn

from inference_node.config import InferenceConfig
from inference_node.vector_store import VectorStore
from inference_node.frame_extractor import FrameExtractor
from inference_node.vlm_reranker import VLMReranker
from inference_node.rag_pipeline import RAGPipeline
from inference_node.api import create_app
from shared.utils import setup_logger


def run_inference_node(config: InferenceConfig):
    logger = setup_logger("InferenceNode")
    logger.info("Starting Inference Node")
    logger.info(f"VLM model: {config.vlm_model}")
    logger.info(f"ChromaDB: {config.chroma_host}:{config.chroma_port}")
    logger.info(f"Video sources: {config.video_sources}")

    # Initialize components
    logger.info("Connecting to ChromaDB...")
    vector_store = VectorStore(
        host=config.chroma_host,
        port=config.chroma_port,
        collection_name=config.chroma_collection,
    )

    logger.info("Initializing frame extractor...")
    frame_extractor = FrameExtractor(video_sources=config.video_sources)

    logger.info("Loading VLM (this may take a few minutes)...")
    vlm_reranker = VLMReranker(
        model_name=config.vlm_model,
        device=config.device,
    )

    logger.info("Building RAG pipeline...")
    pipeline = RAGPipeline(
        vector_store=vector_store,
        frame_extractor=frame_extractor,
        vlm_reranker=vlm_reranker,
        top_k=config.top_k,
        rerank_top_k=config.rerank_top_k,
    )

    logger.info("Starting API server...")
    app = create_app(pipeline=pipeline, vector_store=vector_store)
    uvicorn.run(app, host="0.0.0.0", port=config.api_port, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCTV VLM Inference Node")
    parser.add_argument("--chroma_host", type=str, default="chromadb")
    parser.add_argument("--chroma_port", type=int, default=8000)
    parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--api_port", type=int, default=8100)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--rerank_top_k", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--video_sources",
        type=str,
        default="{}",
        help='JSON dict mapping camera_id to video path, e.g. \'{"cam_1": "/app/dataset/video.avi"}\'',
    )
    args = parser.parse_args()

    video_sources = json.loads(args.video_sources)

    config = InferenceConfig(
        chroma_host=args.chroma_host,
        chroma_port=args.chroma_port,
        vlm_model=args.vlm_model,
        video_sources=video_sources,
        top_k=args.top_k,
        rerank_top_k=args.rerank_top_k,
        api_port=args.api_port,
        device=args.device,
    )
    run_inference_node(config)
