from pydantic import BaseModel
from typing import Dict, Optional


class InferenceConfig(BaseModel):
    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    chroma_collection: str = "track_events"
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    video_sources: Dict[str, str] = {}  # camera_id -> video file path
    top_k: int = 20  # Candidates from ChromaDB
    rerank_top_k: int = 5  # Final results after VLM re-ranking
    api_port: int = 8100
    device: str = "auto"  # "auto", "cuda", "cpu"
