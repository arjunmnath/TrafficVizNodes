from pydantic import BaseModel, Field
from typing import Dict


class ServerConfig(BaseModel):
    zmq_bind: str = "tcp://*:5555"
    appearance_weight: float = 0.6
    temporal_weight: float = 0.2
    attribute_weight: float = 0.2
    match_threshold: float = 0.55
    temporal_window_seconds: float = 300.0
    api_port: int = 8000
    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    chroma_collection: str = "track_events"
    retrieval_model: str = "google/siglip2-base-patch16-224"
    video_sources: Dict[str, str] = Field(default_factory=dict)
    device: str = "auto"
