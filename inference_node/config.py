from pydantic import BaseModel, Field, model_validator
from typing import Dict


class InferenceConfig(BaseModel):
    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    chroma_collection: str = "track_events"

    retrieval_model: str = "google/siglip2-base-patch16-224"
    reasoning_model: str = "microsoft/Florence-2-base-ft"
    vlm_model: str = Field(
        default="microsoft/Florence-2-base-ft",
        description="Deprecated alias for reasoning_model; kept for CLI compatibility.",
    )

    video_sources: Dict[str, str] = {}  # camera_id -> video file path
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    metadata_filter_enabled: bool = True

    top_k: int = Field(
        default=20,
        description="Deprecated alias for retrieval_top_k; kept for CLI compatibility.",
    )

    api_port: int = 8100
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"

    @model_validator(mode="after")
    def resolve_aliases(self) -> "InferenceConfig":
        if self.vlm_model != "microsoft/Florence-2-base-ft":
            self.reasoning_model = self.vlm_model
        if self.top_k != 20 and self.retrieval_top_k == 20:
            self.retrieval_top_k = self.top_k
        return self
