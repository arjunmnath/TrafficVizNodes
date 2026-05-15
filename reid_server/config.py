from pydantic import BaseModel

class ServerConfig(BaseModel):
    zmq_bind: str = "tcp://*:5555"
    appearance_weight: float = 0.6
    attribute_weight: float = 0.2
    temporal_weight: float = 0.2
    match_threshold: float = 0.70
    temporal_window_seconds: float = 300.0  # Time to keep identities active
    api_port: int = 8000
    chroma_host: str = "chromadb"
    chroma_port: int = 8000
