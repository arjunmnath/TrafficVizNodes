try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

from typing import Optional

from shared.utils import setup_logger


class EventStore:
    """Persists matched track events to ChromaDB using SigLIP2 image embeddings."""

    def __init__(
        self,
        host: str = "chromadb",
        port: int = 8000,
        collection_name: str = "track_events",
        indexer: Optional[object] = None,
    ):
        if not HAS_CHROMADB:
            raise ImportError(
                "chromadb is not installed. Run `pip install chromadb` to enable persistence."
            )
        self.logger = setup_logger("EventStore")
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.indexer = indexer
        self.logger.info(f"Connected to ChromaDB at {host}:{port}")

    def store_event(self, event_data: dict):
        """Store a track event using a SigLIP2 embedding of the cropped object image.

        Args:
            event_data: Dict with camera_id, track_id, camera_timestamp,
                        video_pos_ms, bbox, and optional video_path.
        """
        if self.indexer is None:
            self.logger.warning("No indexer configured; skipping event persistence")
            return

        event_id = (
            f"{event_data['camera_id']}_"
            f"{event_data['track_id']}_"
            f"{event_data['camera_timestamp']:.4f}"
        )

        embedding = self.indexer.embed_event(event_data)
        if embedding is None:
            self.logger.warning(f"Skipping event without embedding: {event_id}")
            return

        metadata = {
            "camera_id": event_data.get("camera_id", ""),
            "track_id": int(event_data.get("track_id", 0)),
            "camera_timestamp": float(event_data.get("camera_timestamp", 0.0)),
        }

        if event_data.get("video_pos_ms") is not None:
            metadata["video_pos_ms"] = float(event_data["video_pos_ms"])
        if event_data.get("bbox"):
            metadata["bbox"] = ",".join(str(value) for value in event_data["bbox"])
        if event_data.get("video_path"):
            metadata["video_path"] = event_data["video_path"]

        try:
            self.collection.upsert(
                ids=[event_id],
                embeddings=[embedding.tolist()],
                metadatas=[metadata],
            )
        except Exception as exc:
            self.logger.error(f"Failed to store event: {exc}")
