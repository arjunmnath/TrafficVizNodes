import chromadb
from shared.utils import setup_logger


class EventStore:
    """Persists matched track events to ChromaDB for RAG-based retrieval."""

    def __init__(self, host: str = "chromadb", port: int = 8000):
        self.logger = setup_logger("EventStore")
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection = self.client.get_or_create_collection(
            name="track_events",
            metadata={"hnsw:space": "cosine"},
        )
        self.logger.info(f"Connected to ChromaDB at {host}:{port}")

    def _build_description(self, event_data: dict) -> str:
        """Build a natural language description from event metadata."""
        cls = event_data.get("class_label", "object")
        color = event_data.get("color", "unknown")
        vtype = event_data.get("type")
        camera = event_data.get("camera_id", "unknown")

        desc = f"{cls} with {color} color"
        if vtype:
            desc += f" {vtype}"
        desc += f" seen at camera {camera}"
        return desc

    def store_event(self, global_id: int, event_data: dict):
        """Store a matched event in ChromaDB.

        Args:
            global_id: The globally resolved identity ID.
            event_data: Dict with camera_id, track_id, class_label,
                        color, type, timestamp, video_pos_ms, bbox.
        """
        event_id = (
            f"{event_data['camera_id']}_"
            f"{event_data['track_id']}_"
            f"{event_data['timestamp']:.4f}"
        )

        description = self._build_description(event_data)

        metadata = {
            "camera_id": event_data.get("camera_id", ""),
            "global_id": global_id,
            "class_label": event_data.get("class_label", ""),
            "color": event_data.get("color", "unknown"),
            "timestamp": event_data.get("timestamp", 0.0),
        }

        # ChromaDB metadata only supports str/int/float/bool
        if event_data.get("type") is not None:
            metadata["type"] = event_data["type"]
        if event_data.get("video_pos_ms") is not None:
            metadata["video_pos_ms"] = event_data["video_pos_ms"]
        if event_data.get("bbox"):
            # Store bbox as comma-separated string
            metadata["bbox"] = ",".join(str(v) for v in event_data["bbox"])

        try:
            self.collection.upsert(
                ids=[event_id],
                documents=[description],
                metadatas=[metadata],
            )
        except Exception as e:
            self.logger.error(f"Failed to store event: {e}")
