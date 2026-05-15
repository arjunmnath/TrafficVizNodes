import chromadb
from typing import List, Optional
from shared.utils import setup_logger


class VectorStore:
    """Read-only ChromaDB client for querying track events persisted by the ReID server."""

    def __init__(self, host: str, port: int, collection_name: str):
        self.logger = setup_logger("VectorStore")
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.logger.info(
            f"Connected to ChromaDB at {host}:{port}, "
            f"collection='{collection_name}' ({self.collection.count()} events)"
        )

    def query(
        self,
        query_text: str,
        top_k: int = 20,
        camera_id: Optional[str] = None,
    ) -> List[dict]:
        """Query ChromaDB for events similar to the given text.

        Returns a list of dicts with keys: id, document, metadata, distance.
        """
        where_filter = None
        if camera_id:
            where_filter = {"camera_id": camera_id}

        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        candidates = []
        if not results or not results["ids"] or not results["ids"][0]:
            return candidates

        for i, doc_id in enumerate(results["ids"][0]):
            candidates.append({
                "id": doc_id,
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        return candidates

    def get_event_count(self) -> int:
        return self.collection.count()
