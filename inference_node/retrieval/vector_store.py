"""ChromaDB wrapper for SigLIP2 embedding search with metadata filtering."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import chromadb
import numpy as np

from shared.utils import setup_logger


class VectorStore:
    """Read-only ChromaDB client for SigLIP2 embedding retrieval."""

    def __init__(self, collection_name: str) -> None:
        self.logger = setup_logger("VectorStore")
        self.client = chromadb.CloudClient(
          api_key='ck-5oQZAaS8PWuQXdP2rMZHW4nroQHAhqR3PWnzky7bEQhX',
          tenant='a1cf0a6e-4cc2-453c-aa3b-3c2acb6a2dc5',
          database='testing'
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.logger.info(
            f"Connected to ChromaDB: "
            f"collection='{collection_name}' ({self.collection.count()} events)"
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[dict]:
        """Search by embedding vector with optional metadata filters.

        Returns dicts with keys: id, metadata, distance.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances"],
        )

        candidates: List[dict] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return candidates

        for i, doc_id in enumerate(results["ids"][0]):
            candidates.append(
                {
                    "id": doc_id,
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )

        return candidates

    def get_event_count(self) -> int:
        return self.collection.count()
