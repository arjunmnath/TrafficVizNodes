import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class SimpleRegistry:
    """Identity registry that associates local track IDs to global IDs using appearance features.

    Tracks are registered only upon termination (when the tracker's on_track_terminated
    hook fires). The registry uses cosine similarity on appearance embeddings to match
    a terminated track to an existing global identity or to create a new one.
    """

    def __init__(self, match_threshold: float = 0.6):
        """Constructor.

        Args:
            match_threshold (float): Cosine similarity threshold for matching.
        """
        self.identities: Dict[int, dict] = {}  # global_id -> {"embedding": np.ndarray, "tracks": []}
        self.next_id: int = 1
        self.match_threshold: float = match_threshold

        # Maps local track_id -> global_id for quick lookup during a pipeline run
        self.track_to_global: Dict[int, int] = {}

    def match(self, embedding: np.ndarray) -> Tuple[Optional[int], float]:
        """Find the best matching global identity for the given embedding.

        Args:
            embedding (np.ndarray): Appearance feature vector of the terminated track.

        Returns:
            Tuple[Optional[int], float]: (global_id, similarity) or (None, -1.0) if no match.
        """
        best_id = None
        best_sim = -1.0
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)

        for global_id, data in self.identities.items():
            db_emb = data["embedding"]
            db_norm = db_emb / (np.linalg.norm(db_emb) + 1e-8)
            sim = float(np.dot(emb_norm, db_norm))

            if sim > best_sim:
                best_sim = sim
                best_id = global_id

        return best_id, best_sim

    def register_track(
        self,
        local_track_id: int,
        embedding: np.ndarray,
        class_label: str = "unknown",
        feed_name: str = "",
    ) -> Tuple[int, float]:
        """Register a terminated track into the global registry.

        Matches the track embedding against all existing global identities. If similarity
        exceeds the threshold, the track is associated with the existing global ID and its
        prototype embedding is updated. Otherwise a new global identity is created.

        Args:
            local_track_id (int): The local tracker-assigned track ID.
            embedding (np.ndarray): The appearance feature vector for this track.
            class_label (str): YOLO class label string.
            feed_name (str): Source feed name for the track.

        Returns:
            Tuple[int, float]: (global_id, similarity)
        """
        best_id, best_sim = self.match(embedding)

        track_record = {
            "local_track_id": int(local_track_id),
            "class_label": class_label,
            "feed_name": feed_name,
            "similarity": round(best_sim, 4) if best_id is not None else 1.0,
        }

        if best_id is not None and best_sim >= self.match_threshold:
            self.identities[best_id]["tracks"].append(track_record)
            self.identities[best_id]["embedding"] = self._update_prototype(
                self.identities[best_id]["embedding"], embedding
            )
            self.track_to_global[local_track_id] = best_id
            return best_id, best_sim
        else:
            new_id = self.next_id
            self.next_id += 1
            self.identities[new_id] = {
                "embedding": embedding,
                "tracks": [track_record],
            }
            self.track_to_global[local_track_id] = new_id
            return new_id, best_sim

    def get_global_id(self, local_track_id: int) -> Optional[int]:
        """Look up the global ID for a local track ID.

        Args:
            local_track_id (int): The local tracker-assigned track ID.

        Returns:
            Optional[int]: The global ID, or None if not registered.
        """
        return self.track_to_global.get(local_track_id)

    def get_results_summary(self) -> list:
        """Return a serialisable summary of all global identities and their tracks."""
        summary = []
        for global_id, data in self.identities.items():
            summary.append({
                "global_id": global_id,
                "tracks": data["tracks"],
            })
        return summary

    def get_embeddings_dict(self) -> dict:
        """Return a dict of global_id -> embedding suitable for np.savez.

        Returns:
            dict: Mapping of string global_id to numpy embedding arrays.
        """
        result = {}
        for global_id, data in self.identities.items():
            result[str(global_id)] = np.array(data["embedding"], dtype=np.float32)
        return result

    def _update_prototype(
        self,
        prototype: np.ndarray,
        embedding: np.ndarray,
        alpha: float = 0.1,
        similarity_threshold: float = 0.8,
    ) -> np.ndarray:
        """Update the prototype embedding with exponential moving average."""
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        proto_norm = prototype / (np.linalg.norm(prototype) + 1e-8)
        similarity = np.dot(proto_norm, emb_norm)

        if similarity < similarity_threshold:
            return prototype

        updated = (1 - alpha) * prototype + alpha * embedding
        updated /= np.linalg.norm(updated)
        return updated
