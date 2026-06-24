import numpy as np

class GlobalIdentity:
    def __init__(self, global_id: int, embedding: np.ndarray, cls_label: str, attributes: dict, timestamp: float):
        self.global_id = global_id
        self.embedding = embedding
        self.cls_label = cls_label
        self.attributes = attributes
        self.last_seen = timestamp

    def update(self, embedding: np.ndarray, attributes: dict, timestamp: float, alpha: float = 0.9):
        # Exponential moving average for embedding
        self.embedding = alpha * self.embedding + (1 - alpha) * embedding
        norm = np.linalg.norm(self.embedding)
        if norm > 0:
            self.embedding /= norm
            
        self.attributes = attributes
        self.last_seen = timestamp
    
    def __repr__(self):
        return f"GlobalIdentity(global_id={self.global_id}, cls_label={self.cls_label}, attributes={self.attributes}, last_seen={self.last_seen})"

class GlobalRegistry:
    def __init__(self):
        self.identities = {} # global_id -> GlobalIdentity
        self._next_id = 1
        
    def add_identity(self, embedding: np.ndarray, cls_label: str, attributes: dict, timestamp: float) -> int:
        global_id = self._next_id
        self._next_id += 1
        self.identities[global_id] = GlobalIdentity(global_id, embedding, cls_label, attributes, timestamp)
        return global_id

    def get_identities(self, cls_label: str = None) -> list:
        if cls_label:
            return [i for i in self.identities.values() if i.cls_label == cls_label]
        return list(self.identities.values())

    def update_identity(self, global_id: int, embedding: np.ndarray, attributes: dict, timestamp: float):
        if global_id in self.identities:
            self.identities[global_id].update(embedding, attributes, timestamp)
