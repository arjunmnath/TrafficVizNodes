import numpy as np
from shared.utils import compute_cosine_similarity, compute_attribute_similarity
from reid_server.global_registry import GlobalRegistry
from shared.schemas import TrackEvent
from reid_server.config import ServerConfig

class Matcher:
    def __init__(self, config: ServerConfig, registry: GlobalRegistry):
        self.config = config
        self.registry = registry

    def _temporal_score(self, current_time: float, last_seen: float) -> float:
        diff = abs(current_time - last_seen)
        score = max(0.0, 1.0 - (diff / self.config.temporal_window_seconds))
        return score

    def match(self, event: TrackEvent) -> int:
        best_match_id = None
        best_score = 0.0
        
        event_emb = np.array(event.embedding)
        event_attrs = event.attributes.model_dump()
        
        candidates = self.registry.get_identities(event.class_label)
        
        for identity in candidates:
            app_sim = compute_cosine_similarity(event_emb, identity.embedding)
            attr_sim = compute_attribute_similarity(event_attrs, identity.attributes)
            temp_sim = self._temporal_score(event.timestamp, identity.last_seen)
            
            score = (self.config.appearance_weight * app_sim +
                     self.config.attribute_weight * attr_sim +
                     self.config.temporal_weight * temp_sim)
                     
            if score > best_score:
                best_score = score
                best_match_id = identity.global_id
                
        if best_match_id is not None and best_score >= self.config.match_threshold:
            self.registry.update_identity(best_match_id, event_emb, event_attrs, event.timestamp)
            return best_match_id
            
        return self.registry.add_identity(
            embedding=event_emb,
            cls_label=event.class_label,
            attributes=event_attrs,
            timestamp=event.timestamp
        )
