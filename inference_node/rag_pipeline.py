import base64
import io
from datetime import datetime, timezone
from typing import Optional, List
from PIL import Image

from inference_node.vector_store import VectorStore
from inference_node.frame_extractor import FrameExtractor
from inference_node.vlm_reranker import VLMReranker, CandidateFrame
from shared.schemas import QueryResultItem
from shared.utils import setup_logger


class RAGPipeline:
    """Orchestrates: ChromaDB retrieval → frame extraction → VLM re-ranking."""

    def __init__(
        self,
        vector_store: VectorStore,
        frame_extractor: FrameExtractor,
        vlm_reranker: VLMReranker,
        top_k: int = 20,
        rerank_top_k: int = 5,
    ):
        self.logger = setup_logger("RAGPipeline")
        self.vector_store = vector_store
        self.frame_extractor = frame_extractor
        self.vlm = vlm_reranker
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k

    def _image_to_base64(self, img: Image.Image, max_size: int = 320) -> str:
        """Resize and encode image to base64 JPEG for API response."""
        img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _parse_bbox(self, bbox_str: Optional[str]) -> Optional[list]:
        """Parse bbox from comma-separated metadata string."""
        if not bbox_str:
            return None
        try:
            return [float(v) for v in bbox_str.split(",")]
        except (ValueError, AttributeError):
            return None

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        camera_id: Optional[str] = None,
    ) -> List[QueryResultItem]:
        """Execute the full RAG pipeline.

        1. Query ChromaDB for similar events
        2. Extract frames from video files
        3. Re-rank with VLM
        4. Return scored results with thumbnails
        """
        retrieval_k = top_k or self.top_k

        # Step 1: Retrieve candidates from ChromaDB
        self.logger.info(
            f"Querying ChromaDB: '{query_text}' (top_k={retrieval_k}, camera={camera_id})"
        )
        candidates = self.vector_store.query(
            query_text=query_text,
            top_k=retrieval_k,
            camera_id=camera_id,
        )

        if not candidates:
            self.logger.info("No candidates found in ChromaDB")
            return []

        self.logger.info(f"Retrieved {len(candidates)} candidates from ChromaDB")

        # Step 2: Extract frames and build candidate list for VLM
        vlm_candidates = []
        for cand in candidates:
            meta = cand["metadata"]
            cam = meta.get("camera_id", "")
            video_pos = meta.get("video_pos_ms")
            bbox = self._parse_bbox(meta.get("bbox"))

            if video_pos is None:
                self.logger.warning(f"Skipping candidate without video_pos_ms: {cand['id']}")
                continue

            full_frame, crop = self.frame_extractor.extract_frame(
                camera_id=cam,
                video_pos_ms=video_pos,
                bbox=bbox,
            )

            # Use crop if available, otherwise full frame
            frame = crop if crop is not None else full_frame
            if frame is None:
                self.logger.warning(f"Could not extract frame for candidate: {cand['id']}")
                continue

            vlm_candidates.append(CandidateFrame(
                camera_id=cam,
                timestamp=meta.get("timestamp", 0.0),
                video_pos_ms=video_pos,
                global_id=int(meta.get("global_id", 0)),
                class_label=meta.get("class_label", "unknown"),
                color=meta.get("color", "unknown"),
                type=meta.get("type"),
                bbox=bbox,
                frame=frame,
                distance=cand.get("distance", 1.0),
            ))

        if not vlm_candidates:
            self.logger.info("No frames extracted, returning empty results")
            return []

        self.logger.info(f"Extracted {len(vlm_candidates)} frames, starting VLM re-ranking")

        # Step 3: VLM re-rank
        ranked = self.vlm.rerank(
            query=query_text,
            candidates=vlm_candidates,
            top_k=self.rerank_top_k,
        )

        # Step 4: Build response
        results = []
        for i, r in enumerate(ranked):
            ts_human = datetime.fromtimestamp(
                r.timestamp, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S UTC")

            thumbnail = None
            if r.frame:
                try:
                    thumbnail = self._image_to_base64(r.frame)
                except Exception:
                    pass

            results.append(QueryResultItem(
                rank=i + 1,
                camera_id=r.camera_id,
                timestamp=r.timestamp,
                video_pos_ms=r.video_pos_ms,
                timestamp_human=ts_human,
                global_id=r.global_id,
                class_label=r.class_label,
                color=r.color,
                type=r.type,
                vlm_score=r.vlm_score,
                vlm_explanation=r.vlm_explanation,
                thumbnail_b64=thumbnail,
            ))

        return results
