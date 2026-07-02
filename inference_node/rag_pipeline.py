import base64
import io
from datetime import datetime, timezone
from typing import List, Optional

from PIL import Image

from inference_node.frame_extractor import FrameExtractor
from inference_node.retrieval.search import RetrievalEngine
from inference_node.vqa import BaseVQAReasoner, CandidateImage
from shared.schemas import QueryResultItem
from shared.utils import setup_logger


class RAGPipeline:
    """Orchestrates query understanding, semantic retrieval, frame extraction, and VQA reasoning."""

    def __init__(
        self,
        retrieval_engine: RetrievalEngine,
        frame_extractor: FrameExtractor,
        reasoner: BaseVQAReasoner,
        retrieval_top_k: int = 20,
        rerank_top_k: int = 5,
    ) -> None:
        self.logger = setup_logger("RAGPipeline")
        self.retrieval = retrieval_engine
        self.frame_extractor = frame_extractor
        self.reasoner = reasoner
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k

    def _image_to_base64(self, img: Image.Image, max_size: int = 320) -> str:
        img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _parse_bbox(bbox_str: Optional[str]) -> Optional[list]:
        if not bbox_str:
            return None
        try:
            return [float(value) for value in bbox_str.split(",")]
        except (ValueError, AttributeError):
            return None

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        camera_id: Optional[str] = None,
    ) -> List[QueryResultItem]:
        """Execute the retrieval + reasoning pipeline."""
        retrieval_k = top_k or self.retrieval_top_k

        parsed, candidates = self.retrieval.search(
            query=query_text,
            top_k=retrieval_k,
            camera_id=camera_id,
        )

        if not candidates:
            self.logger.info("No candidates found after retrieval")
            return []

        self.logger.info(f"Retrieved {len(candidates)} candidates")

        reasoning_candidates: List[CandidateImage] = []
        for candidate in candidates:
            bbox = self._parse_bbox(candidate.bbox)
            full_frame, crop = self.frame_extractor.extract_frame(
                camera_id=candidate.camera_id,
                video_pos_ms=candidate.video_pos_ms,
                bbox=bbox,
            )
            frame = crop if crop is not None else full_frame
            if frame is None:
                self.logger.warning(f"Could not extract frame for candidate: {candidate.id}")
                continue

            reasoning_candidates.append(
                CandidateImage(
                    camera_id=candidate.camera_id,
                    camera_timestamp=candidate.camera_timestamp,
                    video_pos_ms=candidate.video_pos_ms,
                    track_id=candidate.track_id,
                    bbox=bbox,
                    frame=frame,
                    retrieval_distance=candidate.distance,
                )
            )

        if not reasoning_candidates:
            self.logger.info("No frames extracted, returning empty results")
            return []

        ranked = self.reasoner.answer(
            query=parsed.semantic_text or query_text,
            candidates=reasoning_candidates,
            top_k=self.rerank_top_k,
        )

        results: List[QueryResultItem] = []
        for index, result in enumerate(ranked):
            ts_human = datetime.fromtimestamp(
                result.camera_timestamp, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S UTC")

            thumbnail = None
            if result.frame:
                try:
                    thumbnail = self._image_to_base64(result.frame)
                except Exception:
                    pass

            results.append(
                QueryResultItem(
                    rank=index + 1,
                    camera_id=result.camera_id,
                    timestamp=result.camera_timestamp,
                    video_pos_ms=result.video_pos_ms,
                    timestamp_human=ts_human,
                    global_id=result.track_id,
                    class_label="unknown",
                    color="unknown",
                    type=None,
                    vlm_score=result.vlm_score,
                    vlm_explanation=result.vlm_explanation,
                    thumbnail_b64=thumbnail,
                )
            )

        return results
