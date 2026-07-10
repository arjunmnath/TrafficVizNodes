import numpy as np
import torch
from typing import Any, List, Optional, Tuple

from ultralytics.trackers.byte_tracker import BYTETracker, STrack
from ultralytics.trackers.utils import matching
from ultralytics.trackers.utils.stracks import parse_bboxes

from .association.base import IoUCost
from .association.appearance import AppearanceCost
from .association.fusion import CostFusion


class EnhancedSTrack(STrack):
    """STrack representation extended with appearance embeddings for matching."""

    def __init__(self, xywh: np.ndarray, score: float, cls: Any, feat: Optional[np.ndarray] = None):
        """Initialize track with appearance features if available.

        Args:
            xywh (np.ndarray): Bounding box in center x, center y, width, height format.
            score (float): Detection confidence score.
            cls (Any): Class label.
            feat (np.ndarray, optional): Appearance feature vector.
        """
        super().__init__(xywh, score, cls)
        self.curr_feat: Optional[np.ndarray] = None
        self.smooth_feat: Optional[np.ndarray] = None
        self.alpha: float = 0.9  # Default blending factor for exponential moving average

        if feat is not None:
            self.update_features(feat)

    def update_features(self, feat: np.ndarray) -> None:
        """Smooth the appearance feature vector using exponential moving average.

        Args:
            feat (np.ndarray): The new appearance feature vector.
        """
        from ultralytics.trackers.utils.reid import smooth_feature
        curr, smooth = smooth_feature(feat, self.smooth_feat, self.alpha)
        if curr is not None:
            self.curr_feat, self.smooth_feat = curr, smooth

    def update(self, new_track: "EnhancedSTrack", frame_id: int) -> None:
        """Update track state and smooth appearance features upon match.

        Args:
            new_track (EnhancedSTrack): The matched detection track.
            frame_id (int): ID of the current frame.
        """
        if hasattr(new_track, "curr_feat") and new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().update(new_track, frame_id)

    def re_activate(self, new_track: "EnhancedSTrack", frame_id: int, new_id: bool = False) -> None:
        """Reactivate track state and smooth appearance features.

        Args:
            new_track (EnhancedSTrack): The matching detection.
            frame_id (int): ID of the current frame.
            new_id (bool): Whether to allocate a new track ID.
        """
        if hasattr(new_track, "curr_feat") and new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().re_activate(new_track, frame_id, new_id)


class EnhancedByteTracker(BYTETracker):
    """An enhanced ByteTrack tracker incorporating appearance-based association cost."""

    track_class = EnhancedSTrack

    def __init__(self, args: Any):
        """Initialize custom tracker and its modular cost/fusion components.

        Args:
            args (Any): Track settings configuration namespace.
        """
        super().__init__(args)
        
        # Load ReID feature encoder (reusing standard build_encoder helper)
        from ultralytics.trackers.utils.reid import build_encoder
        self.encoder = build_encoder(
            with_reid=False,
            model=getattr(args, "model", "auto"),
            device=getattr(args, "device", None)
        )

        # Initialize modular cost handlers
        self.iou_cost_module = IoUCost()
        self.appearance_cost_module = AppearanceCost()

        # Read cost weights from args
        self.iou_weight = getattr(args, "iou_weight", 0.7)
        self.appearance_weight = getattr(args, "appearance_weight", 0.3)

    @classmethod
    def setup_predictor(cls, predictor: Any) -> None:
        """Register detection pre-hook to extract features when using model='auto'.

        This classmethod is dynamically called by the Ultralytics tracking pipeline.

        Args:
            predictor (Any): The YOLO predictor instance.
        """
        if not hasattr(predictor, "trackers") or not predictor.trackers:
            return

        tracker = predictor.trackers[0]
        cfg = tracker.args

        # Register hook to extract intermediate feature maps for 'auto' ReID mode
        if getattr(cfg, "with_reid", False) and getattr(cfg, "model", None) == "auto":
            from ultralytics.nn.modules.head import Detect

            if (
                isinstance(predictor.model.model, torch.nn.Module)
                and isinstance(predictor.model.model.model[-1], Detect)
                and not predictor.model.model.model[-1].end2end
            ):
                # Register hook to extract input of Detect layer
                def pre_hook(module, input):
                    predictor._feats = list(input[0])  # unroll input tensors to prevent mutation

                predictor._hook = predictor.model.model.model[-1].register_forward_pre_hook(pre_hook)
                predictor._feats = None  # Initialize _feats state

    def extract_features(self, frame: np.ndarray, bboxes: np.ndarray) -> Optional[List[np.ndarray]]:
        """Extract appearance embeddings for detection bounding boxes.

        Args:
            frame (np.ndarray): Image frame or pre-hook/manual extracted features.
            bboxes (np.ndarray): Detected boxes in xywh format.

        Returns:
            Optional[List[np.ndarray]]: List of appearance embeddings, or None.
        """
        # If frame is already pre-extracted features passed manually:
        if frame is not None:
            if isinstance(frame, np.ndarray) and frame.ndim == 2:
                return [f for f in frame]
            elif isinstance(frame, list):
                return frame

        if self.encoder is not None and frame is not None and len(bboxes) > 0:
            return self.encoder(frame, bboxes)
        return None

    def init_track(self, results: Any, img: Optional[np.ndarray] = None) -> List[EnhancedSTrack]:
        """Initialize STrack detections with extracted feature embeddings.

        Args:
            results (Any): YOLO detections list.
            img (Optional[np.ndarray]): Input frame or pre-hook/manual features.

        Returns:
            List[EnhancedSTrack]: Detections represented as EnhancedSTrack instances.
        """
        if len(results) == 0:
            return []
        
        bboxes = parse_bboxes(results)
        features = self.extract_features(img, bboxes)

        tracks = []
        for i, (xywh, s, c) in enumerate(zip(bboxes, results.conf, results.cls)):
            feat = features[i] if features is not None and i < len(features) else None
            track = self.track_class(xywh, s, c, feat)
            track.alpha = getattr(self.args, "ema_alpha", 0.9)
            tracks.append(track)
        return tracks

    def get_dists(self, tracks: List[EnhancedSTrack], detections: List[EnhancedSTrack]) -> np.ndarray:
        """Compute the combined cost matrix using custom weights and CostFusion.

        Args:
            tracks (List[EnhancedSTrack]): List of current active tracks.
            detections (List[EnhancedSTrack]): List of new detections.

        Returns:
            np.ndarray: Fused association cost matrix.
        """
        if not tracks or not detections:
            return np.empty((len(tracks), len(detections)), dtype=np.float32)

        # 1. Compute individual component cost matrices
        iou_cost = self.iou_cost_module.compute(tracks, detections)
        appearance_cost = self.appearance_cost_module.compute(tracks, detections)

        # 2. Normalize appearance cost (from cosine distance range [0, 2] to [0, 1])
        appearance_cost = appearance_cost / 2.0

        # Check where appearance features are missing and fall back to pure iou_cost
        track_has_no_feat = np.array([t.smooth_feat is None for t in tracks])
        det_has_no_feat = np.array([d.curr_feat is None for d in detections])

        if track_has_no_feat.any() or det_has_no_feat.any():
            missing_mask = track_has_no_feat[:, None] | det_has_no_feat[None, :]
        else:
            missing_mask = None

        # 3. Combine matrices using CostFusion module
        costs = {
            "iou": iou_cost,
            "appearance": appearance_cost
        }
        weights = {
            "iou": self.iou_weight,
            "appearance": self.appearance_weight
        }
        
        fused_cost = CostFusion.combine(costs, weights)

        # Override missing feature pairs to pure iou_cost
        if missing_mask is not None:
            fused_cost = np.where(missing_mask, iou_cost, fused_cost)

        # 4. Apply optional detection score fusion (matching standard ByteTrack behaviour)
        if self.args.fuse_score:
            fused_cost = matching.fuse_score(fused_cost, detections)

        return fused_cost
