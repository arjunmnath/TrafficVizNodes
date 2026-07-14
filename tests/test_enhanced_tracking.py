import numpy as np
from typing import Any, List

from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH
from ultralytics.trackers.basetrack import TrackState
from reid.tracking.enhanced_bytetrack import EnhancedSTrack, EnhancedByteTracker
from reid.tracking.occlusion import OcclusionManager, bbox_iou
from reid.tracking.quality import TrackQuality
from reid.tracking.tracker import Tracker, Detections


class MockArgs:
    """Mock configuration arguments for EnhancedByteTracker."""

    def __init__(self) -> None:
        self.tracker_type = "enhanced_bytetrack"
        self.track_high_thresh = 0.25
        self.track_low_thresh = 0.1
        self.new_track_thresh = 0.25
        self.track_buffer = 30
        self.match_thresh = 0.2  # Lower match threshold to fail normal association for recalls
        self.fuse_score = True
        self.iou_weight = 0.7
        self.appearance_weight = 0.3
        self.occlusion_enabled = True
        self.occlusion_timeout = 5
        self.occlusion_similarity_threshold = 0.6
        self.occlusion_spatial_threshold = 0.1
        self.quality_enabled = True
        self.quality_weights = {
            "detector_confidence": 0.3,
            "track_duration": 0.2,
            "embedding_stability": 0.3,
            "association_consistency": 0.2,
        }
        self.lifecycle_events_enabled = True


def test_bbox_iou() -> None:
    box1 = np.array([10, 10, 20, 20])
    box2 = np.array([15, 10, 25, 20])
    # Intersection: [15, 10, 20, 20] -> width 5, height 10 -> area 50
    # Box1 area: 100, Box2 area: 100. Union: 100 + 100 - 50 = 150
    # IoU: 50 / 150 = 0.3333
    assert abs(bbox_iou(box1, box2) - 0.333333) < 1e-4

    box_disjoint = np.array([30, 30, 40, 40])
    assert bbox_iou(box1, box_disjoint) == 0.0


def test_occlusion_manager_cleanup() -> None:
    mgr = OcclusionManager(timeout=3, similarity_threshold=0.5, spatial_threshold=0.1)

    # Mock track objects
    class MockTrack:
        def __init__(self, track_id: int, end_frame: int, cls: int) -> None:
            self.track_id = track_id
            self.end_frame = end_frame
            self.cls = cls

    t1 = MockTrack(track_id=1, end_frame=10, cls=0)
    t2 = MockTrack(track_id=2, end_frame=12, cls=0)

    mgr.add_lost_track(t1)
    mgr.add_lost_track(t2)

    assert len(mgr.lost_tracks) == 2

    # Cleanup at frame 14.
    # t1: 14 - 10 = 4 > 3 -> evicted
    # t2: 14 - 12 = 2 <= 3 -> kept
    mgr.cleanup(14)
    assert len(mgr.lost_tracks) == 1
    assert 2 in mgr.lost_tracks
    assert 1 not in mgr.lost_tracks


def test_occlusion_manager_query() -> None:
    mgr = OcclusionManager(timeout=5, similarity_threshold=0.6, spatial_threshold=0.1)

    # Setup lost tracks
    feat_t1 = np.array([1.0, 0.0])
    t1 = EnhancedSTrack(np.array([100, 100, 50, 50, 0]), 0.9, 0, feat_t1)
    t1.track_id = 42
    t1.frame_id = 10

    mgr.add_lost_track(t1)

    # Query track that matches
    feat_q1 = np.array([0.9, 0.1])
    q1 = EnhancedSTrack(np.array([102, 102, 50, 50, 0]), 0.9, 0, feat_q1)

    # Query with mismatching class
    q_wrong_cls = EnhancedSTrack(np.array([102, 102, 50, 50, 0]), 0.9, 1, feat_q1)

    # Query with low similarity
    feat_low_sim = np.array([0.0, 1.0])
    q_low_sim = EnhancedSTrack(np.array([102, 102, 50, 50, 0]), 0.9, 0, feat_low_sim)

    # Query with spatial mismatch
    q_far = EnhancedSTrack(np.array([300, 300, 50, 50, 0]), 0.9, 0, feat_q1)

    assert len(mgr.query(q1, frame_id=12)) == 1
    assert len(mgr.query(q_wrong_cls, frame_id=12)) == 0
    assert len(mgr.query(q_low_sim, frame_id=12)) == 0
    assert len(mgr.query(q_far, frame_id=12)) == 0

    # Query beyond timeout
    assert len(mgr.query(q1, frame_id=20)) == 0


def test_recall_support() -> None:
    kf = KalmanFilterXYAH()
    feat1 = np.array([1.0, 0.0, 0.0])
    track = EnhancedSTrack(np.array([100, 100, 50, 50, 0]), 0.9, 0, feat1)
    track.activate(kf, frame_id=1)
    original_id = track.track_id

    # Make it lost
    track.mark_lost()
    assert track.state == TrackState.Lost

    # Create new detection track
    feat2 = np.array([0.95, 0.05, 0.0])
    new_det = EnhancedSTrack(np.array([105, 105, 50, 50, 0]), 0.95, 0, feat2)

    # Recall the track
    track.recall(new_det, frame_id=2)

    assert track.track_id == original_id
    assert track.state == TrackState.Tracked
    assert track.is_activated is True
    assert track.frame_id == 2
    assert track.score == 0.95
    assert track.recall_count == 1
    assert track.lost_recovered_count == 1
    # Check appearance smoothed
    assert track.curr_feat is not None
    assert abs(np.dot(track.smooth_feat, feat2) - 1.0) < 0.1


def test_track_quality() -> None:
    feat = np.array([1.0, 0.0])
    track = EnhancedSTrack(np.array([100, 100, 50, 50, 0]), 0.8, 0, feat)

    # Add detection updates to improve quality
    kf = KalmanFilterXYAH()
    track.activate(kf, frame_id=1)

    t2 = EnhancedSTrack(np.array([101, 101, 50, 50, 0]), 0.9, 0, feat)
    track.update(t2, frame_id=2)

    assert track.num_detections == 2
    assert abs(track.total_conf - 1.7) < 1e-6
    assert track.consecutive_associations == 2
    assert track.embedding_stability_sum == 2.0  # perfect stability (similarity 1.0 twice)

    score = TrackQuality.evaluate(track)
    assert 0.0 <= score <= 1.0
    assert track.quality_score == score


def test_event_dispatcher() -> None:
    args = MockArgs()
    tracker = EnhancedByteTracker(args)

    events: List[Any] = []

    def on_event(event_type: str, track_id: int, frame_id: int, timestamp: float, track: Any) -> None:
        events.append((event_type, track_id, timestamp))

    tracker.subscribe(on_event)

    # Setup frame detections mock
    results = Detections(
        xywh=np.array([[100.0, 100.0, 50.0, 50.0, 0.0]]),
        conf=np.array([0.9]),
        cls=np.array([0]),
    )
    feat1 = np.array([[1.0, 0.0]])

    # Frame 1: Create Track
    tracker.update(results, feats=feat1, timestamp=1.0)
    assert len(events) == 1
    assert events[0] == ("created", 1, 1.0)

    # Frame 2: Update Track
    results_updated = Detections(
        xywh=np.array([[102.0, 102.0, 50.0, 50.0, 0.0]]),
        conf=np.array([0.95]),
        cls=np.array([0]),
    )
    feat2 = np.array([[0.95, 0.05]])
    tracker.update(results_updated, feats=feat2, timestamp=2.0)
    assert len(events) == 2
    assert events[1] == ("updated", 1, 2.0)

    # Frame 3: Track Lost
    empty = Detections(
        xywh=np.empty((0, 4)),
        conf=np.empty((0,)),
        cls=np.empty((0,)),
    )
    tracker.update(empty, feats=np.empty((0, 2)), timestamp=3.0)
    # The track went lost!
    assert len(events) == 3
    assert events[2] == ("lost", 1, 3.0)

    # Check track added to OcclusionManager
    assert 1 in tracker.occlusion_manager.lost_tracks

    # Frame 4: Track Recalled (fails normal association, but passes occlusion spatial gating)
    results_recalled = Detections(
        xywh=np.array([[127.0, 102.0, 50.0, 50.0, 0.0]]),
        conf=np.array([0.95]),
        cls=np.array([0]),
    )
    tracker.update(results_recalled, feats=feat2, timestamp=4.0)
    # The track should be recalled from occlusion manager
    assert len(events) == 4
    assert events[3] == ("recalled", 1, 4.0)


def test_tracker_wrapper_event_wiring() -> None:
    # Test that the Wrapper Tracker class correctly subscribes and populates internal buffers
    config_dict = {
        "tracker_type": "enhanced_bytetrack",
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.25,
        "track_buffer": 30,
        "match_thresh": 0.2,  # Match threshold to fail normal association on recalls
        "fuse_score": True,
        "iou_weight": 0.7,
        "appearance_weight": 0.3,
        "occlusion_enabled": True,
        "occlusion_timeout": 30,
        "occlusion_similarity_threshold": 0.5,
        "occlusion_spatial_threshold": 0.1,
        "quality_enabled": True,
        "quality_weights": {
            "detector_confidence": 0.3,
            "track_duration": 0.2,
            "embedding_stability": 0.3,
            "association_consistency": 0.2,
        },
        "lifecycle_events_enabled": True,
    }

    t_wrapper = Tracker(config_dict)

    terminated_ids: List[int] = []

    def on_terminated(track: Any) -> None:
        terminated_ids.append(track.track_id)

    t_wrapper.on_track_terminated = on_terminated

    boxes = np.array([[100.0, 100.0, 150.0, 150.0]])
    scores = np.array([0.9])
    classes = np.array([0])
    feats = np.ones((1, 128), dtype=np.float32)

    # Frame 1: Create
    t_wrapper.update(boxes, scores, classes, features=feats, frame_count=1, timestamp=1.0)
    assert 1 in t_wrapper.track_history
    assert 1 in t_wrapper.track_embeddings

    # Make the lost track buffer tiny so it gets removed/terminated quickly
    t_wrapper.tracker.max_frames_lost = 1

    # Frame 2: Lost
    t_wrapper.update(np.empty((0, 4)), np.empty((0,)), np.empty((0,)), frame_count=2, timestamp=2.0)

    # Frame 3: Stale -> Terminated
    t_wrapper.update(np.empty((0, 4)), np.empty((0,)), np.empty((0,)), frame_count=3, timestamp=3.0)

    # Track 1 should be terminated and termination hook should have fired
    assert 1 in terminated_ids
    assert 1 not in t_wrapper.track_history
    assert 1 not in t_wrapper.track_embeddings
