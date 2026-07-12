from dataclasses import dataclass


@dataclass(frozen=True)
class TrackMetadata:
    """Contains immutable metadata about a tracked entity."""

    track_id: int
    camera_id: str
    class_label: str
    start_frame: int
    end_frame: int
    start_timestamp: float
    end_timestamp: float
