# vim: expandtab:ts=4:sw=4
from typing import Tuple, Optional, Callable, Dict, Any, List
import colorsys
from .image_viewer import ImageViewer
from tqdm import tqdm  # type: ignore[import-untyped]


def create_unique_color_float(tag: int, hue_step: float = 0.41) -> Tuple[float, float, float]:
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1.0 - (int(tag * hue_step) % 4) / 5.0
    r, g, b = colorsys.hsv_to_rgb(h, 1.0, v)
    return r, g, b


def create_unique_color_uchar(tag: int, hue_step: float = 0.41) -> Tuple[int, int, int]:
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255 * r), int(255 * g), int(255 * b)


class NoVisualization(object):
    """
    A dummy visualization object that loops through all frames in a given
    sequence to update the tracker without performing any visualization.
    """

    def __init__(self, seq_info: Dict[str, Any]) -> None:
        self.frame_idx: int = seq_info["min_frame_idx"]
        self.last_idx: int = seq_info["max_frame_idx"]

    def set_image(self, image: Any) -> None:
        pass

    def draw_groundtruth(self, track_ids: List[int], boxes: Any) -> None:
        pass

    def draw_detections(self, detections: List[Any]) -> None:
        pass

    def draw_trackers(self, trackers: List[Any]) -> None:
        pass

    def run(self, frame_callback: Callable[["NoVisualization", int], None]) -> None:
        start_frame = self.frame_idx
        for frame_idx in tqdm(range(start_frame, self.last_idx + 1)):
            frame_callback(self, frame_idx)
            self.frame_idx += 1


class Visualization(object):
    """
    This class shows tracking output in an OpenCV image viewer.
    """

    def __init__(self, seq_info: Dict[str, Any], update_ms: int) -> None:
        image_shape = seq_info["image_size"][::-1]
        aspect_ratio = float(image_shape[1]) / image_shape[0]
        image_shape_res = 1024, int(aspect_ratio * 1024)
        self.viewer = ImageViewer(
            update_ms, image_shape_res, "Figure %s" % seq_info["sequence_name"]
        )
        self.viewer.thickness = 2
        self.frame_idx: int = seq_info["min_frame_idx"]
        self.last_idx: int = seq_info["max_frame_idx"]

    def run(self, frame_callback: Callable[["Visualization", int], None]) -> None:
        self.viewer.run(lambda: self._update_fun(frame_callback))

    def _update_fun(self, frame_callback: Callable[["Visualization", int], None]) -> bool:
        if self.frame_idx > self.last_idx:
            return False  # Terminate
        frame_callback(self, self.frame_idx)
        self.frame_idx += 1
        return True

    def set_image(self, image: Any) -> None:
        self.viewer.image = image

    def draw_groundtruth(self, track_ids: List[int], boxes: Any) -> None:
        self.viewer.thickness = 2
        for track_id, box in zip(track_ids, boxes):
            self.viewer.color = create_unique_color_uchar(track_id)
            self.viewer.rectangle(
                int(box[0]), int(box[1]), int(box[2]), int(box[3]), label=str(track_id)
            )

    def draw_detections(self, detections: List[Any]) -> None:
        self.viewer.thickness = 2
        self.viewer.color = (0, 0, 255)
        for i, detection in enumerate(detections):
            tlwh = detection.tlwh
            self.viewer.rectangle(float(tlwh[0]), float(tlwh[1]), float(tlwh[2]), float(tlwh[3]))

    def draw_trackers(self, tracks: List[Any]) -> None:
        self.viewer.thickness = 2
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            self.viewer.color = create_unique_color_uchar(track.track_id)
            tlwh = track.to_tlwh()
            self.viewer.rectangle(
                float(tlwh[0]),
                float(tlwh[1]),
                float(tlwh[2]),
                float(tlwh[3]),
                label=str(track.track_id),
            )
            # self.viewer.gaussian(track.mean[:2], track.covariance[:2, :2],
            #                      label="%d" % track.track_id)
