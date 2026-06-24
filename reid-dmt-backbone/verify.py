from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm

MIN_TIME_GAP_SECONDS = 3.0


def should_keep_detections(
    data: dict,
    min_time_gap_seconds: float,
) -> dict[str, list[dict]]:
    """
    Filter detections so that for each global identity:
      - first occurrence from every camera is kept
      - subsequent occurrences from the same camera are only kept
        if sufficiently far apart in time
    """

    filtered: dict[str, list[dict]] = {}

    for global_id, detections in data.items():

        by_video: dict[str, list[dict]] = defaultdict(list)

        for det in detections:
            by_video[det["video"]].append(det)

        kept: list[dict] = []

        for video_name, video_dets in by_video.items():

            video_dets.sort(
                key=lambda x: (
                    x.get("timestamp_seconds", 0.0),
                    x["frame"],
                )
            )

            last_saved_time: float | None = None

            for det in video_dets:

                current_time = float(
                    det.get("timestamp_seconds", 0.0)
                )

                if last_saved_time is None:
                    kept.append(det)
                    last_saved_time = current_time
                    continue

                if (
                    current_time - last_saved_time
                    >= min_time_gap_seconds
                ):
                    kept.append(det)
                    last_saved_time = current_time

        filtered[global_id] = kept

    return filtered


def extract_reid_crops(
    json_path: str,
    video_dir: str,
    output_dir: str,
    min_time_gap_seconds: float = MIN_TIME_GAP_SECONDS,
) -> None:

    json_path = Path(json_path)
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    data = should_keep_detections(
        data,
        min_time_gap_seconds=min_time_gap_seconds,
    )

    by_video: dict[str, list[dict]] = defaultdict(list)

    for global_id, detections in data.items():
        for det in detections:
            det["global_id"] = global_id
            by_video[det["video"]].append(det)

    for video_name, detections in tqdm(
        by_video.items(),
        desc="Videos",
    ):
        video_path = video_dir / video_name

        if not video_path.exists():
            print(f"[WARNING] Missing video: {video_path}")
            continue

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"[WARNING] Could not open: {video_path}")
            continue

        detections.sort(key=lambda x: x["frame"])

        current_frame_idx = -1
        frame = None

        for det in detections:

            frame_idx = int(det["frame"])

            if frame_idx != current_frame_idx:

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

                success, frame = cap.read()

                if not success:
                    print(
                        f"[WARNING] Failed reading frame "
                        f"{frame_idx} from {video_name}"
                    )
                    continue

                current_frame_idx = frame_idx

            if frame is None:
                continue

            x1, y1, x2, y2 = map(int, det["bbox"])

            h, w = frame.shape[:2]

            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]

            global_id = str(det["global_id"])
            local_track_id = det.get("local_track_id", -1)

            person_dir = output_dir / global_id
            person_dir.mkdir(parents=True, exist_ok=True)

            stem = Path(video_name).stem

            timestamp = det.get("timestamp_seconds", 0.0)

            out_name = (
                f"{stem}"
                f"_f{frame_idx:06d}"
                f"_t{local_track_id}"
                f"_s{timestamp:.2f}.jpg"
            )

            cv2.imwrite(
                str(person_dir / out_name),
                crop,
            )

        cap.release()


if __name__ == "__main__":
    extract_reid_crops(
        json_path="cleaned_reid.json",
        video_dir="input_vids",
        output_dir="reid_crops_cleaned",
        min_time_gap_seconds=3.0,
    )