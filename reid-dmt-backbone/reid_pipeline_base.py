#!/usr/bin/env python3
"""
Base components and common utilities for the ReID processing pipelines.
Provides a shared identity registry, listener interfaces, and the base tracking loop.
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import torch

class SimpleRegistry:
    def __init__(self, match_threshold=0.6):
        self.identities = {}  # global_id -> {"embedding": np.ndarray, "occurrences": []}
        self.next_id = 1
        self.match_threshold = match_threshold

    def match_and_add(self, embedding, video_name, frame_num, timestamp, bbox, local_track_id, class_label):
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

        occurrence = {
            "video": video_name,
            "frame": frame_num,
            "timestamp_seconds": round(timestamp, 2),
            "bbox": [int(x) for x in bbox],
            "local_track_id": int(local_track_id),
            "class_label": class_label,
            "similarity": round(best_sim, 4) if best_id is not None else 1.0
        }

        if best_id is not None and best_sim >= self.match_threshold:
            self.identities[best_id]["occurrences"].append(occurrence)
            self.identities[best_id]["embedding"] = self.update_prototype(
                self.identities[best_id]["embedding"],
                emb_norm
            )
            return best_id, best_sim
        else:
            new_id = self.next_id
            self.next_id += 1
            self.identities[new_id] = {
                "embedding": embedding,
                "occurrences": [occurrence]
            }
            return new_id, 1.0

    def get_results_summary(self):
        summary = []
        for global_id, data in self.identities.items():
            summary.append({
                "global_id": global_id,
                "occurrences": data["occurrences"]
            })
        return summary

    def update_prototype(
            self,
            prototype: np.ndarray,
            embedding: np.ndarray,
            alpha: float = 0.1,
            similarity_threshold: float = 0.8,
    ) -> np.ndarray:
        similarity = np.dot(prototype, embedding)

        # Reject dissimilar embeddings
        if similarity < similarity_threshold:
            return prototype

        # Weighted update
        updated = (1 - alpha) * prototype + alpha * embedding

        # L2 normalize
        updated /= np.linalg.norm(updated)

        return updated


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_valid_crop(bbox, frame_shape) -> bool:
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    W, H, _ = frame_shape
    return (w * h) / (W * H) > 5e-3


def resolve_path(p, base_dir):
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(base_dir, p))


class ReIDPipelineListener:
    """Interface for listening to ReID pipeline execution events."""
    def on_init_start(self):
        pass

    def on_init_status(self, message: str):
        pass

    def on_init_end(self):
        pass

    def on_video_start(self, video_path: str, video_idx: int, total_videos: int, total_frames: int, fps: float):
        pass

    def on_frame_processed(self, video_name: str, video_idx: int, total_videos: int, frame_count: int,
                           total_frames: int, elapsed_time: float, fps: float, registry: SimpleRegistry,
                           log_message: str | None = None):
        pass

    def on_video_end(self, video_path: str, total_frames: int):
        pass

    def on_pipeline_end(self, registry: SimpleRegistry, output_path: str):
        pass

    def on_error(self, message: str):
        pass


class BaseReIDPipeline:
    """Base ReID Tracking Pipeline that implements the unified tracking loop."""
    def __init__(self, yolo_path, threshold=0.8, max_frames=0, sample_fps=0.0, output_path="reid_test_results.json"):
        self.yolo_path = yolo_path
        self.threshold = threshold
        self.max_frames = max_frames
        self.sample_fps = sample_fps
        self.output_path = output_path
        self.registry = SimpleRegistry(match_threshold=threshold)
        
        self.tracker = None

    def initialize(self, listener: ReIDPipelineListener = None):
        if listener:
            listener.on_init_start()

        # Delegated to subclasses for specific model/weights loading
        self._initialize_extractor(listener)

        if listener:
            listener.on_init_status("Loading YOLOv8 Tracker...")
        from ultralytics import YOLO
        self.tracker = YOLO(self.yolo_path)

        if listener:
            listener.on_init_end()

    def _initialize_extractor(self, listener: ReIDPipelineListener = None):
        raise NotImplementedError("Subclasses must implement _initialize_extractor")

    def _extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        """Extract a single L2-normalized embedding vector for a crop in BGR format.
        
        Must return a numpy array of shape (embedding_dim,).
        """
        raise NotImplementedError("Subclasses must implement _extract_embedding")

    def run(self, videos: list[str], listener: ReIDPipelineListener = None):
        try:
            for idx, video in enumerate(videos):
                if not os.path.exists(video):
                    if listener:
                        listener.on_error(f"Video file not found: {video}")
                    continue

                cap = cv2.VideoCapture(video)
                if not cap.isOpened():
                    if listener:
                        listener.on_error(f"Failed to open video: {video}")
                    continue

                video_name = os.path.basename(video)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_count = 0
                processed_count = 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Calculate frame interval for sampling
                frame_interval = 1
                if self.sample_fps > 0:
                    frame_interval = max(1, int(round(fps / self.sample_fps)))
                    if listener:
                        listener.on_frame_processed(
                            video_name=video_name,
                            video_idx=idx + 1,
                            total_videos=len(videos),
                            frame_count=0,
                            total_frames=total_frames,
                            elapsed_time=0.0,
                            fps=0.0,
                            registry=self.registry,
                            log_message=f"Sampling video at {self.sample_fps} FPS (every {frame_interval} frames)"
                        )

                if listener:
                    listener.on_video_start(video, idx + 1, len(videos), total_frames, fps)

                start_time = time.time()

                while True:
                    if self.max_frames > 0 and frame_count >= self.max_frames:
                        break

                    # Skip frame using fast grab if not in interval
                    if frame_interval > 1 and frame_count % frame_interval != 0:
                        if not cap.grab():
                            break
                        frame_count += 1
                        continue

                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    processed_count += 1
                    timestamp = frame_count / fps

                    results = self.tracker.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        verbose=False
                    )

                    elapsed = time.time() - start_time
                    curr_fps = processed_count / (elapsed + 1e-5)

                    if not results or len(results) == 0:
                        if listener:
                            listener.on_frame_processed(
                                video_name=video_name,
                                video_idx=idx + 1,
                                total_videos=len(videos),
                                frame_count=frame_count,
                                total_frames=total_frames,
                                elapsed_time=elapsed,
                                fps=curr_fps,
                                registry=self.registry
                            )
                        continue

                    boxes_res = results[0].boxes
                    if boxes_res is None or boxes_res.id is None:
                        if listener:
                            listener.on_frame_processed(
                                video_name=video_name,
                                video_idx=idx + 1,
                                total_videos=len(videos),
                                frame_count=frame_count,
                                total_frames=total_frames,
                                elapsed_time=elapsed,
                                fps=curr_fps,
                                registry=self.registry
                            )
                        continue

                    boxes = boxes_res.xyxy.cpu().numpy()
                    track_ids = boxes_res.id.int().cpu().numpy()
                    cls_ids = boxes_res.cls.int().cpu().numpy()

                    for crop_idx, (box_coords, track_id, cls_id) in enumerate(zip(boxes, track_ids, cls_ids)):
                        x1, y1, x2, y2 = map(int, box_coords)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                        crop = frame[y1:y2, x1:x2]
                        if not is_valid_crop(box_coords, frame.shape):
                            continue

                        # Extract embedding using the adapted method
                        embedding = self._extract_embedding(crop)

                        class_label = results[0].names.get(int(cls_id), "unknown")

                        global_id, similarity = self.registry.match_and_add(
                            embedding=embedding,
                            video_name=video_name,
                            frame_num=frame_count,
                            timestamp=timestamp,
                            bbox=[x1, y1, x2, y2],
                            local_track_id=track_id,
                            class_label=class_label
                        )

                        t_str = time.strftime('%H:%M:%S')
                        if similarity >= self.registry.match_threshold:
                            log_line = f"[{t_str}] [bold green]MATCH[/bold green] - Track {track_id} ({class_label}) -> Global ID [bold green]{global_id:03d}[/bold green] (sim: {similarity:.3f})"
                        else:
                            log_line = f"[{t_str}] [bold cyan]NEW  [/bold cyan] - Track {track_id} ({class_label}) -> Registered as Global ID [bold cyan]{global_id:03d}[/bold cyan]"

                        if listener:
                            listener.on_frame_processed(
                                video_name=video_name,
                                video_idx=idx + 1,
                                total_videos=len(videos),
                                frame_count=frame_count,
                                total_frames=total_frames,
                                elapsed_time=elapsed,
                                fps=curr_fps,
                                registry=self.registry,
                                log_message=log_line
                            )

                    # Periodic update to refresh layout and display progress
                    if listener:
                        listener.on_frame_processed(
                            video_name=video_name,
                            video_idx=idx + 1,
                            total_videos=len(videos),
                            frame_count=frame_count,
                            total_frames=total_frames,
                            elapsed_time=elapsed,
                            fps=curr_fps,
                            registry=self.registry
                        )

                cap.release()
                if listener:
                    listener.on_video_end(video, frame_count)

        except KeyboardInterrupt:
            if listener:
                listener.on_error("Pipeline interrupted by user (Ctrl+C). Saving partial results...")
            if 'cap' in locals() and cap.isOpened():
                cap.release()

        # Save results
        summary = self.registry.get_results_summary()
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(summary, f, indent=4)

        if listener:
            listener.on_pipeline_end(self.registry, self.output_path)
