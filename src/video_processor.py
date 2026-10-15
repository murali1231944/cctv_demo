import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union
import cv2
import numpy as np

import configs.config as config
from src.tracker import MultiPersonTracker
from src.embeddings import OSNetFeatureExtractor
from src.utils import get_video_info, compute_timestamp, crop_person_image, save_json_atomic

class VideoProcessor:
    """
    Complete video ingestion pipeline:
    - Reads video stream and extracts timeline timestamps
    - Runs YOLO detection and ByteTrack tracking across all frames
    - Records frame-level bounding boxes and timestamps for every track
    - Extracts representative person crops and OSNet appearance embeddings
    - Persists track metadata locally
    """
    def __init__(
        self,
        camera_id: str,
        video_path: Union[str, Path],
        extractor: Optional[OSNetFeatureExtractor] = None,
        conf_threshold: float = config.DETECTION_CONF_THRESHOLD,
        min_track_frames: int = config.MIN_TRACK_FRAMES,
        crops_dir: Union[str, Path] = config.CROPS_DIR,
        metadata_dir: Union[str, Path] = config.METADATA_DIR,
    ):
        self.camera_id = camera_id
        self.video_path = Path(video_path)
        self.conf_threshold = conf_threshold
        self.min_track_frames = min_track_frames
        self.crops_dir = Path(crops_dir)
        self.metadata_dir = Path(metadata_dir)

        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.video_info = get_video_info(self.video_path)
        self.fps = self.video_info["fps"]
        self.total_frames = self.video_info["total_frames"]

        self.tracker = MultiPersonTracker(
            camera_id=self.camera_id,
            fps=self.fps,
            conf_threshold=self.conf_threshold
        )
        self.extractor = extractor or OSNetFeatureExtractor()

    def process_video(
        self,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, float, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Processes video frames, extracts tracks, crops, and embeddings.
        Returns a dict mapping track_id -> track metadata.
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")

        self.tracker.reset()

        frames_to_process = self.total_frames
        if max_frames is not None and max_frames > 0:
            frames_to_process = min(self.total_frames, max_frames)

        # Temporary in-memory accumulation of observations and crop candidates per track
        # track_data[camera_track_id] = {
        #    "raw_track_id": int,
        #    "observations": List[dict],
        #    "crop_candidates": List[tuple(crop, score, frame_idx)]
        # }
        track_data: Dict[str, Dict[str, Any]] = {}

        start_wall_time = time.time()
        frame_idx = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if max_frames is not None and frame_idx >= max_frames:
                    break

                timestamp = compute_timestamp(frame_idx, self.fps)

                # Track active people in this frame
                active_tracks = self.tracker.update(frame, frame_idx)

                for item in active_tracks:
                    t_id = item["track_id"]
                    raw_id = item["raw_track_id"]
                    bbox = item["bbox"]
                    conf = item["confidence"]

                    if t_id not in track_data:
                        track_data[t_id] = {
                            "track_id": t_id,
                            "camera_id": self.camera_id,
                            "raw_track_id": raw_id,
                            "observations": [],
                            "crop_candidates": []
                        }

                    # Record frame-level observation
                    track_data[t_id]["observations"].append({
                        "frame_idx": int(frame_idx),
                        "timestamp": timestamp,
                        "bbox": bbox,
                        "confidence": conf
                    })

                    # Maintain a budget of up to 10 high-quality crops per track for embedding aggregation
                    crop = crop_person_image(frame, bbox)
                    if crop is not None and crop.shape[0] >= config.MIN_BBOX_HEIGHT and crop.shape[1] >= config.MIN_BBOX_WIDTH:
                        box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        quality_score = box_area * conf
                        candidates = track_data[t_id]["crop_candidates"]
                        if len(candidates) < 10:
                            candidates.append((crop, quality_score, frame_idx))
                        else:
                            # Replace lowest scoring candidate if new one is better
                            min_idx = min(range(len(candidates)), key=lambda k: candidates[k][1])
                            if quality_score > candidates[min_idx][1]:
                                candidates[min_idx] = (crop, quality_score, frame_idx)

                frame_idx += 1

                if progress_callback and (frame_idx % 10 == 0 or frame_idx == frames_to_process):
                    elapsed = max(time.time() - start_wall_time, 0.001)
                    curr_fps = frame_idx / elapsed
                    progress_callback(frame_idx, frames_to_process, curr_fps, len(track_data))

        finally:
            cap.release()

        # Post-process valid tracks (must have >= min_track_frames)
        final_tracks: Dict[str, Any] = {}
        for t_id, t_info in track_data.items():
            obs = t_info["observations"]
            if len(obs) < self.min_track_frames:
                continue

            # Sort crop candidates by quality score descending
            candidates = sorted(t_info["crop_candidates"], key=lambda c: c[1], reverse=True)
            if not candidates:
                # If no crop candidate survived, try cropping first observation frame
                continue

            best_crop = candidates[0][0]
            crop_filename = f"{t_id}.jpg"
            crop_path = self.crops_dir / crop_filename
            cv2.imwrite(str(crop_path), best_crop)

            # Aggregate embeddings from up to MAX_REPRESENTATIVE_CROPS
            selected_crops = [c[0] for c in candidates[:config.MAX_REPRESENTATIVE_CROPS]]
            embedding = self.extractor.aggregate_embeddings(selected_crops)

            start_time = obs[0]["timestamp"]
            end_time = obs[-1]["timestamp"]
            duration = round(end_time - start_time, 2)

            final_tracks[t_id] = {
                "track_id": t_id,
                "camera_id": self.camera_id,
                "raw_track_id": t_info["raw_track_id"],
                "source_video_path": str(self.video_path).replace("\\", "/"),
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "total_frames_observed": len(obs),
                "representative_crop_path": str(crop_path).replace("\\", "/"),
                "embedding": embedding.tolist(),
                "frame_observations": obs
            }

        # Persist camera metadata atomically
        camera_metadata_file = self.metadata_dir / f"{self.camera_id}_tracks.json"
        save_json_atomic(final_tracks, camera_metadata_file)

        return final_tracks
