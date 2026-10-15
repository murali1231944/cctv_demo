import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import cv2
import numpy as np

import configs.config as config
from src.utils import load_json, format_timestamp, draw_bounding_box

class EvidenceGenerator:
    """
    Generates annotated video clips and visual evidence with vibrant green bounding boxes
    around the matched person throughout their observed appearances.
    """
    def __init__(
        self,
        metadata_dir: Union[str, Path] = config.METADATA_DIR,
        evidence_dir: Union[str, Path] = config.EVIDENCE_DIR
    ):
        self.metadata_dir = Path(metadata_dir)
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self._cached_meta: Dict[str, Dict[str, Any]] = {}

    def _get_camera_metadata(self, camera_id: str) -> Dict[str, Any]:
        """
        Loads and caches metadata for a camera.
        """
        if camera_id not in self._cached_meta:
            meta_path = self.metadata_dir / f"{camera_id}_tracks.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"Metadata file missing for {camera_id}: {meta_path}")
            self._cached_meta[camera_id] = load_json(meta_path)
        return self._cached_meta[camera_id]

    def generate_annotated_evidence(
        self,
        track_id: str,
        display_label: Optional[str] = None,
        max_frames_to_render: int = 150,
        box_color: Tuple[int, int, int] = (0, 255, 0),  # Bright Green (BGR)
    ) -> Dict[str, Any]:
        """
        Retrieves stored frame-level detections for the track, opens the source video,
        and renders green bounding boxes on every appearance frame.
        Outputs:
        - A playable MP4 evidence video in data/evidence/
        - Keyframe snapshots showing the person at early, middle, and late stages.
        """
        camera_id = "Camera_1" if track_id.startswith("Camera_1_") else "Camera_2"
        meta = self._get_camera_metadata(camera_id)

        if track_id not in meta:
            raise KeyError(f"Track '{track_id}' not found in {camera_id} metadata.")

        track_info = meta[track_id]
        video_path = track_info["source_video_path"]
        observations = track_info["frame_observations"]

        if not observations:
            raise ValueError(f"No frame observations recorded for track '{track_id}'.")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Source video could not be opened: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Output paths (render raw first, then transcode to web-compatible H.264)
        output_filename = f"{track_id}_evidence.mp4"
        output_path = self.evidence_dir / output_filename
        raw_output_path = self.evidence_dir / f"{track_id}_raw.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(raw_output_path), fourcc, fps, (width, height))

        # Map frame_idx -> observation
        obs_map = {obs["frame_idx"]: obs for obs in observations}
        frame_indices = sorted(obs_map.keys())

        # Subsample frames if track is extremely long (to prevent oversized video files)
        stride = 1
        if len(frame_indices) > max_frames_to_render:
            stride = max(1, len(frame_indices) // max_frames_to_render)
            selected_indices = frame_indices[::stride]
        else:
            selected_indices = frame_indices

        annotated_keyframes: List[Dict[str, Any]] = []
        keyframe_step = max(1, len(selected_indices) // 4)

        try:
            for idx_count, f_idx in enumerate(selected_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                obs = obs_map[f_idx]
                bbox = obs["bbox"]
                timestamp = obs["timestamp"]
                conf = obs["confidence"]

                # Badge label on green bounding box
                badge_text = f"{display_label or track_id} | {format_timestamp(timestamp)} | {conf:.2f}"
                annotated = draw_bounding_box(
                    image=frame,
                    bbox=bbox,
                    label=badge_text,
                    color=box_color,
                    thickness=3
                )

                # Write to output video
                out.write(annotated)

                # Save 3-4 keyframe snapshots for quick UI image display
                if idx_count % keyframe_step == 0 and len(annotated_keyframes) < 4:
                    keyframe_name = f"{track_id}_kf_{f_idx}.jpg"
                    keyframe_path = self.evidence_dir / keyframe_name
                    cv2.imwrite(str(keyframe_path), annotated)
                    annotated_keyframes.append({
                        "frame_idx": f_idx,
                        "timestamp": timestamp,
                        "image_path": str(keyframe_path).replace("\\", "/")
                    })

        finally:
            cap.release()
            out.release()

        # Transcode to standard H.264 with yuv420p and faststart for browser playback
        import subprocess
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", str(raw_output_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path)
        ]
        try:
            p = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if p.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                if raw_output_path.exists():
                    raw_output_path.unlink()
            else:
                if raw_output_path.exists():
                    raw_output_path.replace(output_path)
        except Exception:
            if raw_output_path.exists():
                raw_output_path.replace(output_path)

        return {
            "track_id": track_id,
            "camera_id": camera_id,
            "display_label": display_label or track_id,
            "start_time": track_info["start_time"],
            "end_time": track_info["end_time"],
            "duration": track_info["duration"],
            "total_frames_observed": track_info["total_frames_observed"],
            "evidence_video_path": str(output_path).replace("\\", "/"),
            "keyframes": annotated_keyframes,
            "source_video_path": video_path
        }
