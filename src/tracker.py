from typing import List, Dict, Any, Union, Optional
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import configs.config as config
from src.utils import compute_timestamp

class MultiPersonTracker:
    """
    Multi-person tracker wrapping YOLO ByteTrack.
    Produces camera-specific track identifiers (e.g. Camera_1_3, Camera_2_5).
    """
    def __init__(
        self,
        camera_id: str,
        fps: float = 30.0,
        model_path: Union[str, Path] = config.YOLO_MODEL_PATH,
        conf_threshold: float = config.DETECTION_CONF_THRESHOLD,
        tracker_type: str = config.TRACKER_TYPE,
        device: str = config.DEVICE
    ):
        self.camera_id = camera_id
        self.fps = fps
        self.model_path = str(model_path)
        self.conf_threshold = conf_threshold
        self.tracker_type = tracker_type
        self.device = device
        self.model = YOLO(self.model_path)
        self.reset()

    def reset(self) -> None:
        """
        Resets the internal tracker state for a new video stream.
        """
        # Re-initialize tracker predictor in ultralytics to clear previous video states
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            if hasattr(self.model.predictor, "trackers"):
                self.model.predictor.trackers = []

    def update(self, frame: np.ndarray, frame_idx: int) -> List[Dict[str, Any]]:
        """
        Processes a single frame and returns active person tracks.
        """
        if frame is None or frame.size == 0:
            return []

        timestamp = compute_timestamp(frame_idx, self.fps)

        # Track humans (class 0)
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_type,
            classes=[0],
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )

        tracked_objects = []
        if not results:
            return tracked_objects

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return tracked_objects

        # Check if tracking IDs are available
        track_ids = boxes.id
        if track_ids is None:
            return tracked_objects

        track_ids_list = track_ids.int().cpu().numpy().tolist()
        xyxy_list = boxes.xyxy.cpu().numpy()
        conf_list = boxes.conf.cpu().numpy()

        for raw_id, xyxy, conf in zip(track_ids_list, xyxy_list, conf_list):
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            w = x2 - x1
            h = y2 - y1

            # Filter out tiny noise boxes
            if w < config.MIN_BBOX_WIDTH or h < config.MIN_BBOX_HEIGHT or (w * h) < config.MIN_BBOX_AREA:
                continue

            camera_track_id = f"{self.camera_id}_{raw_id}"

            tracked_objects.append({
                "camera_id": self.camera_id,
                "raw_track_id": int(raw_id),
                "track_id": camera_track_id,
                "bbox": [x1, y1, x2, y2],
                "confidence": round(float(conf), 4),
                "frame_idx": int(frame_idx),
                "timestamp": timestamp
            })

        return tracked_objects
