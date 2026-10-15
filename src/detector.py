from typing import List, Dict, Any, Union
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import configs.config as config

class PersonDetector:
    """
    YOLO-based person detector. Filters for human detections (class 0) above confidence threshold.
    """
    def __init__(
        self,
        model_path: Union[str, Path] = config.YOLO_MODEL_PATH,
        conf_threshold: float = config.DETECTION_CONF_THRESHOLD,
        device: str = config.DEVICE
    ):
        self.model_path = str(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = YOLO(self.model_path)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects persons in a single image or video frame.
        Returns a list of dicts with keys: bbox [x1, y1, x2, y2], confidence, class_id.
        """
        if frame is None or frame.size == 0:
            return []

        # Run inference filtering for class 0 (person)
        results = self.model.predict(
            source=frame,
            classes=[0],
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )

        detections = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())

            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            w = x2 - x1
            h = y2 - y1

            # Filter out tiny spurious noise detections
            if w < config.MIN_BBOX_WIDTH or h < config.MIN_BBOX_HEIGHT or (w * h) < config.MIN_BBOX_AREA:
                continue

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 4),
                "class_id": cls_id
            })

        return detections
