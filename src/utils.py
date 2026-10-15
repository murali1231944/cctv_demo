import os
import json
import tempfile
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

def get_video_info(video_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extracts video metadata: fps, total_frames, width, height, and duration_seconds.
    """
    video_path = str(video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # Safe fallback if metadata is corrupted

    duration = total_frames / fps if total_frames > 0 else 0.0

    return {
        "video_path": video_path,
        "fps": round(fps, 3),
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_seconds": round(duration, 2),
    }


def compute_timestamp(frame_idx: int, fps: float) -> float:
    """
    Computes timestamp in seconds for a given frame index and fps.
    """
    if fps <= 0:
        return 0.0
    return round(float(frame_idx) / float(fps), 3)


def format_timestamp(seconds: float) -> str:
    """
    Formats seconds into MM:SS.sss display format.
    """
    minutes = int(seconds // 60)
    rem_seconds = seconds % 60
    return f"{minutes:02d}:{rem_seconds:05.2f}"


def crop_person_image(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin: float = 0.02
) -> Optional[np.ndarray]:
    """
    Crops the detected person bounding box [x1, y1, x2, y2] from the frame with bounds clamping.
    """
    if frame is None or len(frame.shape) < 2:
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    # Apply slight margin expansion
    box_w = x2 - x1
    box_h = y2 - y1
    pad_w = int(box_w * margin)
    pad_h = int(box_h * margin)

    x1 = max(0, int(x1 - pad_w))
    y1 = max(0, int(y1 - pad_h))
    x2 = min(w, int(x2 + pad_w))
    y2 = min(h, int(y2 + pad_h))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2].copy()
    return crop


def save_json_atomic(data: Any, file_path: Union[str, Path]) -> None:
    """
    Saves data to a JSON file atomically using a temporary file to avoid partial write corruptions.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(file_path.parent),
        delete=False,
        suffix=".tmp"
    )
    temp_path = Path(temp_file.name)
    try:
        json.dump(data, temp_file, indent=2, ensure_ascii=False)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_file.close()
        temp_path.replace(file_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise e


def load_json(file_path: Union[str, Path], default: Any = None) -> Any:
    """
    Loads JSON data safely from file. Returns default if file does not exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return default

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def draw_bounding_box(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: Optional[str] = None,
    color: Tuple[int, int, int] = (0, 255, 0),  # Default: Vibrant Green (BGR)
    thickness: int = 2
) -> np.ndarray:
    """
    Draws a bounding box and optional header badge label on an image.
    """
    annotated = image.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Draw primary green bounding box
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        text_thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

        # Label background tag
        label_y1 = max(0, y1 - th - baseline - 4)
        label_y2 = y1
        label_x2 = min(annotated.shape[1], x1 + tw + 6)
        cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), color, -1)
        # Label text in dark contrast
        cv2.putText(
            annotated,
            label,
            (x1 + 3, label_y2 - baseline - 2),
            font,
            font_scale,
            (0, 0, 0),
            text_thickness,
            cv2.LINE_AA
        )

    return annotated
