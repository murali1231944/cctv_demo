import os
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
INPUT_VIDEOS_DIR = DATA_DIR / "input_videos"
CROPS_DIR = DATA_DIR / "crops"
EVIDENCE_DIR = DATA_DIR / "evidence"

# Storage directories
STORAGE_DIR = BASE_DIR / "storage"
QDRANT_PATH = STORAGE_DIR / "qdrant"
METADATA_DIR = STORAGE_DIR / "metadata"
RESOURCES_JSON_PATH = STORAGE_DIR / "resources.json"

# Models directory
MODELS_DIR = BASE_DIR / "models"
YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"
OSNET_MODEL_PATH = MODELS_DIR / "osnet_x1_0_market1501.pth"

# Ensure runtime directories exist
for p in [INPUT_VIDEOS_DIR, CROPS_DIR, EVIDENCE_DIR, QDRANT_PATH, METADATA_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Device configuration (GPU if available, otherwise CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Camera input video paths
CAMERA_VIDEOS = {
    "Camera_1": INPUT_VIDEOS_DIR / "Camera_1.mp4",
    "Camera_2": INPUT_VIDEOS_DIR / "Camera_2.mp4",
}

# Detection and Tracking parameters
DETECTION_CONF_THRESHOLD = 0.40
TRACKER_TYPE = "bytetrack.yaml"
MIN_TRACK_FRAMES = 8          # Require at least 8 observations to form a reliable track
MIN_BBOX_WIDTH = 30
MIN_BBOX_HEIGHT = 60
MIN_BBOX_AREA = 2500

# OSNet Feature Extraction parameters
EMBEDDING_DIM = 512
REID_INPUT_HEIGHT = 256
REID_INPUT_WIDTH = 128
MAX_REPRESENTATIVE_CROPS = 5   # Extract embeddings from up to 5 best quality frames per track and average

# Qdrant Vector Store configuration
QDRANT_COLLECTION_NAME = "person_reid_tracks"

# Cross-Camera and Query Matching Thresholds
DEFAULT_MATCH_THRESHOLD = 0.65 # Cosine similarity threshold for confirmed match
AMBIGUITY_THRESHOLD = 0.05     # Top1 vs Top2 margin to prevent accepting ambiguous candidates
