import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import configs.config as config
from src.detector import PersonDetector
from src.tracker import MultiPersonTracker
from src.embeddings import OSNetFeatureExtractor
from src.utils import get_video_info, compute_timestamp, save_json_atomic, load_json

def test_milestone1():
    print("========================================")
    print("RUNNING MILESTONE 1 VERIFICATION SUITE")
    print("========================================")

    # 1. Check directories
    required_dirs = [
        config.INPUT_VIDEOS_DIR,
        config.CROPS_DIR,
        config.EVIDENCE_DIR,
        config.STORAGE_DIR,
        config.QDRANT_PATH,
        config.METADATA_DIR,
        config.MODELS_DIR
    ]
    for d in required_dirs:
        assert d.exists(), f"Missing directory: {d}"
    print("[PASS] 1. All project directories exist.")

    # 2. Check model weights
    assert config.YOLO_MODEL_PATH.exists(), f"Missing YOLO weights at {config.YOLO_MODEL_PATH}"
    assert config.OSNET_MODEL_PATH.exists(), f"Missing OSNet weights at {config.OSNET_MODEL_PATH}"
    print(f"[PASS] 2. Model weights found: YOLO ({config.YOLO_MODEL_PATH.stat().st_size} bytes), OSNet ({config.OSNET_MODEL_PATH.stat().st_size} bytes).")

    # 3. Check input videos
    for cam_name, vpath in config.CAMERA_VIDEOS.items():
        assert vpath.exists(), f"Missing video for {cam_name} at {vpath}"
        info = get_video_info(vpath)
        assert info["fps"] > 0, f"Invalid fps for {cam_name}"
        assert info["total_frames"] > 0, f"Invalid total_frames for {cam_name}"
        assert info["duration_seconds"] > 0, f"Invalid duration for {cam_name}"
        print(f"[PASS] 3. {cam_name} video verified: {info['width']}x{info['height']}, {info['fps']} fps, {info['total_frames']} frames, {info['duration_seconds']}s.")

    # 4. Test YOLO Detector
    detector = PersonDetector(conf_threshold=0.3)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = detector.detect(dummy_frame)
    assert isinstance(dets, list), "Detector must return a list"
    print("[PASS] 4. YOLO PersonDetector initialized and executed safely.")

    # 5. Test MultiPersonTracker
    tracker = MultiPersonTracker(camera_id="Camera_1", fps=30.0)
    tracker.reset()
    tracks = tracker.update(dummy_frame, frame_idx=0)
    assert isinstance(tracks, list), "Tracker must return a list"
    print("[PASS] 5. MultiPersonTracker initialized and reset cleanly.")

    # 6. Test OSNet Feature Extractor
    extractor = OSNetFeatureExtractor()
    test_crop = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
    feat = extractor.extract_features(test_crop)
    assert feat.shape == (config.EMBEDDING_DIM,), f"Feature shape mismatch: {feat.shape} vs ({config.EMBEDDING_DIM},)"
    norm = np.linalg.norm(feat)
    assert abs(norm - 1.0) < 1e-4, f"Feature is not L2 normalized: norm={norm}"
    
    # Test batch aggregation
    test_crops = [np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8) for _ in range(3)]
    agg_feat = extractor.aggregate_embeddings(test_crops)
    assert agg_feat.shape == (config.EMBEDDING_DIM,), f"Aggregated feature shape mismatch: {agg_feat.shape}"
    agg_norm = np.linalg.norm(agg_feat)
    assert abs(agg_norm - 1.0) < 1e-4, f"Aggregated feature is not L2 normalized: norm={agg_norm}"
    print(f"[PASS] 6. OSNetFeatureExtractor verified: output shape {feat.shape}, L2-normalized (norm={norm:.6f}).")

    # 7. Test atomic JSON operations
    test_json_path = config.STORAGE_DIR / "test_atomic.json"
    test_data = {"test_key": "test_val", "numbers": [1, 2, 3]}
    save_json_atomic(test_data, test_json_path)
    loaded_data = load_json(test_json_path)
    assert loaded_data == test_data, "Atomic JSON save/load mismatch"
    test_json_path.unlink()
    print("[PASS] 7. Atomic JSON storage verified.")

    print("========================================")
    print("ALL MILESTONE 1 VERIFICATIONS PASSED!")
    print("========================================")

if __name__ == "__main__":
    test_milestone1()
