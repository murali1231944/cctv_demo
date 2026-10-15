import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import configs.config as config
from src.video_processor import VideoProcessor
from src.utils import load_json

def test_milestone2():
    print("========================================")
    print("RUNNING MILESTONE 2 VERIFICATION SUITE")
    print("========================================")

    camera_id = "Camera_1"
    video_path = config.CAMERA_VIDEOS[camera_id]
    assert video_path.exists(), f"Video file not found at {video_path}"

    print(f"Initializing VideoProcessor for {camera_id}...")
    processor = VideoProcessor(
        camera_id=camera_id,
        video_path=video_path,
        conf_threshold=0.35,
        min_track_frames=5
    )

    # Process first 200 frames to validate ingestion mechanics cleanly
    test_frames = 200
    print(f"Ingesting test sequence ({test_frames} frames)...")

    def progress(f, total, fps, num_tracks):
        if f % 50 == 0 or f == total:
            print(f"  Processed {f}/{total} frames ({fps:.1f} fps) - Found {num_tracks} tracks so far")

    tracks = processor.process_video(max_frames=test_frames, progress_callback=progress)

    print(f"[PASS] Ingestion executed. Valid tracks detected: {len(tracks)}")
    assert len(tracks) > 0, "Expected at least 1 valid track in 200 frames of Camera_1"

    # Verify track properties
    for t_id, meta in tracks.items():
        assert t_id.startswith("Camera_1_"), f"Track ID {t_id} does not follow Camera_1_X format"
        assert meta["camera_id"] == "Camera_1"
        assert meta["start_time"] >= 0.0
        assert meta["end_time"] >= meta["start_time"]
        assert meta["total_frames_observed"] >= 5
        assert len(meta["embedding"]) == config.EMBEDDING_DIM
        
        # Verify L2 normalization of stored embedding
        emb = np.array(meta["embedding"], dtype=np.float32)
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-3, f"Embedding not normalized: {norm}"

        # Check representative crop exists on disk
        crop_path = Path(meta["representative_crop_path"])
        assert crop_path.exists(), f"Crop image missing at {crop_path}"
        img = cv2.imread(str(crop_path))
        assert img is not None and img.size > 0, f"Cannot read crop image at {crop_path}"

        # Verify frame-level bounding boxes and timestamps
        obs = meta["frame_observations"]
        assert len(obs) == meta["total_frames_observed"]
        for o in obs:
            assert "frame_idx" in o
            assert "timestamp" in o
            assert "bbox" in o
            x1, y1, x2, y2 = o["bbox"]
            assert x2 > x1 and y2 > y1, f"Invalid bbox {o['bbox']}"

        print(f"  [Verified Track] {t_id}: {meta['total_frames_observed']} frames, time: {meta['start_time']}s -> {meta['end_time']}s, crop: {crop_path.name}")

    # Verify persisted metadata file
    meta_file = config.METADATA_DIR / f"{camera_id}_tracks.json"
    assert meta_file.exists(), f"Metadata file not found at {meta_file}"
    loaded_meta = load_json(meta_file)
    assert len(loaded_meta) == len(tracks), "Metadata file count mismatch"
    print(f"[PASS] Metadata successfully written and verified at {meta_file.name}")

    print("========================================")
    print("ALL MILESTONE 2 VERIFICATIONS PASSED!")
    print("========================================")

if __name__ == "__main__":
    test_milestone2()
