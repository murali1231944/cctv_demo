import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import configs.config as config
from src.video_processor import VideoProcessor
from src.qdrant_store import QdrantStore
from src.embeddings import OSNetFeatureExtractor

def ingest_camera(camera_id: str, video_path: Path, extractor: OSNetFeatureExtractor, qstore: QdrantStore):
    print(f"\n==================================================")
    print(f"STARTING FULL INGESTION FOR {camera_id}")
    print(f"Video Path: {video_path}")
    print(f"==================================================")

    t0 = time.time()
    processor = VideoProcessor(
        camera_id=camera_id,
        video_path=video_path,
        extractor=extractor,
        conf_threshold=config.DETECTION_CONF_THRESHOLD,
        min_track_frames=config.MIN_TRACK_FRAMES
    )

    total_frames = processor.total_frames
    print(f"Total video frames to process: {total_frames} (fps={processor.fps:.2f}, duration={processor.video_info['duration_seconds']}s)")

    last_report_time = time.time()

    def progress_callback(curr_frame, total_f, fps, num_tracks):
        nonlocal last_report_time
        now = time.time()
        if now - last_report_time >= 5.0 or curr_frame == total_f:
            pct = (curr_frame / total_f) * 100.0
            elapsed = now - t0
            eta = (elapsed / max(curr_frame, 1)) * (total_f - curr_frame)
            print(f"[{camera_id}] Frame {curr_frame}/{total_f} ({pct:5.1f}%) | Speed: {fps:4.1f} fps | Active Tracks: {num_tracks:2d} | Elapsed: {elapsed:5.1f}s | ETA: {eta:5.1f}s")
            last_report_time = now

    tracks = processor.process_video(progress_callback=progress_callback)
    dt = time.time() - t0

    print(f"[{camera_id}] Completed video processing in {dt:.1f}s ({total_frames / dt:.1f} fps average)")
    print(f"[{camera_id}] Total valid person tracks extracted: {len(tracks)}")

    # Store in Qdrant
    upserted = qstore.upsert_tracks(tracks)
    print(f"[{camera_id}] Upserted {upserted} tracks into Qdrant collection '{config.QDRANT_COLLECTION_NAME}'")

    for t_id, meta in list(tracks.items())[:10]:
        print(f"  Track {t_id:12s} | Duration: {meta['start_time']:6.2f}s -> {meta['end_time']:6.2f}s ({meta['duration']:5.2f}s) | Frames: {meta['total_frames_observed']:3d} | Crop: {Path(meta['representative_crop_path']).name}")
    if len(tracks) > 10:
        print(f"  ... and {len(tracks) - 10} more tracks.")

    return tracks

def main():
    total_start = time.time()
    print("==================================================")
    print("FULL MULTI-CAMERA VIDEO INGESTION PIPELINE")
    print(f"Device: {config.DEVICE}")
    print(f"Qdrant Storage Path: {config.QDRANT_PATH}")
    print("==================================================")

    # Initialize shared models and stores
    extractor = OSNetFeatureExtractor()
    qstore = QdrantStore()

    cam1_tracks = ingest_camera("Camera_1", config.CAMERA_VIDEOS["Camera_1"], extractor, qstore)
    cam2_tracks = ingest_camera("Camera_2", config.CAMERA_VIDEOS["Camera_2"], extractor, qstore)

    qstore.close()

    total_time = time.time() - total_start
    print("\n==================================================")
    print("INGESTION SUMMARY")
    print("==================================================")
    print(f"Camera_1 Valid Tracks: {len(cam1_tracks)}")
    print(f"Camera_2 Valid Tracks: {len(cam2_tracks)}")
    print(f"Total Tracks Stored:   {len(cam1_tracks) + len(cam2_tracks)}")
    print(f"Total Processing Time: {total_time:.1f}s ({total_time / 60:.2f} min)")
    print("==================================================")

if __name__ == "__main__":
    main()
