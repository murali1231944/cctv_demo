import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import configs.config as config
from src.qdrant_store import QdrantStore
from src.utils import load_json

def test_milestone4():
    print("========================================")
    print("RUNNING MILESTONE 4 VERIFICATION SUITE")
    print("========================================")

    # 1. Verify metadata files exist for both cameras
    meta_cam1 = config.METADATA_DIR / "Camera_1_tracks.json"
    meta_cam2 = config.METADATA_DIR / "Camera_2_tracks.json"

    assert meta_cam1.exists(), f"Missing metadata file: {meta_cam1}"
    assert meta_cam2.exists(), f"Missing metadata file: {meta_cam2}"

    cam1_tracks = load_json(meta_cam1)
    cam2_tracks = load_json(meta_cam2)

    num_cam1 = len(cam1_tracks)
    num_cam2 = len(cam2_tracks)
    total_expected = num_cam1 + num_cam2

    print(f"[PASS] 1. Metadata loaded: Camera_1={num_cam1} tracks, Camera_2={num_cam2} tracks (Total: {total_expected}).")
    assert num_cam1 >= 50, f"Expected at least 50 tracks in Camera_1, found {num_cam1}"
    assert num_cam2 >= 50, f"Expected at least 50 tracks in Camera_2, found {num_cam2}"

    # 2. Check representative crops exist for every track
    for t_id, meta in list(cam1_tracks.items()) + list(cam2_tracks.items()):
        crop_path = Path(meta["representative_crop_path"])
        assert crop_path.exists(), f"Missing crop image for {t_id} at {crop_path}"
        assert meta["start_time"] <= meta["end_time"], f"Invalid timestamps for {t_id}"
        assert len(meta["frame_observations"]) == meta["total_frames_observed"]
    print(f"[PASS] 2. All {total_expected} representative crops and frame observations validated on disk.")

    # 3. Check Qdrant collection
    qstore = QdrantStore()
    try:
        total_in_qdrant = qstore.count_tracks()
        cam1_in_qdrant = qstore.count_tracks(camera_id="Camera_1")
        cam2_in_qdrant = qstore.count_tracks(camera_id="Camera_2")

        print(f"Qdrant collection count: total={total_in_qdrant}, Camera_1={cam1_in_qdrant}, Camera_2={cam2_in_qdrant}")

        assert total_in_qdrant == total_expected, f"Qdrant point count mismatch: {total_in_qdrant} vs {total_expected}"
        assert cam1_in_qdrant == num_cam1, f"Camera_1 count mismatch in Qdrant: {cam1_in_qdrant} vs {num_cam1}"
        assert cam2_in_qdrant == num_cam2, f"Camera_2 count mismatch in Qdrant: {cam2_in_qdrant} vs {num_cam2}"
        print("[PASS] 3. Qdrant point count exactly matches metadata for both cameras.")

        # 4. Verify retrieving tracks from Qdrant
        sample_id = list(cam1_tracks.keys())[0]
        track = qstore.get_track(sample_id)
        assert track is not None, f"Failed to retrieve sample track {sample_id}"
        assert track["payload"]["camera_id"] == "Camera_1"
        assert len(track["vector"]) == config.EMBEDDING_DIM
        print(f"[PASS] 4. Qdrant track retrieval verified for sample: {sample_id}.")

    finally:
        qstore.close()

    print("========================================")
    print("ALL MILESTONE 4 VERIFICATIONS PASSED!")
    print("========================================")

if __name__ == "__main__":
    test_milestone4()
