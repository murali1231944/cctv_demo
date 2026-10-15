import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import configs.config as config
from src.qdrant_store import QdrantStore
from src.utils import load_json

def test_milestone3():
    print("========================================")
    print("RUNNING MILESTONE 3 VERIFICATION SUITE")
    print("========================================")

    # Load tracks generated from Milestone 2 test
    meta_file = config.METADATA_DIR / "Camera_1_tracks.json"
    assert meta_file.exists(), f"Metadata file not found at {meta_file}. Run Milestone 2 first."

    tracks = load_json(meta_file)
    assert len(tracks) > 0, "No tracks found in metadata"
    print(f"Loaded {len(tracks)} test tracks from {meta_file.name}")

    # Use a separate test directory for isolated testing
    test_qdrant_dir = config.STORAGE_DIR / "test_qdrant_m3"
    qstore = QdrantStore(storage_path=test_qdrant_dir, collection_name="test_reid_m3")

    try:
        # 1. Upsert tracks into Qdrant
        upserted = qstore.upsert_tracks(tracks)
        print(f"[PASS] 1. Upserted {upserted} tracks into Qdrant.")
        assert upserted == len(tracks)

        count = qstore.count_tracks()
        assert count == len(tracks), f"Expected {len(tracks)} points, got {count}"
        print(f"[PASS] 2. Collection count verified: {count} points.")

        # 2. Test Idempotency (re-inserting same tracks must not duplicate points)
        upserted_again = qstore.upsert_tracks(tracks)
        count_after = qstore.count_tracks()
        assert count_after == count, f"Duplicate points created! {count_after} != {count}"
        print(f"[PASS] 3. Idempotency verified: re-upserting kept count at {count_after}.")

        # 3. Test exact vector retrieval and search
        first_track_id = list(tracks.keys())[0]
        first_track = tracks[first_track_id]
        query_vec = first_track["embedding"]

        search_results = qstore.search_similar(query_vector=query_vec, limit=3)
        assert len(search_results) > 0, "No search results returned"
        best_hit = search_results[0]
        print(f"Top search result: {best_hit['payload']['track_id']}, Score: {best_hit['score']:.4f}")
        assert best_hit["payload"]["track_id"] == first_track_id, f"Expected {first_track_id}, got {best_hit['payload']['track_id']}"
        assert abs(best_hit["score"] - 1.0) < 1e-3, f"Score for self query should be ~1.0, got {best_hit['score']}"
        print("[PASS] 4. Query vector retrieval and score verified.")

        # 4. Test track retrieval by ID
        retrieved = qstore.get_track(first_track_id)
        assert retrieved is not None, f"Could not retrieve track {first_track_id}"
        assert retrieved["payload"]["camera_id"] == "Camera_1"
        assert retrieved["payload"]["representative_crop_path"] == first_track["representative_crop_path"]
        print(f"[PASS] 5. Track retrieval by ID verified: {retrieved['payload']['track_id']}.")

        # 5. Test camera filtering
        cam1_tracks = qstore.get_all_tracks(camera_id="Camera_1")
        assert len(cam1_tracks) == count
        cam2_tracks = qstore.get_all_tracks(camera_id="Camera_2")
        assert len(cam2_tracks) == 0
        print("[PASS] 6. Camera filtering verified.")

    finally:
        qstore.close()
        import shutil
        if test_qdrant_dir.exists():
            shutil.rmtree(test_qdrant_dir)

    print("========================================")
    print("ALL MILESTONE 3 VERIFICATIONS PASSED!")
    print("========================================")

if __name__ == "__main__":
    test_milestone3()
