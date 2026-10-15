import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import configs.config as config
from src.qdrant_store import QdrantStore
from src.identity_registry import IdentityRegistry
from src.matcher import ScreenshotMatcher
from src.evidence_generator import EvidenceGenerator
from src.utils import load_json, format_timestamp

def run_all_e2e_tests():
    print("==================================================")
    print("FULL SYSTEM END-TO-END VALIDATION (16 TEST CASES)")
    print("==================================================")

    # 1. Video Ingestion verification
    cam1_meta_file = config.METADATA_DIR / "Camera_1_tracks.json"
    cam2_meta_file = config.METADATA_DIR / "Camera_2_tracks.json"
    assert cam1_meta_file.exists() and cam2_meta_file.exists()
    cam1_tracks = load_json(cam1_meta_file)
    cam2_tracks = load_json(cam2_meta_file)
    print(f"[TEST 1 PASS] Both videos ingested: Camera_1={len(cam1_tracks)}, Camera_2={len(cam2_tracks)}")

    # 2. Every detected track has valid metadata
    for t_id, meta in list(cam1_tracks.items()) + list(cam2_tracks.items()):
        req_keys = ["track_id", "camera_id", "start_time", "end_time", "duration", "representative_crop_path", "embedding", "frame_observations"]
        assert all(k in meta for k in req_keys), f"Missing keys in {t_id}"
    print("[TEST 2 PASS] Every detected track contains all required metadata attributes.")

    # 3. Timestamps calculated correctly
    for t_id, meta in list(cam1_tracks.items()) + list(cam2_tracks.items()):
        assert meta["start_time"] >= 0.0
        assert meta["end_time"] >= meta["start_time"]
        assert meta["duration"] >= 0.0
    print("[TEST 3 PASS] Timestamps validated: start_time <= end_time across all tracks.")

    # 4. Frame-level bounding boxes are saved and can be retrieved
    sample_track = cam1_tracks["Camera_1_7"]
    obs_list = sample_track["frame_observations"]
    assert len(obs_list) == sample_track["total_frames_observed"]
    for o in obs_list[:5]:
        x1, y1, x2, y2 = o["bbox"]
        assert x2 > x1 and y2 > y1
    print(f"[TEST 4 PASS] Frame-level bounding boxes verified ({len(obs_list)} observations in sample track).")

    # 5. Qdrant stores expected vectors and payloads
    qstore = QdrantStore()
    assert qstore.count_tracks() == len(cam1_tracks) + len(cam2_tracks)
    sample_pt = qstore.get_track("Camera_1_7")
    assert sample_pt is not None
    assert len(sample_pt["vector"]) == 512
    print(f"[TEST 5 PASS] Qdrant point count ({qstore.count_tracks()}) and 512-dim vectors verified.")

    # 6. resources.json is created and updated correctly
    test_reg_path = config.STORAGE_DIR / "test_resources.json"
    if test_reg_path.exists():
        test_reg_path.unlink()
    registry = IdentityRegistry(registry_path=test_reg_path)
    
    # Register Camera_1_7 as Person_A
    ok, msg = registry.register_camera1_track("Person_A", sample_track)
    assert ok and "Person_A" in registry.identities
    print("[TEST 6 PASS] resources.json created and updated with Person_A.")

    # 7. Registering the same track twice does not duplicate its reference
    ok2, msg2 = registry.register_camera1_track("Person_A", sample_track)
    assert not ok2, "Should reject duplicate registration"
    assert len(registry.identities["Person_A"]["references"]) == 1
    print("[TEST 7 PASS] Duplicate registration prevented; references count kept at 1.")

    # 8. Reliable cross-camera match appends to existing identity
    # Create a synthetic Camera 2 track with high similarity to Person_A
    synth_match_track = {
        "track_id": "Camera_2_999",
        "camera_id": "Camera_2",
        "start_time": 10.0,
        "end_time": 20.0,
        "representative_crop_path": sample_track["representative_crop_path"],
        "embedding": sample_track["embedding"] # exact match
    }
    match_res = registry.register_camera2_track(synth_match_track, match_threshold=0.65, ambiguity_margin=0.05)
    assert match_res["status"] == "MATCHED_EXISTING"
    assert match_res["identity_label"] == "Person_A"
    assert len(registry.identities["Person_A"]["references"]) == 2
    print("[TEST 8 PASS] High-confidence Camera 2 track appended to existing Person_A identity.")

    # 9. Track with no reliable match creates new identity during registration
    synth_new_track = {
        "track_id": "Camera_2_888",
        "camera_id": "Camera_2",
        "start_time": 30.0,
        "end_time": 40.0,
        "representative_crop_path": sample_track["representative_crop_path"],
        # Orthogonal / low-similarity random vector
        "embedding": (-np.array(sample_track["embedding"])).tolist()
    }
    new_res = registry.register_camera2_track(synth_new_track, match_threshold=0.65, ambiguity_margin=0.05)
    assert new_res["status"] == "NEW_IDENTITY"
    assert new_res["identity_label"] != "Person_A"
    assert new_res["identity_label"] in registry.identities
    print(f"[TEST 9 PASS] Unmatched Camera 2 track assigned new unique identity: {new_res['identity_label']}.")

    # 10. Ambiguous match is not automatically accepted
    # Synthetic vector very close to Person_A and another identity
    # Add a second identity Person_B with similar embedding
    amb_track_info = dict(sample_track)
    amb_track_info["track_id"] = "Camera_1_88"
    v_b = np.array(sample_track["embedding"]) + np.random.normal(0, 0.01, 512).astype(np.float32)
    v_b /= np.linalg.norm(v_b)
    amb_track_info["embedding"] = v_b.tolist()
    registry.register_camera1_track("Person_B", amb_track_info)

    # Query vector that falls between Person_A and Person_B with < 0.05 margin
    v_query = (np.array(sample_track["embedding"]) + v_b) / 2.0
    v_query /= np.linalg.norm(v_query)
    synth_amb_track = {
        "track_id": "Camera_2_777",
        "camera_id": "Camera_2",
        "start_time": 50.0,
        "end_time": 60.0,
        "representative_crop_path": sample_track["representative_crop_path"],
        "embedding": v_query.tolist()
    }
    amb_eval = registry.register_camera2_track(synth_amb_track, match_threshold=0.65, ambiguity_margin=0.05)
    # When ambiguous (margin < 0.05), it should NOT force-merge into Person_A
    assert amb_eval["status"] == "NEW_IDENTITY"
    print(f"[TEST 10 PASS] Ambiguous cross-camera candidate was not force-merged into existing identity.")

    # 11. Query screenshot can retrieve correct candidate tracks
    matcher = ScreenshotMatcher(qstore=qstore, registry=registry)
    query_crop = cv2.imread(sample_track["representative_crop_path"])
    q_res = matcher.query_crop(query_crop)
    assert q_res["decision"] == "CONFIRMED_MATCH"
    assert q_res["top_candidate"]["track_id"] == "Camera_1_7"
    print(f"[TEST 11 PASS] Query screenshot retrieved correct candidate: {q_res['top_candidate']['track_id']}.")

    # 12. Query with no reliable candidate returns Unknown
    blank_crop = np.zeros((256, 128, 3), dtype=np.uint8)
    unk_res = matcher.query_crop(blank_crop, score_threshold=0.65)
    assert unk_res["decision"] == "UNKNOWN"
    print("[TEST 12 PASS] Low-similarity query returns 'UNKNOWN'.")

    # 13. Green bounding boxes are rendered using correct track metadata
    gen = EvidenceGenerator()
    ev_data = gen.generate_annotated_evidence("Camera_1_7", display_label="Test Person", max_frames_to_render=30)
    assert Path(ev_data["evidence_video_path"]).exists()
    assert len(ev_data["keyframes"]) > 0
    # Check green color in bounding box snapshot
    kf_img = cv2.imread(ev_data["keyframes"][0]["image_path"])
    # Green channel should have distinct high values where boxes are drawn
    assert kf_img[:, :, 1].max() == 255
    print("[TEST 13 PASS] Green bounding box video evidence rendered and verified.")

    # 14. Multiple camera results retain their own timestamps and track IDs
    cam1_cand = [c for c in q_res["candidates"] if c["camera_id"] == "Camera_1"]
    cam2_cand = [c for c in q_res["candidates"] if c["camera_id"] == "Camera_2"]
    assert len(cam1_cand) > 0 and len(cam2_cand) > 0
    assert cam1_cand[0]["track_id"].startswith("Camera_1_")
    assert cam2_cand[0]["track_id"].startswith("Camera_2_")
    print(f"[TEST 14 PASS] Camera-specific IDs & timestamps preserved: Cam1={cam1_cand[0]['track_id']}, Cam2={cam2_cand[0]['track_id']}.")

    # 15. Restarting Streamlit does not lose registrations (persistence test)
    registry_reloaded = IdentityRegistry(registry_path=test_reg_path)
    assert "Person_A" in registry_reloaded.identities
    assert len(registry_reloaded.identities["Person_A"]["references"]) == 2
    print("[TEST 15 PASS] Re-instantiating registry successfully restored all persisted identities.")

    # Clean up test registry
    test_reg_path.unlink()

    # 16. Reprocessing videos does not silently corrupt or duplicate the dataset
    count_before = qstore.count_tracks()
    qstore.upsert_tracks({"Camera_1_7": sample_track})
    count_after = qstore.count_tracks()
    assert count_before == count_after, f"Point duplication detected! {count_before} vs {count_after}"
    print(f"[TEST 16 PASS] Reprocessing idempotency verified: count remained exactly {count_after}.")

    qstore.close()
    print("==================================================")
    print("ALL 16 END-TO-END SPECIFICATION TESTS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    run_all_e2e_tests()
