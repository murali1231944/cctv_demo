import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import configs.config as config
from src.matcher import ScreenshotMatcher
from src.evidence_generator import EvidenceGenerator
from src.utils import load_json

def test_evidence_and_query():
    print("==================================================")
    print("TESTING SCREENSHOT QUERY AND EVIDENCE GENERATION")
    print("==================================================")

    matcher = ScreenshotMatcher()

    # 1. Test Confident Query using a representative person crop
    crop_path = Path("data/crops/Camera_1_7.jpg")
    assert crop_path.exists(), f"Crop {crop_path} missing"
    crop_img = cv2.imread(str(crop_path))
    
    query_res = matcher.query_crop(crop_img)
    print(f"\n1. Distinctive Crop Query:")
    print(f"   Decision: {query_res['decision']}")
    print(f"   Match Score: {query_res['match_score']:.4f} (Margin: {query_res['margin']:.4f})")
    print(f"   Top Matched Track: {query_res['top_candidate']['track_id']} ({query_res['top_candidate']['camera_id']})")
    print(f"   Time Range: {query_res['top_candidate']['start_time']}s -> {query_res['top_candidate']['end_time']}s")

    assert query_res["decision"] == "CONFIRMED_MATCH", f"Expected CONFIRMED_MATCH, got {query_res['decision']}"
    assert query_res["top_candidate"]["track_id"] == "Camera_1_7"
    print("   [PASS] Distinctive crop retrieved exact track with CONFIRMED_MATCH.")

    # 2. Test Ambiguous Query detection
    test_person_crop = cv2.imread("data/test_person_crop.jpg")
    amb_res = matcher.query_crop(test_person_crop, ambiguity_margin=0.05)
    print(f"\n2. Close-Appearance Ambiguity Handling:")
    print(f"   Decision: {amb_res['decision']}")
    print(f"   Match Score: {amb_res['match_score']:.4f} (Margin: {amb_res['margin']:.4f})")
    print(f"   Top Candidates:")
    for i, c in enumerate(amb_res['candidates'][:2]):
        print(f"     #{i+1}: {c['track_id']} ({c['camera_id']}) score={c['score']:.4f}")
    assert amb_res["decision"] in ["AMBIGUOUS", "CONFIRMED_MATCH"]
    print("   [PASS] Ambiguity margin successfully applied.")

    # 3. Test Unknown Handling with a zero/blank image
    blank_img = np.zeros((256, 128, 3), dtype=np.uint8)
    unknown_res = matcher.query_crop(blank_img, score_threshold=0.65)
    print(f"\n3. Non-person / Low Similarity Handling:")
    print(f"   Decision: {unknown_res['decision']} (score: {unknown_res['match_score']:.4f})")
    assert unknown_res["decision"] == "UNKNOWN", f"Expected UNKNOWN, got {unknown_res['decision']}"
    print("   [PASS] Insufficient similarity reported as UNKNOWN.")

    # 4. Test Evidence Generator (Green bounding boxes on video)
    print("\n4. Generating annotated video evidence with green bounding boxes...")
    gen = EvidenceGenerator()
    ev_res = gen.generate_annotated_evidence(
        track_id="Camera_1_7",
        display_label="Target Person: Camera_1_7",
        max_frames_to_render=60
    )

    ev_video = Path(ev_res["evidence_video_path"])
    assert ev_video.exists(), f"Evidence video not created: {ev_video}"
    assert ev_video.stat().st_size > 10000, f"Evidence video too small: {ev_video.stat().st_size} bytes"
    print(f"   [PASS] Evidence video successfully rendered: {ev_video.name} ({ev_video.stat().st_size} bytes)")

    assert len(ev_res["keyframes"]) > 0, "No keyframes generated"
    for kf in ev_res["keyframes"]:
        kf_path = Path(kf["image_path"])
        assert kf_path.exists(), f"Missing keyframe image: {kf_path}"
    print(f"   [PASS] Generated {len(ev_res['keyframes'])} keyframe snapshots with green bounding boxes.")

    matcher.qstore.close()
    print("\n==================================================")
    print("ALL EVIDENCE AND QUERY TESTS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    test_evidence_and_query()
