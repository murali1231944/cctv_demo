from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import cv2
import numpy as np

import configs.config as config
from src.detector import PersonDetector
from src.embeddings import OSNetFeatureExtractor
from src.qdrant_store import QdrantStore
from src.identity_registry import IdentityRegistry
from src.utils import crop_person_image

class ScreenshotMatcher:
    """
    Handles screenshot-based person re-identification queries:
    - Detects and crops person candidates from uploaded screenshots (or uses pre-cropped images)
    - Extracts OSNet 512-dim appearance embedding
    - Queries Qdrant vector database for candidate tracks across cameras
    - Applies validated thresholding and ambiguity checks
    - Maps retrieved tracks to registered identities in resources.json
    - Reports 'Unknown' for uncertain or low-similarity queries
    """
    def __init__(
        self,
        qstore: Optional[QdrantStore] = None,
        extractor: Optional[OSNetFeatureExtractor] = None,
        detector: Optional[PersonDetector] = None,
        registry: Optional[IdentityRegistry] = None
    ):
        self.qstore = qstore or QdrantStore()
        self.extractor = extractor or OSNetFeatureExtractor()
        self.detector = detector or PersonDetector()
        self.registry = registry or IdentityRegistry()

    def process_screenshot_crops(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects people in the uploaded screenshot.
        If people are detected, returns a list of crops and their bboxes.
        If no separate people are detected (e.g. image is already a cropped person), returns the image itself.
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        # Run person detection
        detections = self.detector.detect(image)

        # If detections are found and not just a single box filling the whole frame
        candidate_crops = []
        for i, det in enumerate(detections):
            bbox = det["bbox"]
            box_w = bbox[2] - bbox[0]
            box_h = bbox[3] - bbox[1]

            # If detection covers > 90% of image, it's essentially the whole image
            if box_w * box_h >= 0.90 * (w * h):
                continue

            crop = crop_person_image(image, bbox)
            if crop is not None:
                candidate_crops.append({
                    "crop_id": i + 1,
                    "bbox": bbox,
                    "confidence": det["confidence"],
                    "crop_image": crop
                })

        # If no specific people detected or image is already a crop (tall aspect ratio: h > w)
        if not candidate_crops:
            candidate_crops.append({
                "crop_id": 1,
                "bbox": [0, 0, w, h],
                "confidence": 1.0,
                "crop_image": image
            })

        return candidate_crops

    def query_crop(
        self,
        crop_image: np.ndarray,
        top_k: int = 5,
        score_threshold: float = config.DEFAULT_MATCH_THRESHOLD,
        ambiguity_margin: float = config.AMBIGUITY_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Executes query for a single person crop against Qdrant.
        """
        # 1. Extract 512-dim embedding
        query_vector = self.extractor.extract_features(crop_image)

        # 2. Query Qdrant for top candidates
        qdrant_hits = self.qstore.search_similar(query_vector=query_vector, limit=top_k)

        if not qdrant_hits:
            return {
                "decision": "UNKNOWN",
                "matched_track_id": None,
                "camera_id": None,
                "identity_label": "Unknown",
                "match_score": 0.0,
                "margin": 0.0,
                "candidates": [],
                "message": "No candidates found in Qdrant database."
            }

        # 3. Analyze top candidates
        top1 = qdrant_hits[0]
        top1_score = top1["score"]
        top2_score = qdrant_hits[1]["score"] if len(qdrant_hits) > 1 else 0.0
        margin = top1_score - top2_score

        # Check registered identity
        is_reg, id_label = self.registry.is_track_registered(top1["payload"]["track_id"])
        identity_display = id_label if is_reg else "Unregistered Person"

        # 4. Apply decision logic
        if top1_score >= score_threshold:
            if margin >= ambiguity_margin:
                decision = "CONFIRMED_MATCH"
                msg = f"Confident match with track {top1['payload']['track_id']} ({identity_display}) at similarity {top1_score:.3f} (margin: {margin:.3f})."
            else:
                decision = "AMBIGUOUS"
                identity_display = "Unknown (Ambiguous Candidate)"
                msg = f"Match candidate {top1['payload']['track_id']} exceeded threshold ({top1_score:.3f}), but top-2 margin ({margin:.3f}) is below ambiguity threshold ({ambiguity_margin:.2f})."
        else:
            decision = "UNKNOWN"
            identity_display = "Unknown"
            msg = f"Similarity score ({top1_score:.3f}) is below confidence threshold ({score_threshold:.2f}). Reported as Unknown."

        # Format candidates list
        candidates_info = []
        for hit in qdrant_hits:
            payload = hit["payload"]
            t_id = payload["track_id"]
            reg_status, assigned_id = self.registry.is_track_registered(t_id)

            candidates_info.append({
                "track_id": t_id,
                "camera_id": payload["camera_id"],
                "score": round(hit["score"], 4),
                "identity_label": assigned_id if reg_status else "Unregistered",
                "start_time": payload["start_time"],
                "end_time": payload["end_time"],
                "duration": payload["duration"],
                "representative_crop_path": payload["representative_crop_path"],
                "source_video_path": payload["source_video_path"]
            })

        return {
            "decision": decision,
            "matched_track_id": top1["payload"]["track_id"] if decision == "CONFIRMED_MATCH" else None,
            "camera_id": top1["payload"]["camera_id"] if decision == "CONFIRMED_MATCH" else None,
            "identity_label": identity_display,
            "match_score": round(top1_score, 4),
            "margin": round(margin, 4),
            "top_candidate": candidates_info[0],
            "candidates": candidates_info,
            "message": msg
        }
