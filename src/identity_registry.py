import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

import configs.config as config
from src.utils import save_json_atomic, load_json

class IdentityRegistry:
    """
    Persistent identity registry managing resources.json.
    Maps high-level person identities (e.g. Person_A) to their cross-camera track references.
    Enforces atomic updates, duplicate prevention, and schema integrity.
    """
    def __init__(self, registry_path: Union[str, Path] = config.RESOURCES_JSON_PATH):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.identities: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self.load()

    def load(self) -> None:
        """
        Loads and validates resources.json.
        """
        if not self.registry_path.exists():
            self.identities = {}
            return

        raw_data = load_json(self.registry_path, default={})
        if not isinstance(raw_data, dict):
            self.identities = {}
            return

        # Validate schema integrity
        validated = {}
        for identity_label, identity_data in raw_data.items():
            if not isinstance(identity_data, dict) or "references" not in identity_data:
                continue

            valid_refs = []
            for ref in identity_data.get("references", []):
                if not isinstance(ref, dict):
                    continue
                req_keys = ["camera_id", "track_id", "start_time", "end_time", "crop_path", "vector"]
                if all(k in ref for k in req_keys) and len(ref["vector"]) == config.EMBEDDING_DIM:
                    valid_refs.append(ref)

            if valid_refs:
                validated[identity_label] = {"references": valid_refs}

        self.identities = validated

    def save(self) -> None:
        """
        Persists the identity registry atomically to resources.json.
        """
        save_json_atomic(self.identities, self.registry_path)

    def is_track_registered(self, track_id: str) -> Tuple[bool, Optional[str]]:
        """
        Checks whether a camera-specific track ID is already registered under any identity.
        Returns (is_registered, identity_label).
        """
        for id_label, data in self.identities.items():
            for ref in data["references"]:
                if ref["track_id"] == track_id:
                    return True, id_label
        return False, None

    def get_registered_track_ids(self) -> List[str]:
        """
        Returns a list of all track IDs currently registered in resources.json.
        """
        track_ids = []
        for _, data in self.identities.items():
            for ref in data["references"]:
                track_ids.append(ref["track_id"])
        return track_ids

    def register_camera1_track(
        self,
        identity_label: str,
        track_info: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Registers a Camera 1 track with a user-assigned identity label (e.g. Person_A).
        Guarantees idempotent registration and prevents duplicate track references.
        """
        track_id = track_info["track_id"]

        # Check if already registered
        is_reg, existing_id = self.is_track_registered(track_id)
        if is_reg:
            return False, f"Track '{track_id}' is already registered under identity '{existing_id}'."

        identity_label = identity_label.strip()
        if not identity_label:
            raise ValueError("Identity label cannot be empty.")

        new_ref = {
            "camera_id": track_info["camera_id"],
            "track_id": track_id,
            "start_time": round(float(track_info["start_time"]), 3),
            "end_time": round(float(track_info["end_time"]), 3),
            "crop_path": str(track_info["representative_crop_path"]).replace("\\", "/"),
            "vector": [round(float(v), 6) for v in track_info["embedding"]]
        }

        if identity_label not in self.identities:
            self.identities[identity_label] = {"references": []}

        self.identities[identity_label]["references"].append(new_ref)
        self.save()
        return True, f"Successfully registered '{track_id}' as '{identity_label}'."

    def _next_available_identity_label(self) -> str:
        """
        Finds the next sequential Person_X identity label (e.g., Person_A, Person_B, etc.).
        """
        existing = set(self.identities.keys())
        # Try letters A-Z
        for i in range(26):
            candidate = f"Person_{chr(65 + i)}"
            if candidate not in existing:
                return candidate
        # Fallback to Person_AA, Person_AB...
        idx = 1
        while True:
            candidate = f"Person_New_{idx}"
            if candidate not in existing:
                return candidate
            idx += 1

    def match_against_registry(
        self,
        query_vector: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Compares a query vector against all registered identities in resources.json.
        Returns list of candidate matches sorted by maximum cosine similarity descending.
        """
        if not self.identities:
            return []

        query_vector = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(query_vector)
        if q_norm > 1e-6:
            query_vector = query_vector / q_norm

        candidates = []
        for id_label, data in self.identities.items():
            ref_scores = []
            for ref in data["references"]:
                ref_vec = np.array(ref["vector"], dtype=np.float32)
                r_norm = np.linalg.norm(ref_vec)
                if r_norm > 1e-6:
                    ref_vec = ref_vec / r_norm
                sim = float(np.dot(query_vector, ref_vec))
                ref_scores.append(sim)

            if ref_scores:
                best_sim = max(ref_scores)
                avg_sim = float(np.mean(ref_scores))
                candidates.append({
                    "identity_label": id_label,
                    "best_score": round(best_sim, 4),
                    "avg_score": round(avg_sim, 4),
                    "references_count": len(ref_scores)
                })

        candidates.sort(key=lambda c: c["best_score"], reverse=True)
        return candidates

    def register_camera2_track(
        self,
        track_info: Dict[str, Any],
        match_threshold: float = config.DEFAULT_MATCH_THRESHOLD,
        ambiguity_margin: float = config.AMBIGUITY_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Registers a Camera 2 track:
        1. Checks if already registered (idempotency).
        2. Compares against all registered identities in resources.json.
        3. If top match passes threshold and ambiguity margin:
           Appends reference to the existing identity.
        4. Otherwise:
           Creates a new unique identity record (e.g. Person_B).
        """
        track_id = track_info["track_id"]

        is_reg, existing_id = self.is_track_registered(track_id)
        if is_reg:
            return {
                "status": "ALREADY_REGISTERED",
                "identity_label": existing_id,
                "track_id": track_id,
                "message": f"Track '{track_id}' is already registered under '{existing_id}'."
            }

        new_ref = {
            "camera_id": track_info["camera_id"],
            "track_id": track_id,
            "start_time": round(float(track_info["start_time"]), 3),
            "end_time": round(float(track_info["end_time"]), 3),
            "crop_path": str(track_info["representative_crop_path"]).replace("\\", "/"),
            "vector": [round(float(v), 6) for v in track_info["embedding"]]
        }

        query_vec = np.array(track_info["embedding"], dtype=np.float32)
        candidates = self.match_against_registry(query_vec)

        assigned_label = None
        decision = None
        match_score = 0.0

        if candidates:
            top1 = candidates[0]
            top1_score = top1["best_score"]
            top2_score = candidates[1]["best_score"] if len(candidates) > 1 else 0.0
            margin = top1_score - top2_score

            # Check threshold and ambiguity margin
            if top1_score >= match_threshold and margin >= ambiguity_margin:
                # Confirmed cross-camera match!
                assigned_label = top1["identity_label"]
                decision = "MATCHED_EXISTING"
                match_score = top1_score
                self.identities[assigned_label]["references"].append(new_ref)
                msg = f"Track '{track_id}' confidently matched existing identity '{assigned_label}' (score={top1_score:.3f}, margin={margin:.3f})."
            else:
                # Ambiguous or below threshold -> create new identity
                assigned_label = self._next_available_identity_label()
                decision = "NEW_IDENTITY"
                match_score = top1_score
                self.identities[assigned_label] = {"references": [new_ref]}
                msg = f"Track '{track_id}' did not reliably match existing identities (top score={top1_score:.3f}, margin={margin:.3f} vs threshold {match_threshold:.2f}). Created new identity '{assigned_label}'."
        else:
            # No registered identities exist yet
            assigned_label = self._next_available_identity_label()
            decision = "NEW_IDENTITY"
            self.identities[assigned_label] = {"references": [new_ref]}
            msg = f"No existing identities registered. Created new identity '{assigned_label}'."

        self.save()
        return {
            "status": decision,
            "identity_label": assigned_label,
            "track_id": track_id,
            "match_score": match_score,
            "candidates": candidates,
            "message": msg
        }

    def delete_identity(self, identity_label: str) -> bool:
        """
        Removes an identity from resources.json.
        """
        if identity_label in self.identities:
            del self.identities[identity_label]
            self.save()
            return True
        return False

    def reset_registry(self) -> None:
        """
        Clears all identities and saves empty registry.
        """
        self.identities = {}
        self.save()
