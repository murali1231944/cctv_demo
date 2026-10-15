import os
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    ScoredPoint
)

import configs.config as config

class QdrantStore:
    """
    Manages vector storage and track metadata in Qdrant.
    Uses deterministic point UUIDs based on track_id to ensure idempotency and prevent duplicate records.
    """
    def __init__(
        self,
        storage_path: Union[str, Path] = config.QDRANT_PATH,
        collection_name: str = config.QDRANT_COLLECTION_NAME,
        dim: int = config.EMBEDDING_DIM
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.dim = dim

        self.client = QdrantClient(path=str(self.storage_path))
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """
        Creates the Qdrant collection with Cosine distance metric if it does not already exist.
        """
        collections = self.client.get_collections().collections
        existing_names = [c.name for c in collections]

        if self.collection_name not in existing_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dim,
                    distance=Distance.COSINE
                )
            )

    @staticmethod
    def get_deterministic_id(track_id: str) -> str:
        """
        Generates a deterministic UUID based on track_id for reproducible point IDs.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, track_id))

    def upsert_tracks(self, tracks: Dict[str, Any]) -> int:
        """
        Upserts a dictionary of tracks into Qdrant.
        Returns the number of points upserted.
        """
        if not tracks:
            return 0

        points = []
        for t_id, track in tracks.items():
            point_id = self.get_deterministic_id(t_id)
            vector = track["embedding"]

            # Store track metadata payload (exclude raw frame observations from vector store payload to keep it lean)
            payload = {
                "track_id": track["track_id"],
                "camera_id": track["camera_id"],
                "raw_track_id": track["raw_track_id"],
                "source_video_path": track["source_video_path"],
                "start_time": track["start_time"],
                "end_time": track["end_time"],
                "duration": track["duration"],
                "total_frames_observed": track["total_frames_observed"],
                "representative_crop_path": track["representative_crop_path"],
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            )

        # Batch upsert points
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return len(points)

    def search_similar(
        self,
        query_vector: Union[List[float], np.ndarray],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        camera_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches for candidate person tracks using cosine similarity.
        """
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        query_filter = None
        if camera_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="camera_id",
                        match=MatchValue(value=camera_id)
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter
        )

        results = []
        for pt in response.points:
            results.append({
                "score": float(pt.score),
                "point_id": pt.id,
                "payload": pt.payload
            })

        return results

    def get_track(self, track_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single track by its track_id.
        """
        point_id = self.get_deterministic_id(track_id)
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_vectors=True,
            with_payload=True
        )
        if points:
            pt = points[0]
            return {
                "point_id": pt.id,
                "vector": pt.vector,
                "payload": pt.payload
            }
        return None

    def get_all_tracks(self, camera_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves all tracks stored in Qdrant, optionally filtered by camera.
        """
        query_filter = None
        if camera_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="camera_id",
                        match=MatchValue(value=camera_id)
                    )
                ]
            )

        scroll_res, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=1000,
            with_payload=True,
            with_vectors=True
        )

        all_tracks = []
        for pt in scroll_res:
            all_tracks.append({
                "point_id": pt.id,
                "vector": pt.vector,
                "payload": pt.payload
            })

        return all_tracks

    def count_tracks(self, camera_id: Optional[str] = None) -> int:
        """
        Returns total number of points in the collection (or for a camera).
        """
        count_filter = None
        if camera_id:
            count_filter = Filter(
                must=[
                    FieldCondition(
                        key="camera_id",
                        match=MatchValue(value=camera_id)
                    )
                ]
            )
        res = self.client.count(
            collection_name=self.collection_name,
            count_filter=count_filter
        )
        return res.count

    def clear_collection(self) -> None:
        """
        Deletes and recreates the collection.
        """
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()

    def close(self) -> None:
        """
        Closes local client cleanly.
        """
        try:
            self.client.close()
        except Exception:
            pass
