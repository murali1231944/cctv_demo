import os
from pathlib import Path
from typing import List, Union, Optional
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import configs.config as config
from src.models.osnet import osnet_x1_0

class OSNetFeatureExtractor:
    """
    OSNet appearance feature extractor for Person Re-Identification.
    Consistently extracts 512-dimensional, L2-normalized feature vectors.
    """
    def __init__(
        self,
        weights_path: Union[str, Path] = config.OSNET_MODEL_PATH,
        device: str = config.DEVICE
    ):
        self.weights_path = Path(weights_path)
        self.device = torch.device(device)

        if not self.weights_path.exists():
            raise FileNotFoundError(f"OSNet weights not found at: {self.weights_path}")

        # Initialize OSNet model architecture
        self.model = osnet_x1_0(num_classes=1000, pretrained=False)

        # Load weights
        state_dict = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        model_dict = self.model.state_dict()
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }
        model_dict.update(filtered_state_dict)
        self.model.load_state_dict(model_dict)
        self.model.to(self.device)
        self.model.eval()

        # Standard Person ReID preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((config.REID_INPUT_HEIGHT, config.REID_INPUT_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocesses a single image (OpenCV BGR or PIL) into a normalized tensor.
        """
        if isinstance(image, np.ndarray):
            if image.size == 0:
                raise ValueError("Cannot extract features from empty image crop.")
            # Convert OpenCV BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
        else:
            pil_image = image.convert("RGB")

        tensor = self.transform(pil_image)
        return tensor

    @torch.no_grad()
    def extract_features(
        self,
        crops: Union[np.ndarray, List[np.ndarray]]
    ) -> np.ndarray:
        """
        Extracts L2-normalized 512-dim OSNet embeddings for one or more image crops.
        Returns numpy array of shape (N, 512) or (512,).
        """
        is_single = False
        if isinstance(crops, np.ndarray):
            if crops.ndim == 3:
                crops = [crops]
                is_single = True
            elif crops.ndim == 4:
                crops = [crops[i] for i in range(crops.shape[0])]
        elif not isinstance(crops, list):
            crops = [crops]
            is_single = True

        if len(crops) == 0:
            return np.empty((0, config.EMBEDDING_DIM), dtype=np.float32)

        tensors = []
        for crop in crops:
            t = self.preprocess_image(crop)
            tensors.append(t)

        batch_tensor = torch.stack(tensors, dim=0).to(self.device)

        # Forward pass through OSNet
        features = self.model(batch_tensor)  # (N, 512)

        # Apply L2 normalization
        features = F.normalize(features, p=2, dim=1)

        feats_np = features.cpu().numpy().astype(np.float32)

        if is_single:
            return feats_np[0]
        return feats_np

    def aggregate_embeddings(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extracts features from multiple crops, averages them, and re-normalizes.
        Produces an aggregated, viewpoint-robust track appearance vector (512,).
        """
        if not crops:
            raise ValueError("No crops provided for embedding aggregation.")

        feats = self.extract_features(crops)
        if feats.ndim == 1:
            return feats

        avg_feat = np.mean(feats, axis=0)
        norm = np.linalg.norm(avg_feat)
        if norm > 1e-6:
            avg_feat = avg_feat / norm
        return avg_feat.astype(np.float32)
