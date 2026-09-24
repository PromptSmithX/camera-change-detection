"""Frozen DINOv2 Torch Hub adapter for M3 appearance embeddings."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from change_detection.domain import BBox, Embedding, Observation, embedding_from_value
from change_detection.perception.contracts import PerceptionDependencyError


class DinoV2DependencyError(PerceptionDependencyError):
    """Raised when Torch, Torchvision, DINOv2, or its weights are unavailable."""


class DinoV2Encoder:
    """Encode padded ROI-valid object crops using a frozen DINOv2 backbone."""

    def __init__(
        self,
        model: str = "dinov2_vits14",
        *,
        repository: str = "facebookresearch/dinov2",
        device: str | None = None,
        input_size: int = 224,
        batch_size: int = 8,
        crop_padding_ratio: float = 0.10,
        cache_dir: str | Path | None = None,
        model_instance: Any | None = None,
    ) -> None:
        if input_size < 1 or batch_size < 1 or not 0 <= crop_padding_ratio <= 1:
            raise ValueError("DINOv2 crop/input settings are invalid")
        self.model_name = model
        self.repository = repository
        self.device = device
        self.input_size = int(input_size)
        self.batch_size = int(batch_size)
        self.crop_padding_ratio = float(crop_padding_ratio)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        try:
            import torch
            import torchvision
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise DinoV2DependencyError(
                "DINOv2 requires Torch and Torchvision; install the identity extra"
            ) from exc
        self._torch = torch
        self._torchvision = torchvision
        normalized_device = f"cuda:{device}" if device is not None and str(device).isdigit() else device
        self._resolved_device = torch.device(normalized_device) if normalized_device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if model_instance is not None:
            self._model = model_instance
        else:
            if self.cache_dir is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                torch.hub.set_dir(str(self.cache_dir))
            try:
                self._model = torch.hub.load(self.repository, self.model_name, trust_repo=True)
            except Exception as exc:  # pragma: no cover - network/model specific
                raise DinoV2DependencyError(
                    f"Could not load DINOv2 model {self.repository}:{self.model_name}: {exc}"
                ) from exc
        if hasattr(self._model, "to"):
            self._model = self._model.to(self._resolved_device)
        if hasattr(self._model, "eval"):
            self._model.eval()

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": "dinov2",
            "framework": "torch_hub",
            "repository": self.repository,
            "model": self.model_name,
            "device": str(self._resolved_device),
            "input_size": self.input_size,
            "batch_size": self.batch_size,
            "crop_padding_ratio": self.crop_padding_ratio,
            "cache_dir": str(self.cache_dir) if self.cache_dir is not None else None,
        }

    def _crop(self, image: Any, bbox: BBox, roi_bbox: BBox) -> Any:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - runtime boundary
            raise DinoV2DependencyError("DINOv2 crop preparation requires NumPy") from exc
        height, width = image.shape[:2]
        pad_x = int(math.ceil(bbox.width * self.crop_padding_ratio))
        pad_y = int(math.ceil(bbox.height * self.crop_padding_ratio))
        x1 = max(roi_bbox.x1, 0, bbox.x1 - pad_x)
        y1 = max(roi_bbox.y1, 0, bbox.y1 - pad_y)
        x2 = min(roi_bbox.x2, width, bbox.x2 + pad_x)
        y2 = min(roi_bbox.y2, height, bbox.y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Observation crop is empty after ROI clipping")
        crop = image[y1:y2, x1:x2]
        side = max(crop.shape[0], crop.shape[1])
        # ImageNet mean in BGR byte order; DINO preprocessing converts to RGB.
        canvas = np.full((side, side, 3), (103, 116, 124), dtype=crop.dtype)
        top = (side - crop.shape[0]) // 2
        left = (side - crop.shape[1]) // 2
        canvas[top : top + crop.shape[0], left : left + crop.shape[1]] = crop
        return canvas

    def _tensor(self, crop: Any) -> Any:
        torch = self._torch
        functional = self._torchvision.transforms.functional
        interpolation = self._torchvision.transforms.InterpolationMode.BICUBIC
        rgb = crop[:, :, ::-1].copy()
        tensor = functional.to_tensor(rgb)
        tensor = functional.resize(tensor, [self.input_size, self.input_size], interpolation=interpolation, antialias=True)
        return functional.normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def encode(self, frame: Any, observations: list[Observation], roi_bbox: BBox) -> list[Embedding]:
        image = getattr(frame, "image", frame)
        if not observations:
            return []
        tensors = [self._tensor(self._crop(image, observation.bbox, roi_bbox)) for observation in observations]
        result: list[Embedding] = []
        torch = self._torch
        for start in range(0, len(tensors), self.batch_size):
            batch = torch.stack(tensors[start : start + self.batch_size]).to(self._resolved_device)
            try:
                with torch.inference_mode():
                    output = self._model(batch)
            except Exception as exc:  # pragma: no cover - model/runtime specific
                raise DinoV2DependencyError(f"DINOv2 inference failed: {exc}") from exc
            if isinstance(output, dict):
                output = output.get("x_norm_clstoken", output.get("features"))
            if output is None or not hasattr(output, "detach"):
                raise DinoV2DependencyError("DINOv2 model did not return feature tensors")
            for row in output.detach().float().cpu().tolist():
                embedding = embedding_from_value(row)
                if embedding is None:
                    raise DinoV2DependencyError("DINOv2 produced an empty embedding")
                result.append(embedding)
        return result
