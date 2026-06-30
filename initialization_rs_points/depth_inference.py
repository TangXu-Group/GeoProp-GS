from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class DepthAnythingConfig:
    encoder: str = "vitb"
    checkpoint_path: Optional[Path] = None
    input_size: int = 518
    valid_threshold: float = 1
    invert_depth: bool = True
    visualize_dir: Optional[Path] = None


class DepthAnythingProvider:
    def __init__(
        self,
        config: DepthAnythingConfig,
        device: str = "auto",
    ):
        if config.encoder not in _MODEL_CONFIGS:
            raise ValueError(f"Unsupported encoder '{config.encoder}'. Available: {list(_MODEL_CONFIGS)}")

        self.cfg = config
        self.device = _resolve_device(device)
        model_cfg = _MODEL_CONFIGS[config.encoder]
        try:
            from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "depth_anything_v2 is not available. Install it before running initialization."
            ) from exc
        self.model = DepthAnythingV2(**model_cfg)

        checkpoint = self._resolve_checkpoint(config.checkpoint_path, config.encoder)
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.construct_aux_layers()
        self.model.freeze_network({"encoder", "decoder"})
        self.model.to(self.device).eval()

        if self.cfg.visualize_dir is not None:
            self.cfg.visualize_dir = Path(self.cfg.visualize_dir)
            self.cfg.visualize_dir.mkdir(parents=True, exist_ok=True)

    def compute(self, image_path: Path) -> np.ndarray:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for depth inference: {image_path}")

        raw_img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if raw_img is None:
            raise ValueError(f"Failed to read image {image_path} with OpenCV.")

        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(raw_img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            device_str = self._device_string()
            depth = self.model(tensor, input_size=self.cfg.input_size, device=device_str)

        depth_tensor = torch.as_tensor(depth, device="cpu").squeeze(0)
        depth_tensor = self._sanitize_depth(depth_tensor)

        depth_np = depth_tensor.numpy().astype(np.float32)
        if depth_np.ndim > 2:
            depth_np = np.squeeze(depth_np)
        if self.cfg.visualize_dir is not None:
            self._save_visualization(depth_np, image_path.name)
        return depth_np

    def _sanitize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        mask = depth > self.cfg.valid_threshold
        if mask.any():
            depth_min = depth[mask].min()
            depth = depth.clone()
            depth[~mask] = depth_min
        depth = 1.0 / depth
        return depth

    def _device_string(self) -> str:
        if self.device.index is not None:
            return f"{self.device.type}:{self.device.index}"
        return self.device.type

    def _save_visualization(self, depth: np.ndarray, image_name: str):
        depth_min = float(depth.min())
        depth_max = float(depth.max())
        if depth_max - depth_min < 1e-6:
            viz = np.zeros_like(depth, dtype=np.uint8)
        else:
            norm = (depth - depth_min) / (depth_max - depth_min)
            viz = (np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)
        viz = np.ascontiguousarray(viz)
        out_path = self.cfg.visualize_dir / f"{Path(image_name).stem}_depth.png"
        cv2.imwrite(str(out_path), viz)

    @staticmethod
    def _resolve_checkpoint(explicit: Optional[Path], encoder: str) -> Path:
        if explicit is not None:
            explicit = Path(explicit)
            if explicit.is_dir():
                explicit = explicit / f"depth_anything_v2_{encoder}.pth"
            if not explicit.exists():
                raise FileNotFoundError(f"DepthAnything checkpoint not found: {explicit}")
            return explicit

        default_candidate = Path("checkpoints") / f"depth_anything_v2_{encoder}.pth"
        if default_candidate.exists():
            return default_candidate
        raise FileNotFoundError(
            f"Checkpoint path not provided and default '{default_candidate}' is missing."
        )
