from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass
class PipelineConfig:
    workspace_root: Path
    colmap_model_dir: Path
    rgb_dir: Path
    visible_views: Sequence[str]
    dense_point_path: Path = Path("output/points_dense.ply")
    uncertain_point_path: Path = Path("output/points_proposal.ply")
    pseudo_view_dir: Path = Path("output/pseudo_views")
    pseudo_mask_dir: Path = Path("output/pseudo_masks")
    inpaint_dir: Path = Path("output/inpainted")
    pseudo_pose_count: int = 0
    rgb_scale: float = 1.0
    voxel_size: float = 0.03
    max_points: int = 250000
    max_uncertain_points: int = 50000
    depth_scale: float = 1.0
    min_depth: float = 0.0
    max_depth: float = 200.0
    coverage_threshold: float = 0.05
    inpaint_prompt: str = ""
    inpaint_api_key: Optional[str] = None
    inpaint_steps: int = 30
    inpaint_guidance: float = 20.0
    inpaint_output_format: str = "png"
    inpaint_safety: int = 2
    inpaint_timeout: int = 300
    random_seed: int = 0
    enable_inpainting: bool = False
    depth_source: str = "depth_anything_v2"
    depth_encoder: str = "vitb"
    depth_checkpoint_path: Optional[Path] = None
    depth_input_size: int = 518
    depth_device: str = "auto"
    depth_valid_threshold: float = 1.0
    depth_invert: bool = True
    depth_visualize_dir: Optional[Path] = None

    def resolve(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return (self.workspace_root / path).resolve()
