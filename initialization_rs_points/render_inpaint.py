from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import hashlib
import numpy as np
from sklearn.neighbors import NearestNeighbors

try:
    import torch
except ImportError:
    torch = None
from .depth_fusion import calc_scale_shift
from .colmap_adapter import CameraPose
from .depth_fusion import (
    PointCloud,
    PointCloudStats,
    hierarchical_depth_correction,
    project_sparse_depth,
)
from .inpainting_self_hosted import SelfHostedFluxFillAPI as FluxFillAPI


@dataclass
class RenderResult:
    name: str
    image: np.ndarray  # RGB uint8
    mask: np.ndarray   # bool, True where missing
    depth: np.ndarray  # float32, inf for missing
    coverage: float


class PseudoViewRenderer:
    def __init__(self, cloud: PointCloud):
        self.cloud = cloud

    def render(self, pose: CameraPose) -> RenderResult:
        height, width = pose.height, pose.width
        image = np.zeros((height, width, 3), dtype=np.uint8)
        depth = np.full((height, width), np.inf, dtype=np.float32)

        if self.cloud.size == 0:
            mask = np.ones((height, width), dtype=bool)
            return RenderResult(name=pose.name, image=image, mask=mask, depth=depth, coverage=0.0)

        xyz = self.cloud.xyz
        rgb = self.cloud.rgb

        R = pose.R_wc.astype(np.float32)
        t = pose.t_wc.astype(np.float32).reshape(3, 1)
        points_cam = (R @ xyz.T + t).T
        front = points_cam[:, 2] > 1e-4
        if not front.any():
            mask = np.ones((height, width), dtype=bool)
            return RenderResult(name=pose.name, image=image, mask=mask, depth=depth, coverage=0.0)

        points_cam = points_cam[front]
        rgb = rgb[front]

        fx, fy = pose.K[0, 0], pose.K[1, 1]
        cx, cy = pose.K[0, 2], pose.K[1, 2]

        inv_z = 1.0 / points_cam[:, 2]
        u = np.floor(fx * points_cam[:, 0] * inv_z + cx).astype(np.int32)
        v = np.floor(fy * points_cam[:, 1] * inv_z + cy).astype(np.int32)

        valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not valid.any():
            mask = np.ones((height, width), dtype=bool)
            return RenderResult(name=pose.name, image=image, mask=mask, depth=depth, coverage=0.0)

        u = u[valid]
        v = v[valid]
        z = points_cam[valid, 2]
        rgb = rgb[valid]

        lin = v * width + u
        order = np.lexsort((z, lin))
        lin = lin[order]
        z = z[order]
        rgb = rgb[order]

        unique_lin, first_idx = np.unique(lin, return_index=True)
        sel = first_idx
        flat_image = image.reshape(-1, 3)
        flat_depth = depth.reshape(-1)

        flat_image[unique_lin] = rgb[sel]
        flat_depth[unique_lin] = z[sel]

        mask = np.isinf(depth)
        coverage = unique_lin.size / (height * width)
        return RenderResult(name=pose.name, image=image, mask=mask, depth=depth, coverage=coverage)


class InpaintingIntegrator:
    def __init__(
        self,
        raw_dir: Path,
        mask_dir: Path,
        filled_dir: Path,
        prompt: str,
        api_key: Optional[str],
        steps: int,
        guidance: float,
        output_format: str,
        safety: int,
        timeout: int,
        do_inpaint: bool,
    ):
        self.raw_dir = Path(raw_dir)
        self.mask_dir = Path(mask_dir)
        self.filled_dir = Path(filled_dir)
        self.prompt = prompt
        self.api_key = api_key
        self.steps = steps
        self.guidance = guidance
        self.output_format = output_format
        self.safety = safety
        self.timeout = timeout
        self.do_inpaint = do_inpaint and api_key is not None
        self.client = FluxFillAPI(api_key) if self.do_inpaint else None

    def run(self, result: RenderResult, raw_path: Optional[Path] = None, mask_path: Optional[Path] = None) -> Path:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        self.filled_dir.mkdir(parents=True, exist_ok=True)

        if raw_path is None:
            raw_path = self.raw_dir / result.name
            bgr = cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(raw_path), bgr)
        else:
            raw_path = Path(raw_path)
            if not raw_path.exists():
                bgr = cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(raw_path), bgr)

        if mask_path is None:
            mask_path = self.mask_dir / f"{Path(result.name).stem}_mask.png"
            mask_uint8 = np.where(result.mask, 255, 0).astype(np.uint8)
            cv2.imwrite(str(mask_path), mask_uint8)
        else:
            mask_path = Path(mask_path)
            if not mask_path.exists():
                mask_uint8 = np.where(result.mask, 255, 0).astype(np.uint8)
                cv2.imwrite(str(mask_path), mask_uint8)

        if not self.do_inpaint or not self.client:
            # No inpainting requested; return the raw path (acts as filled image)
            return raw_path

        filled_path = self.filled_dir / f"{Path(result.name).stem}.{self.output_format}"
        self.client.inpaint(
            prompt=self.prompt,
            image_path=str(raw_path),
            mask_path=str(mask_path),
            output_path=str(filled_path),
            steps=self.steps,
            guidance=self.guidance,
            output_format=self.output_format,
            safety_tolerance=self.safety,
            timeout=self.timeout,
        )
        return filled_path


class UncertainPointBuilder:
    def __init__(
        self,
        stats: PointCloudStats,
        max_points: int,
        seed: int,
        min_depth: float,
        max_depth: float,
        depth_provider: Optional[Callable[[CameraPose], np.ndarray]] = None,
        sparse_world_points: Optional[np.ndarray] = None,
        depth_scale: float = 1.0,
        dense_cloud: Optional[PointCloud] = None,
        overlap_distance: float = 0.03,
        depth_hypotheses: int = 3,
        alpha_range: Tuple[float, float] = (0.01, 0.05),
        cooldown_steps: int = 400,
        sigma_scale: float = 2.5,
        depth_jitter_frac: float = 0.04,
        depth_jitter_min: float = 0.05,
    ):
        self.stats = stats
        self.max_points = max_points
        self.seed = seed
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_provider = depth_provider
        self.sparse_world_points = sparse_world_points
        self.depth_scale = depth_scale
        self.overlap_distance = overlap_distance
        self.depth_hypotheses = max(1, int(depth_hypotheses))
        self.alpha_range = (
            float(alpha_range[0]),
            float(alpha_range[1]),
        )
        self.cooldown_steps = max(0, int(cooldown_steps))
        self.base_sigma_scale = max(1.0, float(sigma_scale))
        self.depth_jitter_frac = max(0.0, float(depth_jitter_frac))
        self.depth_jitter_min = max(0.0, float(depth_jitter_min))
        self._proposal_counter = 0
        self._dense_nn = None
        if dense_cloud is not None and dense_cloud.size > 0:
            self._dense_nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(
                dense_cloud.xyz.astype(np.float32, copy=False)
            )

    def build(
        self,
        pose: CameraPose,
        filled_image: np.ndarray,
        result: RenderResult,
        depth_override: Optional[np.ndarray] = None,
    ) -> PointCloud:
        missing_coords = np.argwhere(result.mask)
        if missing_coords.size == 0:
            return PointCloud.empty()

        seed_bytes = f"{pose.name}-{self.seed}".encode("utf-8")
        pose_seed = int(hashlib.sha1(seed_bytes).hexdigest()[:8], 16)
        rng = np.random.default_rng(pose_seed)
        if self.max_points > 0 and missing_coords.shape[0] > self.max_points:
            idx = rng.choice(missing_coords.shape[0], self.max_points, replace=False)
            missing_coords = missing_coords[idx]

        depth_map = self._predict_depth_map(pose, depth_override=depth_override)
        fallback_depth = np.clip(self.stats.median_depth, self.min_depth, self.max_depth)
        fx, fy = pose.K[0, 0], pose.K[1, 1]
        cx, cy = pose.K[0, 2], pose.K[1, 2]

        R_cw = pose.R_wc.T
        cam_center = -R_cw @ pose.t_wc

        xyz_list: List[np.ndarray] = []
        rgb_list: List[np.ndarray] = []
        conf_list: List[float] = []
        alpha_list: List[float] = []
        sigma_list: List[float] = []
        id_list: List[int] = []
        cooldown_list: List[float] = []

        min_alpha, max_alpha = self.alpha_range
        alpha_low = min(min_alpha, max_alpha)
        alpha_high = max(min_alpha, max_alpha)

        for v, u in missing_coords:
            depth_val = fallback_depth
            conf = 0.15
            if depth_map is not None:
                sample = float(depth_map[v, u])
                if np.isfinite(sample) and sample > 0.0:
                    sample = float(np.clip(sample, self.min_depth, self.max_depth))
                    if sample > 0.0:
                        depth_val = sample
                        conf = 0.4
            depth_val *= self.depth_scale
            if depth_val <= 0.0:
                continue

            x = (u - cx) / fx * depth_val
            y = (v - cy) / fy * depth_val
            point_cam = np.array([x, y, depth_val], dtype=np.float32)
            point_world = R_cw @ point_cam + cam_center

            xyz_list.append(point_world)
            rgb_list.append(filled_image[v, u])
            conf_list.append(conf)

            alpha = rng.uniform(alpha_low, alpha_high) if alpha_high > alpha_low else alpha_low
            alpha_list.append(float(alpha))

            base_sigma = max(depth_val * self.depth_jitter_frac, self.depth_jitter_min)
            sigma_list.append(float(base_sigma))
            id_list.append(self._proposal_counter)
            cooldown_list.append(float(self.cooldown_steps))
            self._proposal_counter += 1

        if not xyz_list:
            return PointCloud.empty()

        xyz = np.asarray(xyz_list, dtype=np.float32)
        rgb = np.asarray(rgb_list, dtype=np.uint8)
        confidence = np.asarray(conf_list, dtype=np.float32)
        alpha = np.asarray(alpha_list, dtype=np.float32)
        sigma_scale = np.asarray(sigma_list, dtype=np.float32)
        proposal_id = np.asarray(id_list, dtype=np.int64)
        cooldown = np.asarray(cooldown_list, dtype=np.float32)

        keep_mask = self._filter_overlapping(xyz)
        if not keep_mask.any():
            return PointCloud.empty()

        xyz = xyz[keep_mask]
        rgb = rgb[keep_mask]
        confidence = confidence[keep_mask]
        alpha = alpha[keep_mask]
        sigma_scale = sigma_scale[keep_mask]
        proposal_id = proposal_id[keep_mask]
        cooldown = cooldown[keep_mask]
        return PointCloud(
            xyz=xyz,
            rgb=rgb,
            confidence=confidence,
            alpha=alpha,
            sigma_scale=sigma_scale,
            proposal_id=proposal_id,
            cooldown=cooldown,
        )

    def _predict_depth_map(
        self,
        pose: CameraPose,
        depth_override: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        if depth_override is None and self.depth_provider is None:
            return None
        if depth_override is None:
            try:
                depth = self.depth_provider(pose)
            except Exception:
                return None
        else:
            depth = depth_override

        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim > 2:
            depth = np.squeeze(depth)
        if depth.shape != (pose.height, pose.width):
            depth = cv2.resize(depth, (pose.width, pose.height), interpolation=cv2.INTER_LINEAR)

        if (
            torch is None
            or self.sparse_world_points is None
            or self.sparse_world_points.size == 0
        ):
            return depth

        height, width = depth.shape
        sparse_depth, sparse_mask = project_sparse_depth(
            pose, self.sparse_world_points, height, width
        )
        if not sparse_mask.any():
            return depth


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pred_tensor = torch.from_numpy(depth).float().to(device)
        sparse_tensor = torch.from_numpy(sparse_depth).float().to(device)
        mask_tensor = torch.from_numpy(sparse_mask)
        corrected = hierarchical_depth_correction(
            pred_tensor,
            sparse_tensor,
            mask_tensor,
            device=device,
        )
        return corrected.detach().cpu().numpy()

    def _filter_overlapping(self, xyz: np.ndarray) -> np.ndarray:
        if (
            self._dense_nn is None
            or self.overlap_distance <= 0.0
            or xyz.shape[0] == 0
        ):
            return np.ones((xyz.shape[0],), dtype=bool)

        distances, _ = self._dense_nn.kneighbors(xyz)
        return distances[:, 0] > self.overlap_distance
