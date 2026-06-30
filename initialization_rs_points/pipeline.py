from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData

from .colmap_adapter import CameraPose, load_camera_poses, resolve_visible_poses
from .config import PipelineConfig
from .depth_fusion import (
    PointCloud,
    PointCloudStats,
    build_dense_point_cloud,
    compute_point_cloud_stats,
    downsample_point_cloud,
    merge_point_clouds,
    read_rgb_image,
    save_point_cloud,
)
from .depth_inference import DepthAnythingConfig, DepthAnythingProvider
from .render_inpaint import (
    RenderResult,
    InpaintingIntegrator,
    PseudoViewRenderer,
    UncertainPointBuilder,
)


class SequentialInitializationPipeline:
    """Initialization pipeline that incrementally fuses proposal points after each inpaint."""

    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self._pseudo_depth_provider = None

    def run(self) -> Dict[str, str]:
        print("[SeqInit] Loading COLMAP poses...")
        poses = load_camera_poses(self.cfg.colmap_model_dir)
        # visible = resolve_visible_poses(
        #     poses,
        #     visible_views=self.cfg.visible_views,
        # )
        # if not visible:
        #     raise RuntimeError("No visible views resolved from COLMAP model.")

        # print(f"[SeqInit] Visible views: {len(visible)}")
        # self._apply_rgb_scale(visible)
        # pseudo_poses = self._sample_interpolated_poses(visible)
        # print(f"[SeqInit] Pseudo poses prepared: {len(pseudo_poses)} (mode=interpolate)")

        visible = resolve_visible_poses(
            poses,
            visible_views=self.cfg.visible_views,
        )
        if not visible:
            raise RuntimeError("No visible views resolved from COLMAP model.")

        visible_names = {pose.name for pose in visible}
        holdout = [pose for name, pose in poses.items() if name not in visible_names]
        holdout.sort(key=lambda p: p.name)

        print(f"[SeqInit] Visible views: {len(visible)} | Holdout views: {len(holdout)}")
        self._apply_rgb_scale(visible)
        self._apply_rgb_scale(holdout)

        pseudo_poses = self._sample_interpolated_poses(holdout)
        print(f"[SeqInit] Pseudo poses prepared: {len(pseudo_poses)} (mode=interpolate-holdout)")

        depth_provider = self._build_depth_provider()
        sparse_points = self._load_colmap_points()

        dense = build_dense_point_cloud(
            poses=visible,
            depth_provider=depth_provider,
            rgb_dir=self.cfg.rgb_dir,
            depth_scale=self.cfg.depth_scale,
            min_depth=self.cfg.min_depth,
            max_depth=self.cfg.max_depth,
            sparse_world_points=sparse_points,
        )
        dense_downsampled = downsample_point_cloud(
            dense,
            voxel_size=self.cfg.voxel_size,
            max_points=self.cfg.max_points,
            seed=self.cfg.random_seed,
        )
        dense_path = self.cfg.resolve(self.cfg.dense_point_path)
        save_point_cloud(dense_path, dense_downsampled)
        print(f"[SeqInit] Saved dense point cloud ({dense.size} pts) -> {dense_path}")
        dense_downsampled = dense
        stats = compute_point_cloud_stats(dense_downsampled, visible)



        if not pseudo_poses:
            uncertain = PointCloud.empty()
            uncertain_path = self.cfg.resolve(self.cfg.uncertain_point_path)
            save_point_cloud(uncertain_path, uncertain)
            print("[SeqInit] No pseudo poses resolved; proposal cloud left empty.")
            return {
                "dense_point_cloud": str(dense_path),
                "uncertain_point_cloud": str(uncertain_path),
                "proposal_point_cloud": str(uncertain_path),
                "pseudo_view_dir": "",
                "pseudo_mask_dir": "",
                "inpaint_dir": "",
            }

        raw_dir = self.cfg.resolve(self.cfg.pseudo_view_dir)
        mask_dir = self.cfg.resolve(self.cfg.pseudo_mask_dir)
        filled_dir = self.cfg.resolve(self.cfg.inpaint_dir)

        inpaint = InpaintingIntegrator(
            raw_dir=raw_dir,
            mask_dir=mask_dir,
            filled_dir=filled_dir,
            prompt=self.cfg.inpaint_prompt,
            api_key=self.cfg.inpaint_api_key,
            steps=self.cfg.inpaint_steps,
            guidance=self.cfg.inpaint_guidance,
            output_format=self.cfg.inpaint_output_format,
            safety=self.cfg.inpaint_safety,
            timeout=self.cfg.inpaint_timeout,
            do_inpaint=self.cfg.enable_inpainting,
        )

        current_cloud = dense_downsampled
        current_stats = stats
        builder_counter = 0
        uncertain_parts: List[PointCloud] = []

        prepose_poses = self._generate_prepose_poses(visible)
        if prepose_poses:
            print(f"[SeqInit] Running prepose pass on {len(prepose_poses)} shifted poses.")
            current_cloud, current_stats, builder_counter = self._process_pose_sequence(
                poses=prepose_poses,
                label="Prepose",
                current_cloud=current_cloud,
                current_stats=current_stats,
                depth_provider=depth_provider,
                sparse_points=sparse_points,
                builder_counter=builder_counter,
                uncertain_parts=uncertain_parts,
                inpaint=inpaint,
                visible=visible,
                raw_dir=raw_dir,
                mask_dir=mask_dir,
            )

        current_cloud, current_stats, builder_counter = self._process_pose_sequence(
            poses=pseudo_poses,
            label="Pseudo",
            current_cloud=current_cloud,
            current_stats=current_stats,
            depth_provider=depth_provider,
            sparse_points=sparse_points,
            builder_counter=builder_counter,
            uncertain_parts=uncertain_parts,
            inpaint=inpaint,
            visible=visible,
            raw_dir=raw_dir,
            mask_dir=mask_dir,
        )

        uncertain = merge_point_clouds(uncertain_parts) if uncertain_parts else PointCloud.empty()
        if uncertain.size > 0:
            uncertain = downsample_point_cloud(
                uncertain,
                voxel_size=self.cfg.voxel_size,
                max_points=self.cfg.max_uncertain_points,
                seed=self.cfg.random_seed,
            )
        uncertain_path = self.cfg.resolve(self.cfg.uncertain_point_path)
        save_point_cloud(uncertain_path, uncertain)
        print(f"[SeqInit] Saved proposal point cloud ({uncertain.size} pts) -> {uncertain_path}")

        return {
            "dense_point_cloud": str(dense_path),
            "uncertain_point_cloud": str(uncertain_path),
            "proposal_point_cloud": str(uncertain_path),
            "pseudo_view_dir": str(raw_dir),
            "pseudo_mask_dir": str(mask_dir),
            "inpaint_dir": str(filled_dir),
        }

    def _make_uncertain_builder(
        self,
        stats: PointCloudStats,
        dense_cloud: PointCloud,
        depth_provider: Optional[Callable[[CameraPose], np.ndarray]],
        sparse_points: Optional[np.ndarray],
        proposal_counter: int = 0,
    ) -> Optional[UncertainPointBuilder]:
        if not self.cfg.enable_inpainting:
            return None
        overlap_distance = self.cfg.voxel_size * 1.5 if self.cfg.voxel_size > 0 else 0.05
        builder = UncertainPointBuilder(
            stats=stats,
            max_points=self.cfg.max_uncertain_points,
            seed=self.cfg.random_seed,
            min_depth=self.cfg.min_depth,
            max_depth=self.cfg.max_depth,
            depth_provider=depth_provider,
            sparse_world_points=sparse_points,
            depth_scale=self.cfg.depth_scale,
            dense_cloud=dense_cloud,
            overlap_distance=overlap_distance,
        )
        builder._proposal_counter = proposal_counter  # preserves monotonic proposal ids
        return builder

    def _process_pose_sequence(
        self,
        poses: List[CameraPose],
        label: str,
        current_cloud: PointCloud,
        current_stats: PointCloudStats,
        depth_provider: Optional[Callable[[CameraPose], np.ndarray]],
        sparse_points: Optional[np.ndarray],
        builder_counter: int,
        uncertain_parts: List[PointCloud],
        inpaint: InpaintingIntegrator,
        visible: List[CameraPose],
        raw_dir: Path,
        mask_dir: Path,
    ) -> Tuple[PointCloud, PointCloudStats, int]:
        if not poses:
            return current_cloud, current_stats, builder_counter

        builder = self._make_uncertain_builder(
            current_stats,
            current_cloud,
            depth_provider,
            sparse_points,
            builder_counter,
        )
        if builder is None:
            print(f"[SeqInit] Skipping {label.lower()} poses because inpainting is disabled.")
            return current_cloud, current_stats, builder_counter

        raw_dir = Path(raw_dir)
        mask_dir = Path(mask_dir)

        for pose in poses:
            renderer = PseudoViewRenderer(current_cloud)
            render_result = renderer.render(pose)
            if getattr(self.cfg, "clean_inpaint_mask", False):
                min_area = int(getattr(self.cfg, "clean_mask_min_area", 100))
                cleaned_mask = _clean_inpaint_mask(render_result.mask, min_area)
                render_result = RenderResult(
                    name=render_result.name,
                    image=render_result.image,
                    mask=cleaned_mask,
                    depth=render_result.depth,
                    coverage=render_result.coverage,
                )
            raw_path, mask_path = _write_render_outputs(render_result, raw_dir, mask_dir)
            coverage_pct = render_result.coverage * 100.0
            print(f"[SeqInit] ({label}) {pose.name}: coverage {coverage_pct:.1f}% | raw -> {raw_path}")

            filled_path = inpaint.run(render_result, raw_path=raw_path, mask_path=mask_path)
            filled_image = read_rgb_image(filled_path)
            filled_image = _ensure_size(filled_image, pose.width, pose.height)
            depth_override = None
            if getattr(self.cfg, "pseudo_depth_from_render", False):
                depth_override = self._predict_depth_from_image(pose, Path(filled_path))
            extras = builder.build(pose, filled_image, render_result, depth_override=depth_override)
            builder_counter = getattr(builder, "_proposal_counter", builder_counter)

            if extras.size == 0:
                builder = self._make_uncertain_builder(
                    current_stats,
                    current_cloud,
                    depth_provider,
                    sparse_points,
                    builder_counter,
                )
                continue

            uncertain_parts.append(extras)
            current_cloud = merge_point_clouds([current_cloud, extras])
            current_stats = compute_point_cloud_stats(current_cloud, visible)
            builder = self._make_uncertain_builder(
                current_stats,
                current_cloud,
                depth_provider,
                sparse_points,
                builder_counter,
            )
            print(f"[SeqInit] ({label}) {pose.name}: added {extras.size} proposal points -> {filled_path}")

        return current_cloud, current_stats, builder_counter

    def _generate_prepose_poses(self, visible: List[CameraPose]) -> List[CameraPose]:
        offset = float(getattr(self.cfg, "prepose_offset", 0.0))
        if offset <= 1e-6 or not visible:
            return []
        target = int(getattr(self.cfg, "prepose_count", 0))
        selected = self._select_pose_subset(visible, target)
        shifted: List[CameraPose] = []
        for idx, pose in enumerate(selected):
            R_wc = pose.R_wc.astype(np.float32)
            forward = R_wc[2, :].astype(np.float32)
            center = pose.camera_center.astype(np.float32)
            new_center = center - offset * forward
            t_new = (-R_wc @ new_center.reshape(3, 1)).reshape(3).astype(np.float32)
            shifted.append(
                CameraPose(
                    name=f"prepose_{idx:03d}_{pose.name}",
                    image_id=pose.image_id,
                    camera_id=pose.camera_id,
                    R_wc=R_wc.copy(),
                    t_wc=t_new,
                    K=pose.K.astype(np.float32, copy=True),
                    width=pose.width,
                    height=pose.height,
                )
            )
        return shifted

    def _select_pose_subset(self, poses: List[CameraPose], target: int) -> List[CameraPose]:
        total = len(poses)
        if total == 0 or target <= 0 or target >= total:
            return poses
        if target == 1:
            indices = [total // 2]
        else:
            step = (total - 1) / float(target - 1)
            indices = []
            for idx in range(target):
                value = int(round(idx * step))
                if indices and value <= indices[-1]:
                    value = indices[-1] + 1
                indices.append(min(value, total - 1))
            for i in range(target - 2, -1, -1):
                if indices[i] >= indices[i + 1]:
                    indices[i] = indices[i + 1] - 1
            indices = [max(0, min(val, total - 1)) for val in indices]
        return [poses[i] for i in indices]

    def _build_depth_provider(self) -> Optional[Callable[[CameraPose], np.ndarray]]:


        if self.cfg.depth_source == "depth_anything_v2":
            visualize_dir = (
                self.cfg.resolve(self.cfg.depth_visualize_dir)
                if self.cfg.depth_visualize_dir is not None
                else None
            )
            checkpoint_path = self.cfg.depth_checkpoint_path
            if checkpoint_path is not None and not Path(checkpoint_path).is_absolute():
                checkpoint_path = self.cfg.resolve(Path(checkpoint_path))
            config = DepthAnythingConfig(
                encoder=self.cfg.depth_encoder,
                checkpoint_path=checkpoint_path,
                input_size=self.cfg.depth_input_size,
                valid_threshold=self.cfg.depth_valid_threshold,
                invert_depth=self.cfg.depth_invert,
                visualize_dir=visualize_dir,
            )
            provider = DepthAnythingProvider(config=config, device=self.cfg.depth_device)
            rgb_dir = Path(self.cfg.rgb_dir)
            if not rgb_dir.is_absolute():
                rgb_dir = self.cfg.resolve(rgb_dir)

            def _wrapped(pose) -> np.ndarray:
                image_path = rgb_dir / pose.name
                return provider.compute(image_path)

            return _wrapped

        raise ValueError(f"Unsupported depth_source '{self.cfg.depth_source}'.")

    def _predict_depth_from_image(self, pose: CameraPose, image_path: Path) -> Optional[np.ndarray]:
        provider = self._get_pseudo_depth_provider()
        if provider is None:
            return None
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Pseudo depth image not found: {image_path}")
        depth = provider.compute(image_path)
        if depth is None:
            return None
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim > 2:
            depth = np.squeeze(depth)
        if depth.shape != (pose.height, pose.width):
            depth = cv2.resize(depth, (pose.width, pose.height), interpolation=cv2.INTER_LINEAR)
        return depth

    def _get_pseudo_depth_provider(self) -> Optional[DepthAnythingProvider]:
        if not getattr(self.cfg, "pseudo_depth_from_render", False):
            return None
        if self._pseudo_depth_provider is not None:
            return self._pseudo_depth_provider
        checkpoint_path = self.cfg.depth_checkpoint_path
        if checkpoint_path is None:
            raise ValueError(
                "pseudo_depth_from_render requires --depth-checkpoint so DepthAnything can run on pseudo views."
            )
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.cfg.resolve(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Depth checkpoint for pseudo depth not found: {checkpoint_path}")
        visualize_dir = (
            self.cfg.resolve(self.cfg.depth_visualize_dir)
            if self.cfg.depth_visualize_dir is not None
            else None
        )
        config = DepthAnythingConfig(
            encoder=self.cfg.depth_encoder,
            checkpoint_path=checkpoint_path,
            input_size=self.cfg.depth_input_size,
            valid_threshold=self.cfg.depth_valid_threshold,
            invert_depth=self.cfg.depth_invert,
            visualize_dir=visualize_dir,
        )
        self._pseudo_depth_provider = DepthAnythingProvider(config=config, device=self.cfg.depth_device)
        return self._pseudo_depth_provider

    def _load_colmap_points(self) -> Optional[np.ndarray]:
        model_dir = Path(self.cfg.colmap_model_dir)
        print(f"[SeqInit] model_dir: {model_dir}")

        scene_root = model_dir.parent.parent
        if scene_root.name.isdigit():
            scene_root = scene_root.parent

        if "levir" in str(model_dir).lower():
            ply_path = scene_root / "3_views" / "dense" / "fused.ply"
        elif "rsscene" in str(model_dir).lower():
            ply_path = scene_root / "7_views" / "dense" / "fused.ply"
        else:
            raise ValueError(f"Invalid scene path: {model_dir}")

        if ply_path.exists():
            xyzs = _load_ply_vertices(ply_path)
            print(f"[SeqInit] Loaded dense COLMAP points from {ply_path}")
        else:
            raise FileNotFoundError(f"Dense COLMAP points not found at {ply_path}")

        return xyzs

    def _sample_interpolated_poses(self, holdout: List[CameraPose]) -> List[CameraPose]:
        # target = int(self.cfg.pseudo_pose_count)
        # if target <= 0:
        #     return []
        # ordered = sorted(visible, key=lambda pose: pose.name)
        # if len(ordered) < 2:
        #     print("[SeqInit] Not enough visible poses to interpolate; need at least two.")
        #     return []
        target = int(self.cfg.pseudo_pose_count)
        if target <= 0:
            return []
        ordered = sorted(holdout, key=lambda pose: pose.name)
        if len(ordered) < 2:
            print("[SeqInit] Not enough holdout poses to interpolate; need at least two.")
            return []
        interpolated: List[CameraPose] = []
        num_segments = len(ordered) - 1
        base = target // num_segments
        remainder = target % num_segments
        pose_idx = 0
        for seg_idx in range(num_segments):
            count = base + (1 if seg_idx < remainder else 0)
            if count <= 0:
                continue
            pose_a = ordered[seg_idx]
            pose_b = ordered[seg_idx + 1]
            for local_idx in range(count):
                t = float(local_idx + 1) / float(count + 1)
                name = f"pseudo_interp_{pose_idx:04d}.png"
                interpolated.append(_interp_adjacent_pose(pose_a, pose_b, t, name))
                pose_idx += 1
        print(f"[SeqInit] Sequentially generated {len(interpolated)} interpolated pseudo poses (pseudo_count={target}).")
        return interpolated

    def _apply_rgb_scale(self, poses: List[CameraPose]) -> None:
        if not poses:
            return
        scale = float(self.cfg.rgb_scale)
        if abs(scale - 1.0) < 1e-6:
            return
        if scale <= 0.0:
            raise ValueError("rgb_scale must be positive.")

        for pose in poses:
            if pose.K.dtype != np.float32:
                pose.K = pose.K.astype(np.float32)
            pose.K[0, 0] *= scale
            pose.K[1, 1] *= scale
            pose.K[0, 2] *= scale
            pose.K[1, 2] *= scale
            pose.width = max(1, int(round(pose.width * scale)))
            pose.height = max(1, int(round(pose.height * scale)))
        print(f"[SeqInit] Applied rgb_scale={scale:.4f} to {len(poses)} poses")


def _ensure_size(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return image
    pil_image = Image.fromarray(image)
    resized = pil_image.resize((width, height), Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _write_render_outputs(result: RenderResult, raw_dir: Path, mask_dir: Path) -> Tuple[Path, Path]:
    raw_dir = Path(raw_dir)
    mask_dir = Path(mask_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / result.name
    mask_path = mask_dir / f"{Path(result.name).stem}_mask.png"
    bgr = cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(raw_path), bgr)
    mask_uint8 = np.where(result.mask, 255, 0).astype(np.uint8)
    cv2.imwrite(str(mask_path), mask_uint8)
    return raw_path, mask_path


def _interp_adjacent_pose(
    pose_a: CameraPose,
    pose_b: CameraPose,
    t: float,
    name: str,
) -> CameraPose:
    t_clamped = float(np.clip(t, 0.0, 1.0))
    center0 = pose_a.camera_center
    center1 = pose_b.camera_center
    camera_center = ((1.0 - t_clamped) * center0 + t_clamped * center1).astype(np.float32)
    R_interp = _interp_rotation(pose_a.R_wc, pose_b.R_wc, t_clamped)
    t_interp = (-R_interp @ camera_center.reshape(3, 1)).reshape(3).astype(np.float32)
    return CameraPose(
        name=name,
        image_id=-1,
        camera_id=pose_a.camera_id,
        R_wc=R_interp.astype(np.float32),
        t_wc=t_interp,
        K=pose_a.K.astype(np.float32, copy=True),
        width=pose_a.width,
        height=pose_a.height,
    )


def _interp_rotation(R0: np.ndarray, R1: np.ndarray, t: float) -> np.ndarray:
    if t <= 0.0:
        return R0.copy()
    if t >= 1.0:
        return R1.copy()
    R_rel = R1 @ R0.T
    rvec, _ = cv2.Rodrigues(R_rel)
    rvec_interp = rvec * t
    R_step, _ = cv2.Rodrigues(rvec_interp)
    return (R_step @ R0).astype(np.float32)


def _load_ply_vertices(path: Path) -> np.ndarray:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    x = np.asarray(vertex["x"], dtype=np.float32)
    y = np.asarray(vertex["y"], dtype=np.float32)
    z = np.asarray(vertex["z"], dtype=np.float32)
    return np.stack([x, y, z], axis=1)


def _clean_inpaint_mask(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8) * 255
    _, mask_u8 = cv2.threshold(mask_u8, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    clean = np.zeros_like(mask_u8)
    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area >= max(1, min_area):
            clean[labels == idx] = 255

    clean = cv2.dilate(clean, kernel, iterations=2)
    return clean >= 127
