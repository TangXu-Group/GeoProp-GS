from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .colmap_adapter import CameraPose

try:
    import torch
except ImportError:
    torch = None

try:
    from scipy.ndimage import distance_transform_edt, median_filter
except ImportError:
    distance_transform_edt = None
    median_filter = None


@dataclass
class PointCloud:
    xyz: np.ndarray
    rgb: np.ndarray
    confidence: Optional[np.ndarray] = None
    alpha: Optional[np.ndarray] = None
    sigma_scale: Optional[np.ndarray] = None
    proposal_id: Optional[np.ndarray] = None
    cooldown: Optional[np.ndarray] = None

    @classmethod
    def empty(cls) -> "PointCloud":
        return cls(
            xyz=np.zeros((0, 3), dtype=np.float32),
            rgb=np.zeros((0, 3), dtype=np.uint8),
            confidence=None,
            alpha=None,
            sigma_scale=None,
            proposal_id=None,
            cooldown=None,
        )

    @property
    def size(self) -> int:
        return int(self.xyz.shape[0])


def merge_point_clouds(clouds: Sequence[PointCloud]) -> PointCloud:
    xyz_parts: List[np.ndarray] = []
    rgb_parts: List[np.ndarray] = []
    conf_parts: List[np.ndarray] = []
    alpha_parts: List[np.ndarray] = []
    sigma_parts: List[np.ndarray] = []
    id_parts: List[np.ndarray] = []
    cooldown_parts: List[np.ndarray] = []
    any_conf = any(cloud.confidence is not None for cloud in clouds if cloud.size)
    any_alpha = any(cloud.alpha is not None for cloud in clouds if cloud.size)
    any_sigma = any(cloud.sigma_scale is not None for cloud in clouds if cloud.size)
    any_ids = any(cloud.proposal_id is not None for cloud in clouds if cloud.size)
    any_cooldown = any(cloud.cooldown is not None for cloud in clouds if cloud.size)
    for cloud in clouds:
        if cloud.size == 0:
            continue
        xyz_parts.append(cloud.xyz.astype(np.float32, copy=False))
        rgb_parts.append(cloud.rgb.astype(np.uint8, copy=False))
        if any_conf:
            if cloud.confidence is not None:
                conf_parts.append(cloud.confidence.astype(np.float32, copy=False))
            else:
                conf_parts.append(np.ones((cloud.size,), dtype=np.float32))
        if any_alpha:
            if cloud.alpha is not None:
                alpha_parts.append(cloud.alpha.astype(np.float32, copy=False))
            else:
                alpha_parts.append(np.full((cloud.size,), 0.02, dtype=np.float32))
        if any_sigma:
            if cloud.sigma_scale is not None:
                sigma_parts.append(cloud.sigma_scale.astype(np.float32, copy=False))
            else:
                sigma_parts.append(np.ones((cloud.size,), dtype=np.float32))
        if any_ids:
            if cloud.proposal_id is not None:
                id_parts.append(cloud.proposal_id.astype(np.int64, copy=False))
            else:
                id_parts.append(np.zeros((cloud.size,), dtype=np.int64))
        if any_cooldown:
            if cloud.cooldown is not None:
                cooldown_parts.append(cloud.cooldown.astype(np.float32, copy=False))
            else:
                cooldown_parts.append(np.zeros((cloud.size,), dtype=np.float32))
    if not xyz_parts:
        return PointCloud.empty()
    xyz = np.concatenate(xyz_parts, axis=0)
    rgb = np.concatenate(rgb_parts, axis=0)
    confidence = np.concatenate(conf_parts, axis=0) if any_conf else None
    alpha = np.concatenate(alpha_parts, axis=0) if any_alpha else None
    sigma = np.concatenate(sigma_parts, axis=0) if any_sigma else None
    proposal_id = np.concatenate(id_parts, axis=0) if any_ids else None
    cooldown = np.concatenate(cooldown_parts, axis=0) if any_cooldown else None
    return PointCloud(
        xyz=xyz,
        rgb=rgb,
        confidence=confidence,
        alpha=alpha,
        sigma_scale=sigma,
        proposal_id=proposal_id,
        cooldown=cooldown,
    )


def compute_normals(points: np.ndarray, k: int = 10) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    nbrs = NearestNeighbors(n_neighbors=min(k, len(points)), algorithm="auto").fit(points)
    _, indices = nbrs.kneighbors(points)
    normals = np.zeros_like(points, dtype=np.float32)
    for i, neigh_idx in enumerate(indices):
        neighbor_pts = points[neigh_idx]
        centroid = neighbor_pts.mean(axis=0)
        cov = np.cov((neighbor_pts - centroid).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, np.argmin(eigvals)]
        if normal[2] < 0:
            normal = -normal
        normals[i] = normal.astype(np.float32)
    return normals


@dataclass
class PointCloudStats:
    median_depth: float
    mean_depth: float
    min_depth: float
    max_depth: float


def depth_to_world_points(
    depth: np.ndarray,
    pose: CameraPose,
    rgb: Optional[np.ndarray],
    depth_scale: float,
    min_depth: float,
    max_depth: float,
) -> PointCloud:
    if depth.ndim != 2:
        raise ValueError("Depth map must be HxW")
    height, width = depth.shape
    mask = np.isfinite(depth)
    mask &= depth > min_depth
    mask &= depth < max_depth
    if not mask.any():
        return PointCloud.empty()

    ys, xs = np.nonzero(mask)
    z = depth[ys, xs] * depth_scale
    fx, fy = pose.K[0, 0], pose.K[1, 1]
    cx, cy = pose.K[0, 2], pose.K[1, 2]

    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z
    points_cam = np.stack([x, y, z], axis=1).astype(np.float32)

    R_wc = pose.R_wc.astype(np.float32)
    t_wc = pose.t_wc.astype(np.float32)
    R_cw = R_wc.T
    cam_center = -R_cw @ t_wc
    points_world = (R_cw @ points_cam.T).T + cam_center

    # if rgb is not None:
    #     if rgb.shape[:2] != (height, width):
    #         raise ValueError(f"RGB and depth size mismatch for {pose.name}")
    #     colors = rgb[ys, xs].astype(np.uint8)
    # else:
    #     colors = np.full((points_world.shape[0], 3), 255, dtype=np.uint8)

    if rgb.shape[:2] != (height, width):
        raise ValueError(f"RGB and depth size mismatch for {pose.name}")
    colors = rgb[ys, xs].astype(np.uint8)


    return PointCloud(xyz=points_world, rgb=colors, confidence=None)


def downsample_point_cloud(
    cloud: PointCloud,
    voxel_size: float,
    max_points: int,
    seed: int,
) -> PointCloud:
    if cloud.size == 0:
        return cloud

    xyz = cloud.xyz
    rgb = cloud.rgb
    confidence = cloud.confidence
    alpha = cloud.alpha
    sigma_scale = cloud.sigma_scale
    proposal_id = cloud.proposal_id
    cooldown = cloud.cooldown
    if voxel_size > 0.0:
        grid = np.floor(xyz / voxel_size).astype(np.int64)
        hashes = _hash_voxels(grid)
        order = np.argsort(hashes)
        xyz = xyz[order]
        rgb = rgb[order]
        confidence = confidence[order] if confidence is not None else None
        alpha = alpha[order] if alpha is not None else None
        sigma_scale = sigma_scale[order] if sigma_scale is not None else None
        proposal_id = proposal_id[order] if proposal_id is not None else None
        cooldown = cooldown[order] if cooldown is not None else None
        hashes = hashes[order]

        unique_hashes, first_idx, counts = np.unique(hashes, return_index=True, return_counts=True)
        xyz_vox = []
        rgb_vox = []
        conf_vox: Optional[List[float]] = [] if confidence is not None else None
        alpha_vox: Optional[List[float]] = [] if alpha is not None else None
        sigma_vox: Optional[List[float]] = [] if sigma_scale is not None else None
        id_vox: Optional[List[float]] = [] if proposal_id is not None else None
        cooldown_vox: Optional[List[float]] = [] if cooldown is not None else None
        for start, count in zip(first_idx, counts):
            stop = start + count
            xyz_slice = xyz[start:stop]
            rgb_slice = rgb[start:stop]
            xyz_vox.append(np.mean(xyz_slice, axis=0))
            rgb_vox.append(np.mean(rgb_slice, axis=0))
            if conf_vox is not None:
                conf_vox.append(float(np.mean(confidence[start:stop])))
            if alpha_vox is not None:
                alpha_vox.append(float(np.mean(alpha[start:stop])))
            if sigma_vox is not None:
                sigma_vox.append(float(np.mean(sigma_scale[start:stop])))
            if id_vox is not None:
                id_vox.append(int(proposal_id[start]))  # preserve representative id
            if cooldown_vox is not None:
                cooldown_vox.append(float(np.max(cooldown[start:stop])))
        xyz = np.asarray(xyz_vox, dtype=np.float32)
        rgb = np.asarray(rgb_vox, dtype=np.uint8)
        if conf_vox is not None:
            confidence = np.asarray(conf_vox, dtype=np.float32)
        else:
            confidence = None
        if alpha_vox is not None:
            alpha = np.asarray(alpha_vox, dtype=np.float32)
        else:
            alpha = None
        if sigma_vox is not None:
            sigma_scale = np.asarray(sigma_vox, dtype=np.float32)
        else:
            sigma_scale = None
        if id_vox is not None:
            proposal_id = np.asarray(id_vox, dtype=np.int64)
        else:
            proposal_id = None
        if cooldown_vox is not None:
            cooldown = np.asarray(cooldown_vox, dtype=np.float32)
        else:
            cooldown = None

    if max_points > 0 and xyz.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(xyz.shape[0], size=max_points, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]
        if confidence is not None:
            confidence = confidence[idx]
        if alpha is not None:
            alpha = alpha[idx]
        if sigma_scale is not None:
            sigma_scale = sigma_scale[idx]
        if proposal_id is not None:
            proposal_id = proposal_id[idx]
        if cooldown is not None:
            cooldown = cooldown[idx]

    return PointCloud(
        xyz=xyz,
        rgb=rgb,
        confidence=confidence,
        alpha=alpha,
        sigma_scale=sigma_scale,
        proposal_id=proposal_id,
        cooldown=cooldown,
    )


def compute_point_cloud_stats(
    cloud: PointCloud,
    reference_poses: Sequence[CameraPose],
) -> PointCloudStats:
    if cloud.size == 0:
        raise ValueError("Cannot compute point-cloud depth statistics from an empty cloud.")
    if not reference_poses:
        raise ValueError("Cannot compute point-cloud depth statistics without reference poses.")

    depths: List[np.ndarray] = []
    for pose in reference_poses:
        R = pose.R_wc.astype(np.float32)
        t = pose.t_wc.astype(np.float32)
        points_cam = (R @ cloud.xyz.T + t[:, None]).T
        front = points_cam[:, 2] > 1e-3
        if front.any():
            depths.append(points_cam[front, 2])
    if not depths:
        raise ValueError("Cannot compute point-cloud depth statistics: no points are in front of reference cameras.")
    all_depths = np.concatenate(depths)
    return PointCloudStats(
        median_depth=float(np.median(all_depths)),
        mean_depth=float(np.mean(all_depths)),
        min_depth=float(np.percentile(all_depths, 5.0)),
        max_depth=float(np.percentile(all_depths, 95.0)),
    )


def save_point_cloud(path: Path, cloud: PointCloud):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # normals = compute_normals(cloud.xyz) if cloud.size > 0 else np.zeros((cloud.size, 3), dtype=np.float32)
    normals = np.zeros((cloud.size, 3), dtype=np.float32)
    has_conf = cloud.confidence is not None
    has_alpha = cloud.alpha is not None
    has_sigma = cloud.sigma_scale is not None
    has_ids = cloud.proposal_id is not None
    has_cooldown = cloud.cooldown is not None
    with path.open("w", encoding="ascii") as fh:
        fh.write("ply\n")
        fh.write("format ascii 1.0\n")
        fh.write(f"element vertex {cloud.size}\n")
        fh.write("property float x\n")
        fh.write("property float y\n")
        fh.write("property float z\n")
        fh.write("property uchar red\n")
        fh.write("property uchar green\n")
        fh.write("property uchar blue\n")
        fh.write("property float nx\n")
        fh.write("property float ny\n")
        fh.write("property float nz\n")
        if has_conf:
            fh.write("property float confidence\n")
        if has_alpha:
            fh.write("property float alpha0\n")
        if has_sigma:
            fh.write("property float sigma_scale\n")
        if has_ids:
            fh.write("property int proposal_id\n")
        if has_cooldown:
            fh.write("property float cooldown\n")
        fh.write("end_header\n")
        if cloud.size == 0:
            return
        for idx in range(cloud.size):
            x, y, z = cloud.xyz[idx]
            r, g, b = cloud.rgb[idx]
            nx, ny, nz = normals[idx]
            values = [
                f"{x:.6f}",
                f"{y:.6f}",
                f"{z:.6f}",
                str(int(r)),
                str(int(g)),
                str(int(b)),
                f"{nx:.6f}",
                f"{ny:.6f}",
                f"{nz:.6f}",
            ]
            if has_conf:
                values.append(f"{float(cloud.confidence[idx]):.4f}")
            if has_alpha:
                values.append(f"{float(cloud.alpha[idx]):.4f}")
            if has_sigma:
                values.append(f"{float(cloud.sigma_scale[idx]):.4f}")
            if has_ids:
                values.append(str(int(cloud.proposal_id[idx])))
            if has_cooldown:
                values.append(f"{float(cloud.cooldown[idx]):.1f}")
            fh.write(" ".join(values) + "\n")


DepthProvider = Callable[[CameraPose], np.ndarray]


def build_dense_point_cloud(
    poses: Sequence[CameraPose],
    depth_provider: DepthProvider,
    rgb_dir: Path,
    depth_scale: float,
    min_depth: float,
    max_depth: float,
    sparse_world_points: Optional[np.ndarray] = None,
) -> PointCloud:
    rgb_dir = Path(rgb_dir)
    clouds: List[PointCloud] = []
    use_refine = (
        torch is not None
        and sparse_world_points is not None
        and isinstance(sparse_world_points, np.ndarray)
        and sparse_world_points.size > 0
    )
    if use_refine:
        sparse_world_points = sparse_world_points.astype(np.float32, copy=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = None

    for pose in poses:
        depth = depth_provider(pose)
        if use_refine:
            sparse_depth_map, sparse_mask_map = project_sparse_depth(
                pose, sparse_world_points, depth.shape[0], depth.shape[1]
            )
            valid = int(sparse_mask_map.sum())
            if valid >= 50:
                pred_tensor = torch.from_numpy(depth).float().to(device)
                sparse_tensor = torch.from_numpy(sparse_depth_map).float().to(device)
                mask_tensor = torch.from_numpy(sparse_mask_map).to(device)
                scale, shift = calc_scale_shift(
                    sparse_tensor, pred_tensor, mask_tensor, device=device
                )
                scale = scale.to(device)
                shift = shift.to(device)
                pred_tensor = pred_tensor * scale + shift
                depth = pred_tensor.cpu().numpy()
                print(f"refine_depth_with_sparse: {refine_depth_with_sparse}")
                depth = refine_depth_with_sparse(
                    depth,
                    pose,
                    sparse_world_points,
                    device=device,
                    min_points=50,
                    cached=(sparse_depth_map, sparse_mask_map),
                    apply_global=False,
                )
        rgb_path = rgb_dir / pose.name
        rgb = read_rgb_image(rgb_path) if rgb_path.exists() else None
        cloud = depth_to_world_points(
            depth=depth,
            pose=pose,
            rgb=rgb,
            depth_scale=depth_scale,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        clouds.append(cloud)
    return merge_point_clouds(clouds)


def read_rgb_image(path: Path) -> np.ndarray:
    from PIL import Image

    image = Image.open(path)
    image = image.convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _hash_voxels(grid: np.ndarray) -> np.ndarray:
    primes = np.array([73856093, 19349663, 83492791], dtype=np.int64)
    return np.dot(grid, primes)


def refine_depth_with_sparse(
    depth: np.ndarray,
    pose: CameraPose,
    sparse_world_points: np.ndarray,
    device: Optional["torch.device"],
    min_points: int,
    cached: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    apply_global: bool = True,
) -> np.ndarray:
    if device is None or torch is None or sparse_world_points.size == 0:
        return depth

    height, width = depth.shape

    sparse_depth, sparse_mask = project_sparse_depth(pose, sparse_world_points, height, width)
    # valid = sparse_mask.sum()
    # if valid < min_points:
    #     return depth

    pred_tensor = torch.from_numpy(depth).float().to(device)
    sparse_tensor = torch.from_numpy(sparse_depth).float().to(device)
    mask_tensor = torch.from_numpy(sparse_mask).to(device)
    
    scale, shift = calc_scale_shift(sparse_tensor, pred_tensor, mask_tensor, device=device)
    pred_tensor = pred_tensor * scale + shift

    corrected = hierarchical_depth_correction(
        pred_tensor,
        sparse_tensor,
        mask_tensor,
        device=device,
    )
    return corrected.cpu().numpy()


def project_sparse_depth(
    pose: CameraPose,
    sparse_world_points: np.ndarray,
    height: int,
    width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    R = pose.R_wc.astype(np.float32)
    t = pose.t_wc.astype(np.float32).reshape(3, 1)
    K = pose.K.astype(np.float32)

    points_cam = (R @ sparse_world_points.T + t).T
    z = points_cam[:, 2]
    front = z > 1e-4
    if not np.any(front):
        return np.zeros((height, width), dtype=np.float32), np.zeros((height, width), dtype=bool)

    points_cam = points_cam[front]
    z = z[front]

    inv_z = 1.0 / z
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = np.floor(fx * points_cam[:, 0] * inv_z + cx).astype(np.int32)
    v = np.floor(fy * points_cam[:, 1] * inv_z + cy).astype(np.int32)

    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(in_bounds):
        return np.zeros((height, width), dtype=np.float32), np.zeros((height, width), dtype=bool)

    u = u[in_bounds]
    v = v[in_bounds]
    z = z[in_bounds]

    lin = v * width + u
    depth_flat = np.full(height * width, np.inf, dtype=np.float32)
    mask_flat = np.zeros(height * width, dtype=bool)

    order = np.argsort(z)
    lin_sorted = lin[order]
    z_sorted = z[order]
    _, unique_idx = np.unique(lin_sorted, return_index=True)
    keep = order[unique_idx]
    depth_flat[lin[keep]] = z[keep]
    mask_flat[lin[keep]] = True

    depth_map = depth_flat.reshape(height, width)
    mask_map = mask_flat.reshape(height, width)
    depth_map[~mask_map] = 0.0
    return depth_map, mask_map



def hierarchical_depth_correction(pred_depth, sparse_depth, sparse_mask, device='cuda'):
    """
    分层修正策略
    """
    if pred_depth.dim() > 2:
        pred_depth = pred_depth.squeeze()
    
    H, W = pred_depth.shape
    pred_depth = pred_depth.to(device)
    sparse_depth = sparse_depth.to(device)
    sparse_mask = sparse_mask.to(device)
    
    # ===== 步骤1: 全局scale/shift对齐 =====
    scale, shift = calc_scale_shift(sparse_depth, pred_depth, sparse_mask, device=device)
    corrected = pred_depth * scale + shift
    
    # ===== 步骤2: 计算修正场（只在稀疏点有值）=====
    correction_field = torch.zeros_like(corrected)
    correction_field[sparse_mask] = sparse_depth[sparse_mask] - corrected[sparse_mask]
    
    # ===== 步骤3: 用距离加权插值扩散修正场 =====
    from scipy.ndimage import distance_transform_edt, generic_filter
    
    # 距离变换
    mask_np = sparse_mask.cpu().numpy()
    distance_map = distance_transform_edt(~mask_np)
    
    # 最近邻插值修正值
    from scipy.ndimage import distance_transform_edt
    indices = distance_transform_edt(~mask_np, return_indices=True)[1]
    correction_interpolated = correction_field.cpu().numpy()[indices[0], indices[1]]
    
    # 距离加权：离稀疏点越远，修正越小
    max_influence = 100  # 最大影响距离（像素）
    weight = np.exp(-distance_map / max_influence)
    correction_smooth = correction_interpolated * weight
    
    correction_smooth = torch.from_numpy(correction_smooth).float().to(device)
    
    # 应用修正
    final_depth = corrected + correction_smooth
    
    # ===== 步骤4: 边界特殊处理（用中值滤波平滑） =====
    from scipy.ndimage import median_filter
    
    final_np = final_depth.cpu().numpy()
    
    # 只对边界区域做中值滤波
    margin = 100
    boundary_mask_np = np.zeros((H, W), dtype=bool)
    boundary_mask_np[:margin, :] = True
    boundary_mask_np[-margin:, :] = True
    boundary_mask_np[:, :margin] = True
    boundary_mask_np[:, -margin:] = True
    
    # 对边界做平滑
    final_np_smooth = median_filter(final_np, size=10)
    final_np[boundary_mask_np] = final_np_smooth[boundary_mask_np]
    
    # 再次确保稀疏点精确
    final_depth = torch.from_numpy(final_np).float().to(device)
    final_depth[sparse_mask] = sparse_depth[sparse_mask]
    
    return final_depth





def calc_scale_shift(
    k_sparse_targets: "torch.Tensor",
    k_pred_targets: "torch.Tensor",
    sparse_mask: "torch.Tensor",
    currk_dists: Optional["torch.Tensor"] = None,
    knn: bool = False,
    device: Optional["torch.device"] = None,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    device = device or torch.device("cpu")

    k_sparse_targets = k_sparse_targets.to(device)
    k_pred_targets = k_pred_targets.to(device)
    sparse_mask = sparse_mask.to(device)

    k_sparse_targets = k_sparse_targets.squeeze(0).squeeze(0)
    k_pred_targets = k_pred_targets.squeeze(0).squeeze(0)

    sparse_mask = sparse_mask.flatten().bool().cpu()
    k_sparse_targets = k_sparse_targets.flatten().cpu()
    k_pred_targets = k_pred_targets.flatten().cpu()

    if sparse_mask.sum() < 10:
        return (
            torch.tensor(1.0, device=device),
            torch.tensor(0.0, device=device),
        )

    k_sparse_targets = k_sparse_targets[sparse_mask]
    k_pred_targets = k_pred_targets[sparse_mask]

    k_pred_targets = k_pred_targets.unsqueeze(0)
    k_sparse_targets = k_sparse_targets.unsqueeze(0)

    k_pred_targets = k_pred_targets + torch.rand_like(k_pred_targets) * 1e-5

    X = torch.stack([k_pred_targets, torch.ones_like(k_pred_targets)], dim=2)

    if knn and currk_dists is not None:
        k_sparse_targets, X = perform_weighted(k_sparse_targets, X, currk_dists)
    elif k_pred_targets.shape[0] > 1:
        k_sparse_targets = k_sparse_targets.unsqueeze(-1)

    lstsq_res = torch.linalg.lstsq(X, k_sparse_targets)
    solution = lstsq_res.solution if hasattr(lstsq_res, "solution") else lstsq_res[0]
    scale = solution[:, 0].squeeze()
    shift = solution[:, 1].squeeze()

    scale = scale.to(device)
    shift = shift.to(device)
    return scale, shift
def perform_weighted(
    sparse_ori: "torch.Tensor",
    pred_ori: "torch.Tensor",
    dists: "torch.Tensor",
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    weights = 1 / dists
    wsum = weights.sum(dim=1, keepdim=True)
    weights = weights / (wsum + 1e-8)
    W = torch.diag_embed(weights)
    pred_weighted = W @ pred_ori
    sparse_weighted = W @ sparse_ori.unsqueeze(-1)
    return sparse_weighted, pred_weighted
