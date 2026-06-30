import numpy as np
from typing import Tuple
from utils.stepfun import sample_np

def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(lookdir, up, position, subtract_position=False):
  """Construct lookat view matrix."""
  vec2 = normalize((lookdir - position) if subtract_position else lookdir)
  vec0 = normalize(np.cross(up, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m


def poses_avg(poses):
  """New pose using average position, z-axis, and up vector of input poses."""
  position = poses[:, :3, 3].mean(0)
  z_axis = poses[:, :3, 2].mean(0)
  up = poses[:, :3, 1].mean(0)
  cam2world = viewmatrix(z_axis, up, position)
  return cam2world


def focus_point_fn(poses):
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt



def recenter_poses(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Recenter poses around the origin."""
  cam2world = poses_avg(poses)
  transform = np.linalg.inv(pad_poses(cam2world))
  poses = transform @ pad_poses(poses)
  return unpad_poses(poses), transform



def generate_spiral_path(poses_arr,
                         n_frames: int = 180,
                         n_rots: int = 2,
                         zrate: float = .5) -> np.ndarray:
  """Calculates a forward facing spiral path for rendering."""
  poses = poses_arr[:, :-2].reshape([-1, 3, 5])
  bounds = poses_arr[:, -2:]
  fix_rotation = np.array([
      [0, -1, 0, 0],
      [1, 0, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
  ], dtype=np.float32)
  poses = poses[:, :3, :4] @ fix_rotation

  scale = 1. / (bounds.min() * .75)
  poses[:, :3, 3] *= scale
  bounds *= scale
  poses, transform = recenter_poses(poses)

  close_depth, inf_depth = bounds.min() * .9, bounds.max() * 5.
  dt = .75
  focal = 1 / (((1 - dt) / close_depth + dt / inf_depth))

  # Get radii for spiral path using 90th percentile of camera positions.
  positions = poses[:, :3, 3]
  radii = np.percentile(np.abs(positions), 90, 0)
  radii = np.concatenate([radii, [1.]])

  # Generate poses for spiral path.
  render_poses = []
  cam2world = poses_avg(poses)
  up = poses[:, :3, 1].mean(0)
  for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
    t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
    position = cam2world @ t
    lookat = cam2world @ [0, 0, -focal, 1.]
    z_axis = position - lookat
    render_pose = np.eye(4)
    render_pose[:3] = viewmatrix(z_axis, up, position)
    render_pose = np.linalg.inv(transform) @ render_pose
    render_pose[:3, 1:3] *= -1
    render_pose[:3, 3] /= scale
    render_poses.append(np.linalg.inv(render_pose))
  render_poses = np.stack(render_poses, axis=0)
  return render_poses


def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)

def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]

def transform_poses_pca(poses):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform
    return poses_recentered, transform

def generate_ellipse_path(views, n_frames=600, const_speed=True, z_variation=0., z_phase=0.):
    poses = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
    poses = np.stack(poses, 0)
    poses, transform = transform_poses_pca(poses)


    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0] , center[1],  0 ])
    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)

    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)


    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack([
            (low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5)),
            (low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5)),
            z_variation * (z_low[2] + (z_high - z_low)[2] *
                           (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
        ], -1)

    theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    if const_speed:
        # Resample theta angles so that the velocity is closer to constant.
        lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
        theta = sample_np(None, theta, np.log(lengths), n_frames + 1)
        positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])
    # up = normalize(poses[:, :3, 1].sum(0))

    render_poses = []
    for p in positions:
        render_pose = np.eye(4)
        render_pose[:3] = viewmatrix(p - center, up, p)
        render_pose = np.linalg.inv(transform) @ render_pose
        render_pose[:3, 1:3] *= -1
        render_poses.append(np.linalg.inv(render_pose))
    return render_poses



def generate_random_poses_llff(views):
    """Generates random poses."""
    n_poses = 10000 # args.n_random_poses
    poses, bounds = [], []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
        bounds.append(view.bounds)
    poses = np.stack(poses, 0)
    bounds = np.stack(bounds) # np.array([[ 16.21311152, 153.86329729]])

    scale = 1. / (bounds.min() * .75)
    poses[:, :3, 3] *= scale
    bounds *= scale
    poses, transform = recenter_poses(poses)

    # Find a reasonable 'focus depth' for this dataset as a weighted average
    # of near and far bounds in disparity space.
    close_depth, inf_depth = bounds.min() * .9, bounds.max() * 5.
    dt = .75
    focal = 1 / (((1 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), 100, 0)
    radii = np.concatenate([radii, [1.]])

    # Generate random poses.
    random_poses = []
    cam2world = poses_avg(poses)
    up = poses[:, :3, 1].mean(0)
    for _ in range(n_poses):
      t = radii * np.concatenate([2 * np.random.rand(3) - 1., [1,]])
      position = cam2world @ t
      lookat = cam2world @ [0, 0, -focal, 1.]
      z_axis = position - lookat
      random_pose = np.eye(4)
      random_pose[:3] = viewmatrix(z_axis, up, position)
      random_pose = np.linalg.inv(transform) @ random_pose
      random_pose[:3, 1:3] *= -1
      random_pose[:3, 3] /= scale
      random_poses.append(np.linalg.inv(random_pose))
    render_poses = np.stack(random_poses, axis=0)
    return render_poses



def generate_random_poses_360(views, n_frames=10000, z_variation=0.1, z_phase=0):
    poses = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
    poses = np.stack(poses, 0)
    poses, transform = transform_poses_pca(poses)


    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0] , center[1],  0 ])
    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)

    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)


    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack([
            (low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5)),
            (low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5)),
            z_variation * (z_low[2] + (z_high - z_low)[2] *
                           (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
        ], -1)

    theta = np.random.rand(n_frames) * 2. * np.pi
    positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])
    # up = normalize(poses[:, :3, 1].sum(0))

    render_poses = []
    for p in positions:
        render_pose = np.eye(4)
        render_pose[:3] = viewmatrix(p - center, up, p)
        render_pose = np.linalg.inv(transform) @ render_pose
        render_pose[:3, 1:3] *= -1
        render_poses.append(np.linalg.inv(render_pose))
    return render_poses

import numpy as np

def generate_random_poses_llff_aerial(
    views,
    n_poses=2000,
    off_nadir_deg_max=25.0,
    yaw_jitter_deg=180.0,
    altitude_percentile=(70, 95),
    radius_percentile=(60, 95),
    focus_depth_mode="harmonic",
    add_first_topdown=True
):
    """
    Generate LLFF-style random poses with aerial-view constraints.

    Cameras are sampled above the scene with controlled off-nadir angles and
    random yaw while preserving the existing LLFF pose convention.

    Args:
        views: Input camera records with R, T, and bounds fields.
        n_poses: Number of poses to generate.
        off_nadir_deg_max: Maximum optical-axis deviation from nadir.
        yaw_jitter_deg: Random yaw range in degrees.
        altitude_percentile: Percentile range for camera altitude sampling.
        radius_percentile: Percentile range for horizontal radius sampling.
        focus_depth_mode: Depth estimate mode: "harmonic", "median", or "center".
        add_first_topdown: Whether to seed the sequence with a near-nadir pose.

    Returns:
        Array of generated poses with shape [N, 4, 4].
    """

    # Convert input views to camera-to-world matrices in the LLFF convention.
    poses, bounds = [], []
    for view in views:
        tmp = np.eye(4)
        tmp[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)  # [R^T | T]
        tmp = np.linalg.inv(tmp)         # -> c2w
        tmp[:, 1:3] *= -1                # LLFF y/z axis flip.
        poses.append(tmp)
        bounds.append(view.bounds)
    poses = np.stack(poses, 0)           # [N,4,4]
    bounds = np.stack(bounds)            # [N,2] or [N,k]

    # Normalize and recenter poses using the LLFF preprocessing path.
    scale = 1. / (bounds.min() * 0.75)
    poses[:, :3, 3] *= scale
    bounds *= scale
    poses, transform = recenter_poses(poses)
    cam2world_avg = poses_avg(poses)
    up = poses[:, :3, 1].mean(0)
    up = up / (np.linalg.norm(up) + 1e-8)

    # Estimate a focal depth from the scene bounds.
    close_depth, inf_depth = bounds.min() * 0.9, bounds.max() * 5.0
    if focus_depth_mode == "harmonic":
        dt = 0.75
        focal_depth = 1.0 / (((1 - dt) / close_depth + dt / inf_depth))
    elif focus_depth_mode == "median":
        focal_depth = np.median(bounds) * 0.7
    else:  # "center"
        focal_depth = (close_depth + inf_depth) * 0.5
    focal_depth = float(max(focal_depth, 1e-3))

    # Derive aerial sampling ranges from the input camera distribution.
    centers = poses[:, :3, 3]
    horiz_r = np.linalg.norm(centers[:, [0, 2]], axis=1)
    r_min = np.percentile(horiz_r, radius_percentile[0])
    r_max = np.percentile(horiz_r, radius_percentile[1])

    heights = centers @ up
    h_min = np.percentile(heights, altitude_percentile[0])
    h_max = np.percentile(heights, altitude_percentile[1])
    h_min = h_min + 0.2 * (h_max - h_min)

    def sample_position():
        r = np.random.uniform(r_min, r_max)
        theta = np.random.uniform(-np.pi, np.pi)

        # Build two horizontal basis vectors orthogonal to the scene up vector.
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, up)) > 0.9:
            tmp = np.array([0.0, 0.0, 1.0])
        e1 = np.cross(up, tmp); e1 /= (np.linalg.norm(e1) + 1e-8)
        e2 = np.cross(up, e1);  e2 /= (np.linalg.norm(e2) + 1e-8)

        horiz = r * (np.cos(theta) * e1 + np.sin(theta) * e2)
        h = np.random.uniform(h_min, h_max)
        pos = cam2world_avg[:3, 3] + horiz + h * up
        return pos

    def sample_lookat():
        jitter = 0.02 * (r_max - r_min)
        j = (np.random.rand(3) - 0.5) * 2.0 * jitter
        return cam2world_avg[:3, 3] + j

    def build_pose(position, lookat):
        down_vec = -up
        to_target = (lookat - position); to_target /= (np.linalg.norm(to_target) + 1e-8)

        max_rad = np.deg2rad(off_nadir_deg_max)
        cosang = np.clip(np.dot(to_target, down_vec), -1.0, 1.0)
        ang = np.arccos(cosang)
        if ang > max_rad:
            axis = np.cross(to_target, down_vec)
            if np.linalg.norm(axis) < 1e-8:
                lookdir = down_vec
            else:
                axis /= np.linalg.norm(axis)
                t = max_rad / (ang + 1e-8)
                lookdir = (1 - t) * down_vec + t * to_target
                lookdir /= (np.linalg.norm(lookdir) + 1e-8)
        else:
            lookdir = to_target

        yaw = np.deg2rad(np.random.uniform(-yaw_jitter_deg, yaw_jitter_deg))
        right = np.cross(lookdir, up); n_right = np.linalg.norm(right)
        if n_right < 1e-8:
            right = np.array([1.0, 0.0, 0.0])
            right -= up * np.dot(right, up)
            right /= (np.linalg.norm(right) + 1e-8)
        else:
            right /= n_right
        cam_up = np.cross(right, lookdir); cam_up /= (np.linalg.norm(cam_up) + 1e-8)

        c, s = np.cos(yaw), np.sin(yaw)
        R_h = np.stack([right, cam_up, lookdir], axis=1)
        K = np.array([[0, -lookdir[2], lookdir[1]],
                      [lookdir[2], 0, -lookdir[0]],
                      [-lookdir[1], lookdir[0], 0]])
        R_yaw = np.eye(3) + s * K + (1 - c) * (K @ K)
        R = R_h @ R_yaw

        cam = np.eye(4)
        # cam[:3] = viewmatrix(lookat, up, position, subtract_position=True)

        # Convert back from normalized LLFF coordinates if this path is used.
        # cam = np.linalg.inv(transform) @ cam
        # cam[:3, 1:3] *= -1
        # cam[:3, 3] /= scale
        cam = np.eye(4)
        cam[:3,:3] = R
        cam[:3, 3]  = position

        return np.linalg.inv(cam)

    random_poses = []

    if add_first_topdown and n_poses > 0:
        pos0 = sample_position()
        lookat0 = sample_lookat()
        backup = off_nadir_deg_max
        off_nadir_deg_max_small = min(backup, 5.0)
        cam0 = build_pose(position=pos0, lookat=lookat0)
        random_poses.append(cam0)

    for _ in range(len(random_poses), n_poses):
        p = sample_position()
        la = sample_lookat()
        cam_pose = build_pose(position=p, lookat=la)
        random_poses.append(cam_pose)

    render_poses = np.stack(random_poses, axis=0)
    return render_poses
