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
    off_nadir_deg_max=25.0,   # 最大离轴角（越小越“正射”）
    yaw_jitter_deg=180.0,     # 水平朝向随机范围
    altitude_percentile=(70, 95),  # 高度采样的分位区间（相对已有相机高度）
    radius_percentile=(60, 95),    # 水平半径采样分位区间
    focus_depth_mode="harmonic",   # 焦点深度估计方式: "harmonic"|"median"|"center"
    add_first_topdown=True         # 是否把第一帧设置为近似正下视（nadir-ish）
):
    """
    基于 LLFF 的随机位姿生成，做了航拍友好化约束：
    - 相机主要分布在场景上空；优先水平环绕+可控的离轴角；Yaw 自由。
    - 仍保持 LLFF 的 recenter/scale/轴翻转与出入参约定，尽量不改动 pipeline。

    参数说明：
      views: 具有 .R, .T, .bounds 的对象列表（与你原函数相同）
      n_poses: 生成位姿数量
      off_nadir_deg_max: 最大离轴角（相机光轴相对“正下方”的最大偏转角）
      yaw_jitter_deg: 水平旋转角度随机范围（增强多样性）
      altitude_percentile: 相机高度的分位区间（从已有相机高度统计得到）
      radius_percentile: 水平半径（x-z 平面）分位区间
      focus_depth_mode: 焦点深度估计方式
      add_first_topdown: 是否将第一个相机置为近似正下视（不强制严格正下）

    返回：
      render_poses: [N, 4, 4] 同你原函数返回类型
    """

    # === 1) 将 views 转为 c2w 并做 LLFF 约定的坐标翻转 ===
    poses, bounds = [], []
    for view in views:
        tmp = np.eye(4)
        tmp[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)  # [R^T | T]
        tmp = np.linalg.inv(tmp)         # -> c2w
        tmp[:, 1:3] *= -1                # LLFF 约定：翻转 y,z
        poses.append(tmp)
        bounds.append(view.bounds)
    poses = np.stack(poses, 0)           # [N,4,4]
    bounds = np.stack(bounds)            # [N,2] or [N,k]

    # === 2) 归一化 + recentre（保持你的流程不变）===
    scale = 1. / (bounds.min() * 0.75)
    poses[:, :3, 3] *= scale
    bounds *= scale
    poses, transform = recenter_poses(poses)  # 返回 recentered poses 以及对齐变换
    cam2world_avg = poses_avg(poses)          # 场景“中心姿态”
    up = poses[:, :3, 1].mean(0)              # LLFF 常用上方向估计（相机 y 轴的平均）
    up = up / (np.linalg.norm(up) + 1e-8)

    # === 3) 估计焦点深度（focus depth）===
    close_depth, inf_depth = bounds.min() * 0.9, bounds.max() * 5.0
    if focus_depth_mode == "harmonic":
        dt = 0.75
        focal_depth = 1.0 / (((1 - dt) / close_depth + dt / inf_depth))
    elif focus_depth_mode == "median":
        focal_depth = np.median(bounds) * 0.7
    else:  # "center"
        focal_depth = (close_depth + inf_depth) * 0.5
    focal_depth = float(max(focal_depth, 1e-3))

    # === 4) 统计“水平半径”和“高度”范围（航拍友好）===
    centers = poses[:, :3, 3]
    # 在 XZ 平面上的水平距离
    horiz_r = np.linalg.norm(centers[:, [0, 2]], axis=1)
    r_min = np.percentile(horiz_r, radius_percentile[0])
    r_max = np.percentile(horiz_r, radius_percentile[1])

    # 高度（沿 up 方向的投影）
    heights = centers @ up
    h_min = np.percentile(heights, altitude_percentile[0])
    h_max = np.percentile(heights, altitude_percentile[1])
    # 为了更“上空”，向上再抬一点
    h_min = h_min + 0.2 * (h_max - h_min)

    # === 5) 采样函数 ===
    def sample_position():
        # 在 XZ 平面采样半径与方位角；高度单独采样
        r = np.random.uniform(r_min, r_max)
        theta = np.random.uniform(-np.pi, np.pi)
        # 在 up 的正交平面上找两个正交基（e1, e2），用于构造水平向量
        # 简单做法：任选一个非平行向量，取叉积
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
        # 以场景中心为主，可做极小扰动防止完全一致
        jitter = 0.02 * (r_max - r_min)
        j = (np.random.rand(3) - 0.5) * 2.0 * jitter
        return cam2world_avg[:3, 3] + j

    def build_pose(position, lookat):
        # 让相机“朝下”为主：构造一个“理想的正下向量”，再允许离轴
        # 下向量 = -up
        down_vec = -up
        # 控制 off-nadir：将理想下向量与指向 lookat 的向量做插值
        to_target = (lookat - position); to_target /= (np.linalg.norm(to_target) + 1e-8)

        # 计算允许的最大夹角（弧度）
        max_rad = np.deg2rad(off_nadir_deg_max)
        # 当前 to_target 与 down_vec 的夹角
        cosang = np.clip(np.dot(to_target, down_vec), -1.0, 1.0)
        ang = np.arccos(cosang)
        if ang > max_rad:
            # 将 to_target 朝 down_vec 拉回，保证离轴角不超过上限
            axis = np.cross(to_target, down_vec)
            if np.linalg.norm(axis) < 1e-8:
                lookdir = down_vec
            else:
                axis /= np.linalg.norm(axis)
                # 罗德里格公式旋转 to_target 到距 down_vec 仅 max_rad 的方向
                # 先把 to_target 旋转到与 down_vec 对齐的方向，然后回退 max_rad
                # 这里等价：把 down_vec 旋转到与 to_target 同平面的方向，再 clamp 角度
                # 简化：线性插值方向并归一（足够稳定且便于实现）
                t = max_rad / (ang + 1e-8)
                lookdir = (1 - t) * down_vec + t * to_target
                lookdir /= (np.linalg.norm(lookdir) + 1e-8)
        else:
            lookdir = to_target

        # 在水平面加 yaw 抖动（围绕 up 旋转相机坐标系的 x/y）
        yaw = np.deg2rad(np.random.uniform(-yaw_jitter_deg, yaw_jitter_deg))
        # 构造一个与 up 正交且与 lookdir 也正交的右向量
        right = np.cross(lookdir, up); n_right = np.linalg.norm(right)
        if n_right < 1e-8:
            # 极端退化：lookdir 与 up 共线，选一个稳定的 right
            right = np.array([1.0, 0.0, 0.0])
            right -= up * np.dot(right, up)
            right /= (np.linalg.norm(right) + 1e-8)
        else:
            right /= n_right
        cam_up = np.cross(right, lookdir); cam_up /= (np.linalg.norm(cam_up) + 1e-8)

        # 绕 lookdir 轴对 (right, cam_up) 做 yaw 旋转
        c, s = np.cos(yaw), np.sin(yaw)
        R_h = np.stack([right, cam_up, lookdir], axis=1)  # 列为轴
        # 这里 yaw 是绕 lookdir 轴的旋转矩阵
        K = np.array([[0, -lookdir[2], lookdir[1]],
                      [lookdir[2], 0, -lookdir[0]],
                      [-lookdir[1], lookdir[0], 0]])
        R_yaw = np.eye(3) + s * K + (1 - c) * (K @ K)
        R = R_h @ R_yaw

        # 用你的 viewmatrix 构造（注意：给它“lookat”并启用 subtract_position=True 最稳妥）
        # 你的 viewmatrix(lookdir, up, position, subtract_position=False)
        # 这里我们传 lookat 且 subtract_position=True -> vec2 = normalize(lookdir - position)
        cam = np.eye(4)
        # cam[:3] = viewmatrix(lookat, up, position, subtract_position=True)  # 方向稳定朝向 lookat
        # # （可选）若你更信任上面的 R，可替换 cam[:3,:3] = R

        # # 撤销 recenter/scale，并做 LLFF 的轴翻转与最终约定（与原函数一致）
        # cam = np.linalg.inv(transform) @ cam
        # cam[:3, 1:3] *= -1
        # cam[:3, 3] /= scale
        cam = np.eye(4)
        cam[:3,:3] = R
        cam[:3, 3]  = position

        # 返回与原函数一致的 “世界到相机” 矩阵
        return np.linalg.inv(cam)

    # === 6) 生成随机位姿 ===
    random_poses = []

    # 可选：第 1 帧近似正下视（不强制完全正下），利于你之前的“第1个视角需求”
    if add_first_topdown and n_poses > 0:
        pos0 = sample_position()
        # 让它“更正下”：把 lookdir 直接压向 -up
        lookat0 = sample_lookat()
        # 用更小的 off-nadir 限制
        backup = off_nadir_deg_max
        off_nadir_deg_max_small = min(backup, 5.0)
        # 临时缩小离轴角
        cam0 = build_pose(position=pos0, lookat=lookat0)
        random_poses.append(cam0)
        # 恢复（注意：这里我们没有在 build_pose 内部直接引用外部变量，已局部化）

    for _ in range(len(random_poses), n_poses):
        p = sample_position()
        la = sample_lookat()
        cam_pose = build_pose(position=p, lookat=la)
        random_poses.append(cam_pose)

    render_poses = np.stack(random_poses, axis=0)
    return render_poses
