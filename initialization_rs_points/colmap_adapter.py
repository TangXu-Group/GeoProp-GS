from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .colmap_reader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat


@dataclass
class CameraPose:
    name: str
    image_id: int
    camera_id: int
    R_wc: np.ndarray
    t_wc: np.ndarray
    K: np.ndarray
    width: int
    height: int

    @property
    def camera_center(self) -> np.ndarray:
        R_cw = self.R_wc.T
        return -R_cw @ self.t_wc


def load_camera_poses(model_dir: Path) -> Dict[str, CameraPose]:
    model_dir = Path(model_dir)
    images_path = model_dir / "images.bin"
    cameras_path = model_dir / "cameras.bin"
    if not images_path.exists():
        raise FileNotFoundError(f"Missing COLMAP images.bin at {images_path}")
    if not cameras_path.exists():
        raise FileNotFoundError(f"Missing COLMAP cameras.bin at {cameras_path}")

    cameras = read_intrinsics_binary(str(cameras_path))
    images = read_extrinsics_binary(str(images_path))

    poses: Dict[str, CameraPose] = {}
    for image_id, image in images.items():
        camera = cameras[image.camera_id]
        fx, fy, cx, cy = _intr_from_camera(camera)
        K = np.array([[fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0]], dtype=np.float32)
        R_wc = qvec2rotmat(image.qvec).astype(np.float32)
        t_wc = np.asarray(image.tvec, dtype=np.float32)
        poses[image.name] = CameraPose(
            name=image.name,
            image_id=image_id,
            camera_id=image.camera_id,
            R_wc=R_wc,
            t_wc=t_wc,
            K=K,
            width=int(camera.width),
            height=int(camera.height),
        )
    return poses


def resolve_visible_poses(
    poses: Dict[str, CameraPose],
    visible_views: Sequence[str],
) -> List[CameraPose]:
    visible: List[CameraPose] = []
    missing: List[str] = []
    for name in visible_views:
        key = _normalize_view_name(name)
        pose = poses.get(name) or poses.get(key)
        if pose is None:
            missing.append(name)
            continue
        visible.append(pose)
    if missing:
        raise KeyError(f"Views not found in COLMAP model: {missing}")

    visible.sort(key=lambda p: p.name)
    return visible


def _intr_from_camera(camera) -> Tuple[float, float, float, float]:
    params = camera.params
    if len(params) >= 4:
        fx, fy, cx, cy = map(float, params[:4])
    elif len(params) == 3:
        fx = fy = float(params[0])
        cx = float(params[1])
        cy = float(params[2])
    else:
        raise ValueError(f"Unsupported intrinsic parameter length: {len(params)}")
    return fx, fy, cx, cy


def _normalize_view_name(name: str) -> str:
    stem = Path(name).stem
    return f"{stem}.png"
