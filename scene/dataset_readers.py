#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import glob
import os
import sys

from PIL import Image
from typing import NamedTuple, Optional
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import cv2
from tqdm import tqdm
from pathlib import Path
from plyfile import PlyData
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mask: np.array
    bounds: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    depth_informaiton: dict
    proposal_cloud: Optional[BasicPointCloud] = None

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}



def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, path, rgb_mapping, colmap_cam_extrinsics, colmap_cam_intrinsics):
    cam_infos = []
    for idx, key in enumerate(sorted(cam_extrinsics.keys())):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        bounds = np.load(os.path.join(path, 'poses_bounds.npy'))[idx, -2:]

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        rgb_path = rgb_mapping[idx]   # os.path.join(images_folder, rgb_mapping[idx])
        rgb_name = os.path.basename(rgb_path).split(".")[0]
        image = Image.open(rgb_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                image_name=image_name, width=width, height=height, mask=None, bounds=bounds)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    names = vertices.data.dtype.names

    def _get(name: str):
        return np.asarray(vertices[name]) if name in names else None

    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    color_arrays = [_get("red"), _get("green"), _get("blue")]
    if all(arr is not None for arr in color_arrays):
        colors = np.vstack(color_arrays).T / 255.0
    else:
        colors = np.zeros_like(positions, dtype=np.float32)

    normal_arrays = [_get("nx"), _get("ny"), _get("nz")]
    if all(arr is not None for arr in normal_arrays):
        normals = np.vstack(normal_arrays).T
    else:
        normals = np.zeros_like(positions, dtype=np.float32)

    confidence = _get("confidence")
    alpha = _get("alpha0")
    sigma_scale = _get("sigma_scale")
    proposal_id = _get("proposal_id")
    cooldown = _get("cooldown")

    # Ensure dtypes are consistent
    def _ensure_array(array, dtype):
        if array is None:
            return None
        arr = np.array(array, dtype=dtype, copy=True)
        return arr

    confidence = _ensure_array(confidence, np.float32)
    alpha = _ensure_array(alpha, np.float32)
    sigma_scale = _ensure_array(sigma_scale, np.float32)
    proposal_id = _ensure_array(proposal_id, np.int64)
    cooldown = _ensure_array(cooldown, np.float32)

    return BasicPointCloud(
        points=positions,
        colors=colors,
        normals=normals,
        confidence=confidence,
        alpha=alpha,
        sigma_scale=sigma_scale,
        proposal_id=proposal_id,
        cooldown=cooldown,
    )


# def storePly(path, xyz, rgb):
#     # Define the dtype for the structured array
#     dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
#             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
#             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

#     normals = np.zeros_like(xyz)

#     elements = np.empty(xyz.shape[0], dtype=dtype)
#     attributes = np.concatenate((xyz, normals, rgb), axis=1)
#     elements[:] = list(map(tuple, attributes))

#     # Create the PlyData object and write to file
#     vertex_element = PlyElement.describe(elements, 'vertex')
#     ply_data = PlyData([vertex_element])
#     ply_data.write(path)


def readColmapSceneInfo(path, images, eval, n_views=0, llffhold=8, colmap="sparse_vggt"):
    if 'levir' in path.lower():
        scene_name = path.split("/")[-1]
    elif 'rsscene' in path.lower():
        scene_name = '/'.join(path.split("/")[-2:])
    else:
        raise ValueError(f"Unsupported dataset: {path}")


    print("This is colmap:{}".format(colmap))
    print("This is the scene_name:{}".format(scene_name))
    if  colmap == "real_large_voxel_points":
        colmap_type = "real_large_voxel_points"
        ply_path = os.path.join(f"/YOU_PATH/INITIALIZATION/{scene_name}/points_dense_INIT_FINALLY.ply")
        proposal_path = os.path.join(
        f"/YOU_PATH/INITIALIZATION/{scene_name}/points_proposal_INIT_FINALLY.ply"
        )
        print("This is the ply_path:{}".format(ply_path))
        print("This is the proposal_path:{}".format(proposal_path))

    images_folder = path

    try:
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

        colmap_cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        colmap_cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        colmap_cam_extrinsics = read_extrinsics_binary(colmap_cameras_extrinsic_file)
        colmap_cam_intrinsics = read_intrinsics_binary(colmap_cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)


        colmap_cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        colmap_cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        colmap_cam_extrinsics = read_extrinsics_binary(colmap_cameras_extrinsic_file)
        colmap_cam_intrinsics = read_intrinsics_binary(colmap_cameras_intrinsic_file)

    depth_informaiton = None
    pcd = fetchPly(ply_path)
    proposal_cloud = None
    if os.path.exists(proposal_path):
        try:
            proposal_cloud = fetchPly(proposal_path)
            print(f"[Scene] Loaded proposal cloud: {proposal_path}")
        except Exception as exc:
            print(f"[Scene] Warning: failed to load proposal cloud ({proposal_path}): {exc}")


    reading_dir = "Images"
    rgb_mapping = [f for f in sorted(glob.glob(os.path.join(path, reading_dir, '*')))
                   if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    cam_extrinsics = {cam_extrinsics[k].name: cam_extrinsics[k] for k in cam_extrinsics}
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                             images_folder=os.path.join(path, reading_dir),  path=path, rgb_mapping=rgb_mapping, colmap_cam_extrinsics=colmap_cam_extrinsics, colmap_cam_intrinsics=colmap_cam_intrinsics)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        if 'levir' in path.lower():
            if n_views == 5:    
                train_cam_idx = [0, 5, 10, 15, 20]
            elif n_views == 3:
                train_cam_idx = [0, 7, 15]

            train_cam_infos = [cam_infos[i] for i in train_cam_idx]
            assert len(train_cam_infos) == n_views
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx not in train_cam_idx]
        
        elif 'rsscene' in path.lower():
            train_idx = [1, 11, 23, 34, 46, 58, 69]
            test_idx = [0,7,14,21,28,35,42,49,56,63]
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in train_idx]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_idx]    
            assert len(train_cam_infos) == n_views       
        else:
            raise ValueError(f"Unsupported dataset: {path}")
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    
    nerf_normalization = getNerfppNorm(train_cam_infos)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           depth_informaiton=depth_informaiton,
                           proposal_cloud=proposal_cloud)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        skip = 8 if transformsfile == 'transforms_test.json' else 1
        frames = contents["frames"][::skip]
        for idx, frame in tqdm(enumerate(frames)):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            mask = norm_data[:, :, 3:4]
            if skip == 1:
                depth_image = np.load('../SparseNeRF/depth_midas_temp_DPT_Hybrid/Blender/' +
                                      image_path.split('/')[-4]+'/'+image_name+'_depth.npy')
            else:
                depth_image = None

            arr = cv2.resize(arr, (400, 400))
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
            depth_image = None if depth_image is None else cv2.resize(depth_image, (400, 400))
            mask = None if mask is None else cv2.resize(mask, (400, 400))


            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                                        image_name=image_name, width=image.size[0], height=image.size[1],
                                        depth_image=depth_image, mask=mask))
    return cam_infos



def readNerfSyntheticInfo(path, white_background, eval, n_views=0, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    pseudo_cam_infos = train_cam_infos #train_cam_infos
    if n_views > 0:
        train_cam_infos = train_cam_infos[:n_views]
        assert len(train_cam_infos) == n_views

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, str(n_views) + "_views/dense/fused.ply")


    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           pseudo_cameras=pseudo_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
