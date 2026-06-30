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

import os
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.pose_utils import generate_random_poses_llff, generate_random_poses_360
from scene.cameras import PseudoCamera
import struct


def _preview_ply(path: str, limit: int = 5):
    if not path or not os.path.exists(path):
        print(f"[Preview] Skipped: {path or 'None'} (missing)")
        return
    try:
        with open(path, "rb") as f:
            header = []
            while True:
                line = f.readline()
                if not line:
                    raise RuntimeError("Unexpected EOF inside PLY header")
                header.append(line.decode("ascii", errors="ignore").strip())
                if line.strip() == b"end_header":
                    break
        vertex_count = None
        props = []
        for line in header:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                if len(parts) >= 3 and parts[1] != "list":
                    props.append((parts[1], parts[2]))
        if vertex_count is None:
            print(f"[Preview] {path}: no vertex element found")
            return
        type_map = {
            "float": ("f", 4),
            "double": ("d", 8),
            "uchar": ("B", 1),
            "uint8": ("B", 1),
            "int": ("i", 4),
            "uint": ("I", 4),
            "short": ("h", 2),
            "ushort": ("H", 2),
        }
        fmt = "<" + "".join(type_map.get(t, ("f", 4))[0] for t, _ in props)
        stride = sum(type_map.get(t, ("f", 4))[1] for t, _ in props)
        records = []
        with open(path, "rb") as f:
            while True:
                line = f.readline()
                if line.strip() == b"end_header":
                    break
            for _ in range(min(limit, vertex_count)):
                raw = f.read(stride)
                if len(raw) != stride:
                    break
                values = struct.unpack(fmt, raw)
                record = {
                    name: (round(values[i], 6) if isinstance(values[i], float) else values[i])
                    for i, (_, name) in enumerate(props)
                }
                records.append(record)
        print(f"[Preview] File: {path}")
        print(f"[Preview] Vertices: {vertex_count}")
        print(f"[Preview] Properties: {[name for _, name in props]}")
        for idx, rec in enumerate(records):
            print(f"[Preview]   #{idx}: {rec}")
    except Exception as exc:
        print(f"[Preview] Failed to read {path}: {exc}")


def _preview_cloud(name: str, cloud, limit: int = 5):
    if cloud is None or getattr(cloud, "points", None) is None:
        print(f"[Preview] {name}: empty")
        return
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors) if getattr(cloud, "colors", None) is not None else None
    print(f"[Preview] {name}: {points.shape[0]} points")
    lim = min(limit, points.shape[0])
    for idx in range(lim):
        pt = points[idx]
        rec = {"x": round(float(pt[0]), 6), "y": round(float(pt[1]), 6), "z": round(float(pt[2]), 6)}
        if colors is not None and colors.shape[0] == points.shape[0]:
            rec["rgb"] = [int(c) for c in colors[idx].tolist()]
        print(f"[Preview]   #{idx}: {rec}")

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.pseudo_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.n_views, colmap=args.colmap)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.n_views)
        else:
            assert False, "Could not recognize scene type!"
        proposal_cloud = getattr(scene_info, "proposal_cloud", None)
        self.depth_informaiton = scene_info.depth_informaiton
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(self.cameras_extent, 'cameras_extent')

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

            pseudo_cams = []
            # print(args.source_path.find('llff'))

            if args.source_path.find('llff') + 1:
                pseudo_poses = generate_random_poses_llff(self.train_cameras[resolution_scale])
            elif args.source_path.find('360') + 1:
                pseudo_poses = generate_random_poses_360(self.train_cameras[resolution_scale])
            elif args.source_path.find('Levir') + 1 or args.source_path.find('LEVIR') + 1:
                print('generate_random_poses_llff_aerial')  
                pseudo_poses = generate_random_poses_llff(self.train_cameras[resolution_scale])
            elif args.source_path.find('rsscene') + 1:
                print('generate_random_poses_rsscene')  
                pseudo_poses = generate_random_poses_llff(self.train_cameras[resolution_scale])

            view = self.train_cameras[resolution_scale][0]
            for pose in pseudo_poses:
                pseudo_cams.append(PseudoCamera(
                    R=pose[:3, :3].T, T=pose[:3, 3], FoVx=view.FoVx, FoVy=view.FoVy,
                    width=view.image_width, height=view.image_height
                ))
            self.pseudo_cameras[resolution_scale] = pseudo_cams
            # self.pseudo_cameras[resolution_scale] = self.test_cameras[resolution_scale]

        proposals_enabled = not getattr(args, "disable_proposals", False)
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            if proposals_enabled and proposal_cloud is not None and getattr(proposal_cloud, "points", None) is not None and proposal_cloud.points.size > 0:
                self.gaussians.append_proposals(proposal_cloud)
        if getattr(args, "preview_initial_ply", False):
            preview_limit = int(getattr(args, "preview_ply_limit", 5))
            print("[Preview] === Dataset fused cloud sample ===")
            _preview_cloud("Dataset fused cloud", scene_info.point_cloud, preview_limit)
            dataset_ply = getattr(scene_info, "ply_path", None)
            if dataset_ply:
                _preview_ply(dataset_ply, preview_limit)
            run_input_ply = os.path.join(self.model_path, "input.ply")
            print("[Preview] === Run input.ply sample ===")
            _preview_ply(run_input_ply, preview_limit)
            if proposals_enabled and proposal_cloud is not None and proposal_cloud.points.size > 0:
                print("[Preview] === Proposal cloud sample ===")
                _preview_cloud("Proposal cloud", proposal_cloud, preview_limit)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getPseudoCameras(self, scale=1.0):
        if len(self.pseudo_cameras) == 0:
            return [None]
        else:
            return self.pseudo_cameras[scale]
