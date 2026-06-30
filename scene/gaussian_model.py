# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, chamfer_dist


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, args):
        self.args = args
        self.active_sh_degree = 0
        self.max_sh_degree = args.sh_degree
        self.init_point = torch.empty(0)
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.bg_color = torch.empty(0)
        self.confidence = torch.empty(0)
        self.opacity_scale = torch.empty(0)
        self._is_proposal = torch.empty(0, dtype=torch.bool)
        self.lr_multipliers = torch.empty(0)
        self.proposal_lr_multiplier = getattr(args, "proposal_lr_multiplier", 0.3)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.confidence,
            self.opacity_scale,
            self._is_proposal,
            self.lr_multipliers,
        )

    def restore(self, model_args, training_args):
        self.active_sh_degree = model_args[0]
        self._xyz = model_args[1]
        self._features_dc = model_args[2]
        self._features_rest = model_args[3]
        self._scaling = model_args[4]
        self._rotation = model_args[5]
        self._opacity = model_args[6]
        self.max_radii2D = model_args[7]
        xyz_gradient_accum = model_args[8]
        denom = model_args[9]
        opt_dict = model_args[10]
        self.spatial_lr_scale = model_args[11]

        extras = model_args[12:]
        device = self._opacity.device

        default_confidence = torch.ones((self._opacity.shape[0], 1), dtype=torch.float32, device=device)
        default_opacity_scale = torch.ones((self._opacity.shape[0], 1), dtype=torch.float32, device=device)
        default_is_proposal = torch.zeros((self._opacity.shape[0],), dtype=torch.bool, device=device)
        default_lr = torch.ones((self._opacity.shape[0], 1), dtype=torch.float32, device=device)

        if len(extras) >= 4:
            default_confidence = extras[0].to(device).view(-1, 1)
            default_opacity_scale = extras[1].to(device).view(-1, 1)
            default_is_proposal = extras[2].to(device=device, dtype=torch.bool).view(-1)
            default_lr = extras[3].to(device).view(-1, 1)
        elif len(extras) == 3:
            default_confidence = extras[0].to(device).view(-1, 1)
            default_opacity_scale = extras[1].to(device).view(-1, 1)
            default_is_proposal = extras[2].to(device=device, dtype=torch.bool).view(-1)
        elif len(extras) == 2:
            default_confidence = extras[0].to(device).view(-1, 1)
            default_opacity_scale = extras[1].to(device).view(-1, 1)

        self.training_setup(training_args)

        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.confidence = default_confidence
        self.opacity_scale = default_opacity_scale
        self._is_proposal = default_is_proposal
        self.lr_multipliers = default_lr
        # self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        _ = self.rotation_activation(self._rotation)
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def is_proposal(self):
        return self._is_proposal

    @property
    def get_opacity_scale(self):
        if self.opacity_scale.numel() == 0:
            device = self._opacity.device if isinstance(self._opacity, torch.Tensor) else torch.device("cuda")
            self.opacity_scale = torch.ones((self.get_xyz.shape[0], 1), device=device)
        return self.opacity_scale

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).cuda().float()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        features = torch.zeros(
            (fused_point_cloud.shape[0], 3, (self.max_sh_degree + 1) ** 2)
        ).float().cuda()
        if self.args.use_color:
            features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        self.init_point = fused_point_cloud

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud)[0], 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.confidence = torch.ones_like(opacities, device="cuda")
        self.opacity_scale = torch.ones_like(opacities, device="cuda")
        self._is_proposal = torch.zeros(
            (fused_point_cloud.shape[0],), dtype=torch.bool, device="cuda"
        )
        self.lr_multipliers = torch.ones(
            (fused_point_cloud.shape[0], 1), dtype=torch.float32, device="cuda"
        )
        if self.args.train_bg:
            self.bg_color = nn.Parameter(
                (torch.zeros(3, 1, 1) + 0.0).cuda().requires_grad_(True)
            )

    def append_proposals(self, proposal: BasicPointCloud):
        print("[proposal-init] append_proposals called")
        if proposal is None or getattr(proposal, "points", None) is None:
            return
        points = np.asarray(proposal.points)
        if points.size == 0:
            return

        device = (
            self._xyz.device
            if isinstance(self._xyz, torch.Tensor) and self._xyz.numel() > 0
            else torch.device("cuda")
        )
        positions = torch.tensor(points, dtype=torch.float32, device=device)
        count = positions.shape[0]

        colors_np = (
            np.asarray(proposal.colors)
            if getattr(proposal, "colors", None) is not None
            else None
        )
        base_xyz = self.get_xyz.detach() if isinstance(self.get_xyz, torch.Tensor) else torch.empty(0)
        base_features_dc = (
            self._features_dc.detach()
            if isinstance(self._features_dc, torch.Tensor)
            else torch.empty(0)
        )
        base_features_rest = (
            self._features_rest.detach()
            if isinstance(self._features_rest, torch.Tensor)
            else torch.empty(0)
        )
        has_anchor_colors = base_xyz.numel() > 0 and base_features_dc.numel() > 0
 
        distilled_rest = None
        if colors_np is None or colors_np.shape[0] != count:
            sh_dc = torch.zeros((count, 3), dtype=torch.float32, device=device)
        else:
            colors_np = colors_np.astype(np.float32, copy=False)
            if colors_np.max() > 1.0 + 1e-3:
                colors_np = colors_np / 255.0
            colors = torch.from_numpy(colors_np).to(device=device).clamp_(0.0, 1.0)
            sh_dc = RGB2SH(colors)

        features_dc = torch.zeros((count, 1, 3), dtype=torch.float32, device=device)
        features_dc[:, 0, :] = sh_dc
        if getattr(self.args, "debug_proposal_color_init", False):
            print(f"[proposal-init] new_dc_mean={features_dc[:, 0, :].mean().item():.6f}")
        rest_dim = (self.max_sh_degree + 1) ** 2 - 1
        if rest_dim > 0:
            if distilled_rest is not None:
                features_rest = distilled_rest
            else:
                features_rest = torch.zeros(
                    (count, rest_dim, 3), dtype=torch.float32, device=device
                )
        else:
            features_rest = torch.zeros(
                (count, 0, 3), dtype=torch.float32, device=device
            )

        alpha_np = (
            np.asarray(proposal.alpha)
            if getattr(proposal, "alpha", None) is not None
            else None
        )
        # if alpha_np is not None and alpha_np.shape[0] == count:
        #     alpha = torch.tensor(alpha_np, dtype=torch.float32, device=device)
        #     alpha = torch.clamp(alpha, min=0.1, max=0.95)
        # else:
        #     alpha = torch.full((count,), 0.5, dtype=torch.float32, device=device)
        alpha = torch.full((count,), 0.1, dtype=torch.float32, device=device)
        opacities = inverse_sigmoid(alpha.view(-1, 1))

        sigma_np = (
            np.asarray(proposal.sigma_scale)
            if getattr(proposal, "sigma_scale", None) is not None
            else None
        )
        if sigma_np is None or sigma_np.shape[0] != count:
            sigma_np = np.full((count,), 0.02, dtype=np.float32)
        sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device).clamp_(min=1e-5)
        ratio = float(getattr(self.args, "proposal_sigma_ratio", 0.05))
        if ratio > 0:
            sigma = sigma * ratio
        if count > 1:
            nn_dist = torch.sqrt(torch.clamp_min(distCUDA2(positions)[0], 1e-8))
            sigma = torch.min(sigma, nn_dist * 0.5)
        scene_extent = (
            float(self.spatial_lr_scale)
            if self.spatial_lr_scale and self.spatial_lr_scale > 0
            else None
        )
        max_thresh_ratio = float(getattr(self.args, "proposal_scale_threshold", 0.3))
        if scene_extent is not None and max_thresh_ratio > 0:
            sigma = torch.clamp(sigma, max=scene_extent * max_thresh_ratio)
        sigma = torch.clamp(sigma, min=1e-5)
        scaling = torch.log(sigma.view(-1, 1).repeat(1, 3))

        rotations = torch.zeros((count, 4), dtype=torch.float32, device=device)
        rotations[:, 0] = 1.0

        confidence_np = (
            np.asarray(proposal.confidence)
            if getattr(proposal, "confidence", None) is not None
            else None
        )
        if confidence_np is None or confidence_np.shape[0] != count:
            confidence_np = np.full((count,), 0.3, dtype=np.float32)
        confidence = torch.tensor(confidence_np, dtype=torch.float32, device=device).view(
            -1, 1
        )

        opacity_scale = torch.ones((count, 1), dtype=torch.float32, device=device)
        lr_values = torch.full(
            (count, 1),
            float(self.proposal_lr_multiplier),
            dtype=torch.float32,
            device=device,
        )
        proposal_flags = torch.ones((count,), dtype=torch.bool, device=device)

        def _stack_param(param: torch.Tensor, addition: torch.Tensor) -> nn.Parameter:
            if param.numel() == 0:
                return nn.Parameter(addition.requires_grad_(True))
            combined = torch.cat([param.detach(), addition], dim=0)
            return nn.Parameter(combined.requires_grad_(True))

        if self.optimizer is None:
            self._xyz = _stack_param(self._xyz, positions)
            self._features_dc = _stack_param(self._features_dc, features_dc)
            self._features_rest = _stack_param(self._features_rest, features_rest)
            self._opacity = _stack_param(self._opacity, opacities)
            self._scaling = _stack_param(self._scaling, scaling)
            self._rotation = _stack_param(self._rotation, rotations)
        else:
            additions = {
                "xyz": positions,
                "f_dc": features_dc,
                "f_rest": features_rest,
                "opacity": opacities,
                "scaling": scaling,
                "rotation": rotations,
            }
            optimizable_tensors = self.cat_tensors_to_optimizer(additions)
            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]

        if self.init_point.numel() == 0:
            self.init_point = positions.detach().clone()
        else:
            self.init_point = torch.cat(
                [self.init_point, positions.detach().clone()], dim=0
            )

        if self.xyz_gradient_accum.numel() == 0:
            self.xyz_gradient_accum = torch.zeros(
                (self._xyz.shape[0], 1), dtype=torch.float32, device=device
            )
            self.denom = torch.zeros(
                (self._xyz.shape[0], 1), dtype=torch.float32, device=device
            )
        else:
            pad = torch.zeros((count, 1), dtype=torch.float32, device=device)
            self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, pad], dim=0)
            self.denom = torch.cat([self.denom, pad.clone()], dim=0)

        if self.max_radii2D.numel() == 0:
            self.max_radii2D = torch.zeros(
                (self._xyz.shape[0]), dtype=torch.float32, device=device
            )
        else:
            self.max_radii2D = torch.cat(
                [
                    self.max_radii2D,
                    torch.zeros((count,), dtype=torch.float32, device=device),
                ],
                dim=0,
            )

        self.confidence = (
            torch.cat([self.confidence, confidence], dim=0)
            if self.confidence.numel() > 0
            else confidence
        )
        self.opacity_scale = (
            torch.cat([self.opacity_scale, opacity_scale], dim=0)
            if self.opacity_scale.numel() > 0
            else opacity_scale
        )
        self._is_proposal = (
            torch.cat([self._is_proposal, proposal_flags], dim=0)
            if self._is_proposal.numel() > 0
            else proposal_flags
        )
        self.lr_multipliers = (
            torch.cat([self.lr_multipliers.view(-1, 1), lr_values], dim=0)
            if self.lr_multipliers.numel() > 0
            else lr_values
        )


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1), device="cuda"
        )
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        device = (
            self._xyz.device
            if isinstance(self._xyz, torch.Tensor)
            else torch.device("cuda")
        )

        if self.confidence.numel() == 0:
            self.confidence = torch.ones((self.get_xyz.shape[0], 1), device=device)
        else:
            self.confidence = self.confidence.to(device)

        if self.opacity_scale.numel() == 0:
            self.opacity_scale = torch.ones(
                (self.get_xyz.shape[0], 1), device=device
            )
        else:
            self.opacity_scale = self.opacity_scale.to(device)

        if self.lr_multipliers.numel() == 0:
            self.lr_multipliers = torch.ones(
                (self.get_xyz.shape[0], 1), device=device
            )
        else:
            self.lr_multipliers = self.lr_multipliers.to(device).view(-1, 1)

        if self._is_proposal.numel() == 0:
            self._is_proposal = torch.zeros(
                (self.get_xyz.shape[0],), dtype=torch.bool, device=device
            )
        else:
            self._is_proposal = self._is_proposal.to(device=device)

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]
        if self.args.train_bg:
            l.append(
                {"params": [self.bg_color], "lr": 0.001, "name": "bg_color"}
            )

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def apply_proposal_gradient_scale(self):
        if self.lr_multipliers.numel() == 0:
            return
        if self.lr_multipliers.shape[0] != self._xyz.shape[0]:
            return
        lr = self.lr_multipliers
        if self._xyz.grad is not None:
            self._xyz.grad.mul_(lr)
        if self._features_dc.grad is not None:
            self._features_dc.grad.mul_(lr.view(-1, 1, 1))
        if self._features_rest.grad is not None and self._features_rest.grad.numel() > 0:
            self._features_rest.grad.mul_(lr.view(-1, 1, 1))
        if self._opacity.grad is not None:
            self._opacity.grad.mul_(lr)
        if self._scaling.grad is not None:
            self._scaling.grad.mul_(lr)
        if self._rotation.grad is not None:
            self._rotation.grad.mul_(lr)

    def proposal_dc_stat_regularizer(self, mean_weight: float = 0.0, var_weight: float = 0.0):
        mean_weight = float(mean_weight)
        var_weight = float(var_weight)
        if mean_weight <= 0.0 and var_weight <= 0.0:
            return self._xyz.new_tensor(0.0)
        if self._features_dc.numel() == 0 or self._is_proposal.numel() == 0:
            return self._xyz.new_tensor(0.0)

        proposal_mask = self._is_proposal.bool()
        if proposal_mask.sum() == 0:
            return self._xyz.new_tensor(0.0)

        dc_all = self._features_dc[:, 0, :]
        prop_dc = dc_all[proposal_mask]
        if prop_dc.numel() == 0:
            return self._xyz.new_tensor(0.0)

        loss = 0.0

        if mean_weight > 0.0:
            global_mean = dc_all.mean(dim=0)
            prop_mean = prop_dc.mean(dim=0)
            mean_diff = (prop_mean - global_mean).pow(2).mean()
            loss = loss + mean_weight * mean_diff

        if var_weight > 0.0:
            global_var = dc_all.var(dim=0, unbiased=False).clamp_min(1e-6)
            prop_var = prop_dc.var(dim=0, unbiased=False).clamp_min(1e-6)
            var_diff = (prop_var - global_var).pow(2).mean()
            loss = loss + var_weight * var_diff

        return loss if isinstance(loss, torch.Tensor) else self._xyz.new_tensor(loss)

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        xyz_lr = self.xyz_scheduler_args(iteration)
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group["lr"] = xyz_lr
                return xyz_lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        l.append("confidence")
        l.append("opacity_scale")
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        confidence = self.confidence.detach().cpu().numpy()
        opacity_scale = self.opacity_scale.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (
                xyz,
                normals,
                f_dc,
                f_rest,
                opacities,
                scale,
                rotation,
                confidence,
                opacity_scale,
            ),
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        if self._opacity.numel() == 0:
            return

        current_opacity = self.get_opacity.detach()
        target = torch.min(current_opacity, torch.ones_like(current_opacity) * 0.05)

        if self._is_proposal.numel() == current_opacity.shape[0]:
            reset_mask = (~self._is_proposal).view(-1, 1)
            if reset_mask.sum() == 0:
                return
        else:
            reset_mask = torch.ones_like(current_opacity, dtype=torch.bool)

        opacities_new = self._opacity.detach().clone()
        opacities_new[reset_mask] = inverse_sigmoid(target[reset_mask])

        if len(self.optimizer.state.keys()):
            optimizable_tensors = self.replace_tensor_to_optimizer(
                opacities_new, "opacity"
            )
            self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        property_names = [p.name for p in plydata.elements[0].properties]

        def _maybe(name, default):
            if name in property_names:
                return np.asarray(plydata.elements[0][name])
            return default

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.active_sh_degree = self.max_sh_degree
        confidence = _maybe("confidence", np.ones((xyz.shape[0],), dtype=np.float32))
        opacity_scale = _maybe(
            "opacity_scale", np.ones((xyz.shape[0],), dtype=np.float32)
        )

        device = torch.device("cuda")
        self.confidence = torch.tensor(
            confidence, dtype=torch.float32, device=device
        ).view(-1, 1)
        self.opacity_scale = torch.tensor(
            opacity_scale, dtype=torch.float32, device=device
        ).view(-1, 1)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["bg_color", "view_transformer"]:
                continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def dist_prune(self):
        dist = chamfer_dist(self.init_point, self._xyz)
        valid_points_mask = dist < 3.0
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.confidence = self.confidence[valid_points_mask]
        self.opacity_scale = self.opacity_scale[valid_points_mask]
        if self._is_proposal.numel() > 0:
            self._is_proposal = self._is_proposal[valid_points_mask]
        if self.lr_multipliers.numel() > 0:
            self.lr_multipliers = self.lr_multipliers.view(-1, 1)[valid_points_mask]

    def prune_points(self, mask, iter, force: bool = False):
        if not force and iter <= self.args.prune_from_iter:
            return
        total_points = self._xyz.shape[0]
        if mask.numel() != total_points:
            if mask.numel() < total_points:
                pad = torch.zeros(
                    total_points - mask.numel(),
                    dtype=mask.dtype,
                    device=mask.device,
                )
                mask = torch.cat([mask, pad], dim=0)
            else:
                mask = mask[:total_points]
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.confidence = self.confidence[valid_points_mask]
        self.opacity_scale = self.opacity_scale[valid_points_mask]
        if self._is_proposal.numel() > 0:
            self._is_proposal = self._is_proposal[valid_points_mask]
        if self.lr_multipliers.numel() > 0:
            self.lr_multipliers = self.lr_multipliers.view(-1, 1)[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["bg_color", "view_transformer"]:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (
                        stored_state["exp_avg_sq"],
                        torch.zeros_like(extension_tensor),
                    ),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(
                        True
                    )
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(
                        True
                    )
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        source_indices=None,
        new_confidence=None,
        new_opacity_scale=None,
        new_is_proposal=None,
        new_lr_multipliers=None,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        device = self._xyz.device
        count = new_opacities.shape[0]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)

        if count == 0:
            return

        if source_indices is not None and source_indices.numel() > 0:
            source_indices = source_indices.long().to(device=device)
        else:
            source_indices = None

        if new_confidence is None:
            if source_indices is not None:
                new_confidence = self.confidence[source_indices]
            else:
                new_confidence = torch.ones(
                    (count, 1), dtype=torch.float32, device=device
                )
        else:
            new_confidence = new_confidence.to(device).view(-1, 1)

        if new_opacity_scale is None:
            if source_indices is not None:
                new_opacity_scale = self.opacity_scale[source_indices]
            else:
                new_opacity_scale = torch.ones(
                    (count, 1), dtype=torch.float32, device=device
                )
        else:
            new_opacity_scale = new_opacity_scale.to(device).view(-1, 1)

        if new_is_proposal is None:
            if source_indices is not None:
                new_is_proposal = self._is_proposal[source_indices]
            else:
                new_is_proposal = torch.zeros(
                    (count,), dtype=torch.bool, device=device
                )
        else:
            new_is_proposal = new_is_proposal.to(device=device).view(-1)

        if new_lr_multipliers is None:
            if source_indices is not None:
                new_lr_multipliers = self.lr_multipliers[source_indices]
            else:
                new_lr_multipliers = torch.ones(
                    (count, 1), dtype=torch.float32, device=device
                )
        else:
            new_lr_multipliers = new_lr_multipliers.to(device).view(-1, 1)

        self.confidence = torch.cat([self.confidence, new_confidence], 0)
        self.opacity_scale = torch.cat([self.opacity_scale, new_opacity_scale], 0)
        self._is_proposal = torch.cat([self._is_proposal, new_is_proposal], 0)
        self.lr_multipliers = torch.cat(
            [self.lr_multipliers.view(-1, 1), new_lr_multipliers], 0
        )

    def proximity(self, scene_extent, N=3):
        dist, nearest_indices = distCUDA2(self.get_xyz)
        selected_pts_mask = torch.logical_and(
            dist > (5.0 * scene_extent),
            torch.max(self.get_scaling, dim=1).values > (scene_extent),
        )

        if selected_pts_mask.sum() == 0:
            return

        new_indices = nearest_indices[selected_pts_mask].reshape(-1).long()
        source_xyz = self._xyz[selected_pts_mask].repeat(1, N, 1).reshape(-1, 3)
        target_xyz = self._xyz[new_indices]
        new_xyz = (source_xyz + target_xyz) / 2
        new_scaling = self._scaling[new_indices]
        new_rotation = torch.zeros_like(self._rotation[new_indices])
        new_rotation[:, 0] = 1
        new_features_dc = torch.zeros_like(self._features_dc[new_indices])
        new_features_rest = torch.zeros_like(self._features_rest[new_indices])
        new_opacity = self._opacity[new_indices]

        parent_idx = torch.nonzero(selected_pts_mask, as_tuple=False).view(-1)
        parent_idx = parent_idx.repeat_interleave(N)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            source_indices=parent_idx,
        )

    def densify_and_split(self, grads, grad_threshold, scene_extent, iter, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(
            padded_grad >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        dist, _ = distCUDA2(self.get_xyz)
        selected_pts_mask2 = torch.logical_and(
            dist > (self.args.dist_thres * scene_extent),
            torch.max(self.get_scaling, dim=1).values > (scene_extent),
        )
        selected_pts_mask = torch.logical_or(
            selected_pts_mask, selected_pts_mask2
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = (
            torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            + self.get_xyz[selected_pts_mask].repeat(N, 1)
        )
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(
            N, 1, 1
        )
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        if new_xyz.shape[0] == 0:
            return

        parent_idx = torch.nonzero(selected_pts_mask, as_tuple=False).view(-1)
        parent_idx = parent_idx.repeat_interleave(N)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            source_indices=parent_idx,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(
                    N * selected_pts_mask.sum(), device="cuda", dtype=bool
                ),
            )
        )
        self.prune_points(prune_filter, iter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        if new_xyz.shape[0] == 0:
            return

        parent_idx = torch.nonzero(selected_pts_mask, as_tuple=False).view(-1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            source_indices=parent_idx,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iter):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent, iter)
        if iter < 2000:
            self.proximity(extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = (
                self.get_scaling.max(dim=1).values > 0.1 * extent
            )
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )

        self.prune_points(prune_mask, iter)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
