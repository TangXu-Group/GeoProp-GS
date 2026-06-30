import torch
from torch import nn
from typing import Optional

from scene.gaussian_model import GaussianModel


class ResidualGaussianModel(GaussianModel):
    """
    Variant of GaussianModel where proposal Gaussians are parameterised as
    anchor (SfM) values plus learnable residuals. This keeps proposals tied to
    their nearest SfM point while still allowing them to adapt during training.
    """

    def __init__(self, args):
        super().__init__(args)
        self._proposal_anchor_idx = torch.empty(0, dtype=torch.long)
        self._proposal_delta_xyz: Optional[nn.Parameter] = None
        self._proposal_delta_features_dc: Optional[nn.Parameter] = None
        self._proposal_delta_features_rest: Optional[nn.Parameter] = None
        self._proposal_delta_scaling: Optional[nn.Parameter] = None
        self._proposal_delta_rotation: Optional[nn.Parameter] = None
        self._proposal_delta_opacity: Optional[nn.Parameter] = None
        self._residual_params_registered = False

    # -------------------------------------------------------------------------
    # Helper utilities
    # -------------------------------------------------------------------------
    def _device(self):
        if isinstance(self._xyz, torch.Tensor) and self._xyz.numel() > 0:
            return self._xyz.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _has_proposals(self) -> bool:
        return self._proposal_anchor_idx.numel() > 0

    def _proposal_indices(self):
        if self._is_proposal.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self._device())
        return torch.nonzero(self._is_proposal, as_tuple=True)[0]

    def _cat_parameter(self, current: Optional[nn.Parameter], addition: torch.Tensor) -> nn.Parameter:
        if current is None or current.numel() == 0:
            return nn.Parameter(addition)
        return nn.Parameter(torch.cat([current.detach(), addition], dim=0))

    def _compose_with_residual(self, base: torch.Tensor, delta: Optional[torch.Tensor]) -> torch.Tensor:
        if delta is None or delta.numel() == 0 or not self._has_proposals():
            return base.detach().clone()
        proposal_idx = self._proposal_indices()
        anchor_vals = base.detach()[self._proposal_anchor_idx]
        actual = base.detach().clone()
        actual[proposal_idx] = anchor_vals + delta.detach()
        return actual

    def _actual_parameters(self):
        xyz = self._compose_with_residual(self._xyz, self._proposal_delta_xyz)
        f_dc = self._compose_with_residual(self._features_dc, self._proposal_delta_features_dc)
        if self._features_rest.numel() > 0:
            f_rest = self._compose_with_residual(self._features_rest, self._proposal_delta_features_rest)
        else:
            f_rest = self._features_rest.detach().clone()
        scaling = self._compose_with_residual(self._scaling, self._proposal_delta_scaling)
        rotation = self._compose_with_residual(self._rotation, self._proposal_delta_rotation)
        opacity = self._compose_with_residual(self._opacity, self._proposal_delta_opacity)
        return xyz, f_dc, f_rest, scaling, rotation, opacity

    def _nearest_anchor_indices(
        self,
        anchor_xyz: torch.Tensor,
        proposal_xyz: torch.Tensor,
        prop_chunk: int = 1024,
        anchor_chunk: int = 4096,
    ) -> torch.Tensor:
        """
        Compute nearest anchor index for each proposal using chunked cdist to avoid
        materialising a huge distance matrix.
        """
        device = anchor_xyz.device
        anchor_count = anchor_xyz.shape[0]
        proposal_count = proposal_xyz.shape[0]
        result = torch.empty((proposal_count,), dtype=torch.long, device=device)

        for p_start in range(0, proposal_count, prop_chunk):
            p_end = min(p_start + prop_chunk, proposal_count)
            prop_chunk_xyz = proposal_xyz[p_start:p_end]

            chunk_min_dist = None
            chunk_min_idx = None

            for a_start in range(0, anchor_count, anchor_chunk):
                a_end = min(a_start + anchor_chunk, anchor_count)
                anchor_chunk_xyz = anchor_xyz[a_start:a_end]
                dist = torch.cdist(prop_chunk_xyz, anchor_chunk_xyz)
                local_min, local_idx = torch.min(dist, dim=1)
                local_idx = local_idx + a_start

                if chunk_min_dist is None:
                    chunk_min_dist = local_min
                    chunk_min_idx = local_idx
                else:
                    better_mask = local_min < chunk_min_dist
                    chunk_min_dist = torch.where(better_mask, local_min, chunk_min_dist)
                    chunk_min_idx = torch.where(better_mask, local_idx, chunk_min_idx)

            result[p_start:p_end] = chunk_min_idx

        return result

    def _attach_residuals_to_optimizer(self):
        if self.optimizer is None or self._residual_params_registered:
            return
        residual_groups = []
        proposal_lr = getattr(self.args, "proposal_lr_multiplier", 0.3)
        print("This is proposal_lr:{}".format(proposal_lr))
        if self._proposal_delta_xyz is not None and self._proposal_delta_xyz.numel() > 0:
            residual_groups.append({"params": [self._proposal_delta_xyz], "lr": self.optimizer.param_groups[0]["lr"] * proposal_lr, "name": "proposal_delta_xyz"})
        if self._proposal_delta_features_dc is not None and self._proposal_delta_features_dc.numel() > 0:
            # Features share feature_lr scaling
            feature_lr = next(g["lr"] for g in self.optimizer.param_groups if g["name"] == "f_dc")
            residual_groups.append({"params": [self._proposal_delta_features_dc], "lr": feature_lr * proposal_lr, "name": "proposal_delta_f_dc"})
        if self._proposal_delta_features_rest is not None and self._proposal_delta_features_rest.numel() > 0:
            feature_lr = next(g["lr"] for g in self.optimizer.param_groups if g["name"] == "f_rest")
            residual_groups.append({"params": [self._proposal_delta_features_rest], "lr": feature_lr * proposal_lr, "name": "proposal_delta_f_rest"})
        if self._proposal_delta_opacity is not None and self._proposal_delta_opacity.numel() > 0:
            opacity_lr = next(g["lr"] for g in self.optimizer.param_groups if g["name"] == "opacity")
            residual_groups.append({"params": [self._proposal_delta_opacity], "lr": opacity_lr * proposal_lr, "name": "proposal_delta_opacity"})
        if self._proposal_delta_scaling is not None and self._proposal_delta_scaling.numel() > 0:
            scaling_lr = next(g["lr"] for g in self.optimizer.param_groups if g["name"] == "scaling")
            residual_groups.append({"params": [self._proposal_delta_scaling], "lr": scaling_lr * proposal_lr, "name": "proposal_delta_scaling"})
        if self._proposal_delta_rotation is not None and self._proposal_delta_rotation.numel() > 0:
            rotation_lr = next(g["lr"] for g in self.optimizer.param_groups if g["name"] == "rotation")
            residual_groups.append({"params": [self._proposal_delta_rotation], "lr": rotation_lr * proposal_lr, "name": "proposal_delta_rotation"})

        for group in residual_groups:
            self.optimizer.add_param_group(group)
        self._residual_params_registered = True

    # -------------------------------------------------------------------------
    # Proposal handling
    # -------------------------------------------------------------------------
    def append_proposals(self, proposal):
        base_count = self._xyz.shape[0] if isinstance(self._xyz, torch.Tensor) and self._xyz.numel() > 0 else 0
        super().append_proposals(proposal)

        total_count = self._xyz.shape[0] if isinstance(self._xyz, torch.Tensor) and self._xyz.numel() > 0 else 0
        new_count = total_count - base_count
        if new_count <= 0:
            return

        device = self._device()
        proposal_indices = torch.arange(total_count - new_count, total_count, device=device, dtype=torch.long)

        anchor_candidates = torch.nonzero(~self._is_proposal, as_tuple=True)[0]
        if anchor_candidates.numel() == 0:
            raise RuntimeError("No SfM anchors available to condition proposals.")

        anchor_xyz = self._xyz[anchor_candidates]
        proposal_xyz = self._xyz[proposal_indices]

        with torch.no_grad():
            closest = self._nearest_anchor_indices(anchor_xyz, proposal_xyz)
            anchors = anchor_candidates[closest]

        # Compute residuals before overwriting base tensors
        delta_xyz_new = (self._xyz[proposal_indices].detach() - self._xyz[anchors].detach())
        delta_f_dc_new = (self._features_dc[proposal_indices].detach() - self._features_dc[anchors].detach())
        if self._features_rest.numel() > 0:
            delta_f_rest_new = (self._features_rest[proposal_indices].detach() - self._features_rest[anchors].detach())
        else:
            delta_f_rest_new = torch.zeros((new_count, 0, 3), device=device, dtype=self._features_dc.dtype)
        delta_scaling_new = (self._scaling[proposal_indices].detach() - self._scaling[anchors].detach())
        delta_rotation_new = (self._rotation[proposal_indices].detach() - self._rotation[anchors].detach())
        delta_opacity_new = (self._opacity[proposal_indices].detach() - self._opacity[anchors].detach())

        # Overwrite proposal slots with anchor values so base tensors represent anchors.
        with torch.no_grad():
            self._xyz[proposal_indices] = self._xyz[anchors]
            self._features_dc[proposal_indices] = self._features_dc[anchors]
            if self._features_rest.numel() > 0:
                self._features_rest[proposal_indices] = self._features_rest[anchors]
            self._scaling[proposal_indices] = self._scaling[anchors]
            self._rotation[proposal_indices] = self._rotation[anchors]
            self._opacity[proposal_indices] = self._opacity[anchors]

        # Persist anchors and residual parameters
        anchors = anchors.to(device=device, dtype=torch.long)
        if self._proposal_anchor_idx.numel() == 0:
            self._proposal_anchor_idx = anchors
        else:
            self._proposal_anchor_idx = torch.cat([self._proposal_anchor_idx.to(device=device), anchors], dim=0)

        self._proposal_delta_xyz = self._cat_parameter(self._proposal_delta_xyz, delta_xyz_new.to(device))
        self._proposal_delta_features_dc = self._cat_parameter(self._proposal_delta_features_dc, delta_f_dc_new.to(device))
        self._proposal_delta_features_rest = self._cat_parameter(self._proposal_delta_features_rest, delta_f_rest_new.to(device))
        self._proposal_delta_scaling = self._cat_parameter(self._proposal_delta_scaling, delta_scaling_new.to(device))
        self._proposal_delta_rotation = self._cat_parameter(self._proposal_delta_rotation, delta_rotation_new.to(device))
        self._proposal_delta_opacity = self._cat_parameter(self._proposal_delta_opacity, delta_opacity_new.to(device))

        self._residual_params_registered = False  # Will be attached during training_setup

    # -------------------------------------------------------------------------
    # Properties overriding base behaviour
    # -------------------------------------------------------------------------
    @property
    def get_xyz(self):
        base_xyz = super().get_xyz
        if not self._has_proposals():
            return base_xyz
        combined = base_xyz.clone()
        proposal_idx = self._proposal_indices()
        anchor_xyz = base_xyz[self._proposal_anchor_idx]
        combined[proposal_idx] = anchor_xyz + self._proposal_delta_xyz
        return combined

    @property
    def get_features(self):
        base_dc = self._features_dc
        base_rest = self._features_rest
        if not self._has_proposals():
            return torch.cat((base_dc, base_rest), dim=1)
        dc = base_dc.clone()
        rest = base_rest.clone()
        proposal_idx = self._proposal_indices()
        anchor_dc = base_dc[self._proposal_anchor_idx]
        dc[proposal_idx] = anchor_dc + self._proposal_delta_features_dc
        if rest.numel() > 0:
            anchor_rest = rest[self._proposal_anchor_idx]
            rest[proposal_idx] = anchor_rest + self._proposal_delta_features_rest
        return torch.cat((dc, rest), dim=1)

    @property
    def get_scaling(self):
        base_scaling = self._scaling
        if not self._has_proposals():
            return self.scaling_activation(base_scaling)
        scaling_log = base_scaling.clone()
        proposal_idx = self._proposal_indices()
        anchor_log = base_scaling[self._proposal_anchor_idx]
        scaling_log[proposal_idx] = anchor_log + self._proposal_delta_scaling
        return self.scaling_activation(scaling_log)

    @property
    def get_rotation(self):
        base_rotation = self._rotation
        if not self._has_proposals():
            return self.rotation_activation(base_rotation)
        rotation = base_rotation.clone()
        proposal_idx = self._proposal_indices()
        anchor_rot = base_rotation[self._proposal_anchor_idx]
        rotation[proposal_idx] = anchor_rot + self._proposal_delta_rotation
        return self.rotation_activation(rotation)

    @property
    def get_opacity(self):
        base_opacity = self._opacity
        if not self._has_proposals():
            return self.opacity_activation(base_opacity)
        logits = base_opacity.clone()
        proposal_idx = self._proposal_indices()
        anchor_logits = base_opacity[self._proposal_anchor_idx]
        logits[proposal_idx] = anchor_logits + self._proposal_delta_opacity
        return self.opacity_activation(logits)

    # Opacity scale uses base implementation but make sure tensor sizes agree.
    @property
    def get_opacity_scale(self):
        scale = super().get_opacity_scale
        return scale

    # -------------------------------------------------------------------------
    # Optimisation helper overrides
    # -------------------------------------------------------------------------
    def training_setup(self, training_args):
        super().training_setup(training_args)
        self._attach_residuals_to_optimizer()

    def apply_proposal_gradient_scale(self):
        super().apply_proposal_gradient_scale()
        if not self._has_proposals():
            return
        lr = self.lr_multipliers.view(-1, 1)
        proposal_lr = lr[self._proposal_indices()]
        if self._proposal_delta_xyz is not None and self._proposal_delta_xyz.grad is not None:
            self._proposal_delta_xyz.grad.mul_(proposal_lr)
        if self._proposal_delta_features_dc is not None and self._proposal_delta_features_dc.grad is not None:
            self._proposal_delta_features_dc.grad.mul_(proposal_lr.view(-1, 1, 1))
        if self._proposal_delta_features_rest is not None and self._proposal_delta_features_rest.grad is not None and self._proposal_delta_features_rest.numel() > 0:
            self._proposal_delta_features_rest.grad.mul_(proposal_lr.view(-1, 1, 1))
        if self._proposal_delta_scaling is not None and self._proposal_delta_scaling.grad is not None:
            self._proposal_delta_scaling.grad.mul_(proposal_lr)
        if self._proposal_delta_rotation is not None and self._proposal_delta_rotation.grad is not None:
            self._proposal_delta_rotation.grad.mul_(proposal_lr)
        if self._proposal_delta_opacity is not None and self._proposal_delta_opacity.grad is not None:
            self._proposal_delta_opacity.grad.mul_(proposal_lr)

    # -------------------------------------------------------------------------
    # Checkpoint helpers
    # -------------------------------------------------------------------------
    def capture(self):
        base_state = super().capture()
        extras = (
            self._proposal_anchor_idx,
            getattr(self._proposal_delta_xyz, "detach", lambda: torch.empty(0, device=self._device()))(),
            getattr(self._proposal_delta_features_dc, "detach", lambda: torch.empty(0, device=self._device()))(),
            getattr(self._proposal_delta_features_rest, "detach", lambda: torch.empty(0, device=self._device()))(),
            getattr(self._proposal_delta_scaling, "detach", lambda: torch.empty(0, device=self._device()))(),
            getattr(self._proposal_delta_rotation, "detach", lambda: torch.empty(0, device=self._device()))(),
            getattr(self._proposal_delta_opacity, "detach", lambda: torch.empty(0, device=self._device()))(),
        )
        return base_state + extras

    def restore(self, model_args, training_args):
        super().restore(model_args, training_args)
        extras = model_args[len(super().capture()):]
        if len(extras) >= 7:
            (
                anchor_idx,
                delta_xyz,
                delta_f_dc,
                delta_f_rest,
                delta_scaling,
                delta_rotation,
                delta_opacity,
            ) = extras[:7]
            if anchor_idx.numel() > 0:
                device = self._device()
                self._proposal_anchor_idx = anchor_idx.to(device=device, dtype=torch.long)
                self._proposal_delta_xyz = nn.Parameter(delta_xyz.to(device=device))
                self._proposal_delta_features_dc = nn.Parameter(delta_f_dc.to(device=device))
                self._proposal_delta_features_rest = nn.Parameter(delta_f_rest.to(device=device))
                self._proposal_delta_scaling = nn.Parameter(delta_scaling.to(device=device))
                self._proposal_delta_rotation = nn.Parameter(delta_rotation.to(device=device))
                self._proposal_delta_opacity = nn.Parameter(delta_opacity.to(device=device))
        self._residual_params_registered = False

    def save_ply(self, path):
        xyz, f_dc, f_rest, scaling, rotation, opacity = self._actual_parameters()

        # Temporarily stash originals to reuse base saver without mutating state
        orig = {
            "xyz": self._xyz,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "opacity": self._opacity,
        }
        try:
            self._xyz = nn.Parameter(xyz.to(self._device()))
            self._features_dc = nn.Parameter(f_dc.to(self._device()))
            self._features_rest = nn.Parameter(f_rest.to(self._device()))
            self._scaling = nn.Parameter(scaling.to(self._device()))
            self._rotation = nn.Parameter(rotation.to(self._device()))
            self._opacity = nn.Parameter(opacity.to(self._device()))
            super().save_ply(path)
        finally:
            self._xyz = orig["xyz"]
            self._features_dc = orig["f_dc"]
            self._features_rest = orig["f_rest"]
            self._scaling = orig["scaling"]
            self._rotation = orig["rotation"]
            self._opacity = orig["opacity"]
