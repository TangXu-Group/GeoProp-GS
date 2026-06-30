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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.preview_initial_ply = False
        self.preview_ply_limit = 5
        self.eval = False
        self.n_views = 0
        self.colmap = "colmap"
        self.disable_proposals = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.use_confidence = False
        self.use_color = True
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 10_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 10_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.view_transformer_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.prune_from_iter = 500
        self.densify_until_iter = 10_000
        self.densify_grad_threshold = 0.0005
        self.prune_threshold = 0.005
        self.start_sample_pseudo = 2000
        self.end_sample_pseudo = 9500
        self.sample_pseudo_interval = 10
        self.dist_thres = 10.
        self.depth_weight = 0.05
        self.depth_pseudo_weight = 0.5
        self.proposal_render_factor = 1.0
        self.proposal_eval_interval = 200
        self.proposal_promote_alpha = 0.5
        self.proposal_prune_alpha = 0.01
        self.proposal_min_steps = 400
        self.proposal_max_age = 2000
        self.proposal_scale_threshold = 0.3
        self.proposal_grad_threshold = 5e-4
        self.enable_view_sem = False
        self.view_sem_lam = 0.05
        self.view_sem_samples = 1024
        self.semantic_model_type = 'clip_vit'
        self.semantic_model_layers = -1
        self.semantic_size = 224
        self.semantic_cache_root = ''
        self.proposal_semantic_threshold = 0.3
        self.proposal_semantic_momentum = 0.5
        self.proposal_prune_fraction = 0.02
        self.proposal_prune_max = 0
        self.proposal_score_sem = 1.0
        self.proposal_score_opacity = 1.0
        self.proposal_score_scale = 0.5
        self.proposal_sigma_ratio = 0.05
        self.proposal_densify_interval = 0
        self.proposal_densify_grad_threshold = 0.0
        self.proposal_split_factor = 2
        self.enable_proto_link = False
        self.proposal_lr_multiplier = 2.0
        self.proposal_dc_mean_reg = 0.0
        self.proposal_dc_var_reg = 0.0
        self.proto_link_lam = 0.05
        self.proto_num_prototypes = 16
        self.proto_temperature = 0.07
        self.proto_momentum = 0.1
        self.proto_min_points = 64
        self.proto_model_type = 'dinov2_vits14'
        self.proto_model_layers = -1
        self.proto_cache_root = ''
        self.proto_color_lam = 0.02
        self.proto_size = 224
        self.pseudo_buffer_size = 12
        self.pseudo_min_score = 0.4
        self.pseudo_batch_size = 2
        self.pseudo_generate_interval = 100
        self.pseudo_train_interval = 50
        self.pseudo_score_sem_weight = 0.6
        self.pseudo_warmup_iter = 500
        self.pseudo_penalty_beta = 2.0
        self.pseudo_inpaint_conf_fill = 0.2
        self.pseudo_frame_score_scale = 1.0
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
