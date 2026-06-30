import argparse
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from initialization_rs_points.config import PipelineConfig
    from initialization_rs_points import SequentialInitializationPipeline
else:
    from .config import PipelineConfig
    from . import SequentialInitializationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sequential initialization pipeline for RS Depth 3åçDGS.")
    parser.add_argument("--workspace", type=Path, required=True, help="Project root for resolving relative outputs.")
    parser.add_argument("--colmap-model", type=Path, required=True, help="Path to COLMAP sparse model directory.")
    parser.add_argument("--rgb-dir", type=Path, required=True, help="Directory with RGB images (matching COLMAP names).")
    parser.add_argument("--depth-source", choices=["depth_anything_v2"], default="depth_anything_v2", help="Depth source.")
    parser.add_argument("--visible", nargs="+", required=True, help="Visible views used for dense point cloud.")
    parser.add_argument(
        "--voxel-size",
        "--voxel_size",
        dest="voxel_size",
        type=float,
        default=0.03,
        help="Voxel size for point downsampling.",
    )
    parser.add_argument("--max-points", type=int, default=250000, help="Maximum points to keep in dense cloud after downsampling.")
    parser.add_argument("--max-uncertain", type=int, default=1000000, help="Maximum uncertain points sampled per pipeline.")
    parser.add_argument("--depth-scale", type=float, default=1.0, help="Scale factor applied to depth values.")
    parser.add_argument("--min-depth", type=float, default=0.0, help="Minimum valid depth (after scaling).")
    parser.add_argument("--max-depth", type=float, default=200.0, help="Maximum valid depth (after scaling).")
    parser.add_argument("--coverage-threshold", type=float, default=0.05, help="Minimum render coverage before warning.")
    parser.add_argument("--inpaint-prompt", default="", help="Prompt passed to Flux Fill inpainting.")
    parser.add_argument("--inpaint-api-key", default=None, help="BFL API key; omit to skip remote inpainting.")
    parser.add_argument("--inpaint-steps", type=int, default=30, help="Inpainting inference steps.")
    parser.add_argument("--inpaint-guidance", type=float, default=20.0, help="Inpainting guidance scale.")
    parser.add_argument("--inpaint-format", default="png", help="Inpainting output format (png/jpeg).")
    parser.add_argument("--inpaint-safety", type=int, default=2, help="Safety tolerance for Flux Fill.")
    parser.add_argument("--inpaint-timeout", type=int, default=300, help="Inpainting timeout in seconds.")
    parser.add_argument("--enable-inpainting", action="store_true", help="Enable remote inpainting; requires API key.")
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument("--dense-point-path", type=Path, default=Path("output/points_dense.ply"), help="Dense point cloud output path.")
    parser.add_argument("--uncertain-point-path", type=Path, default=Path("output/points_proposal.ply"), help="Proposal point cloud output path.")
    parser.add_argument("--pseudo-view-dir", type=Path, default=Path("output/pseudo_views"), help="Directory to save pseudo rendered images.")
    parser.add_argument("--pseudo-mask-dir", type=Path, default=Path("output/pseudo_masks"), help="Directory to save pseudo view masks.")
    parser.add_argument("--inpaint-dir", type=Path, default=Path("output/inpainted"), help="Directory to save inpainted images.")
    parser.add_argument(
        "--pseudo-pose-mode",
        choices=["interpolate"],
        default="interpolate",
        help="Compatibility option; pseudo poses are always interpolated from visible poses.",
    )
    parser.add_argument(
        "--pseudo-count",
        type=int,
        default=0,
        help="Number of interpolated pseudo views to generate from visible poses.",
    )
    parser.add_argument(
        "--prepose-count",
        type=int,
        default=0,
        help="Number of shifted training poses to process before pseudo views (0 uses all visible views).",
    )
    parser.add_argument(
        "--prepose-offset",
        type=float,
        default=1.5,
        help="Distance (in scene units) to shift visible poses backward along their viewing direction during the prepose pass.",
    )
    parser.add_argument(
        "--rgb-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to COLMAP intrinsics to match downsampled RGB resolutions.",
    )
    parser.add_argument(
        "--clean-inpaint-mask",
        action="store_true",
        help="Enable morphological cleanup on inpaint masks before saving/rasterizing.",
    )
    parser.add_argument(
        "--clean-mask-min-area",
        type=int,
        default=100,
        help="Minimum connected-component area preserved during mask cleanup.",
    )
    parser.add_argument(
        "--pseudo-depth-from-render",
        action="store_true",
        help="Predict depth for inpainted pseudo views directly from the rendered/inpainted RGB instead of loading saved depth maps.",
    )
    parser.add_argument("--depth-encoder", default="vitb", help="DepthAnything encoder size (vits/vitb/vitl/vitg).")
    parser.add_argument("--depth-checkpoint", type=Path, help="Path to DepthAnything checkpoint file or directory.")
    parser.add_argument("--depth-input-size", type=int, default=518, help="Input resolution passed to DepthAnything.")
    parser.add_argument("--depth-device", default="auto", help="Inference device id (cuda, cuda:0, cpu, mps, auto).")
    parser.add_argument("--depth-valid-threshold", type=float, default=1.0, help="Minimum valid depth value before fallback handling.")
    parser.add_argument("--depth-invert", action="store_true", default=True, help="Invert depth to match disparity-style output.")
    parser.add_argument("--no-depth-invert", dest="depth_invert", action="store_false", help="Disable depth inversion.")
    parser.add_argument("--depth-visualize-dir", type=Path, help="Optional directory to save depth visualizations.")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = PipelineConfig(
        workspace_root=args.workspace.resolve(),
        colmap_model_dir=args.colmap_model,
        rgb_dir=args.rgb_dir,
        visible_views=args.visible,
        voxel_size=args.voxel_size,
        max_points=args.max_points,
        max_uncertain_points=args.max_uncertain,
        depth_scale=args.depth_scale,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        coverage_threshold=args.coverage_threshold,
        inpaint_prompt=args.inpaint_prompt,
        inpaint_api_key=args.inpaint_api_key,
        inpaint_steps=args.inpaint_steps,
        inpaint_guidance=args.inpaint_guidance,
        inpaint_output_format=args.inpaint_format,
        inpaint_safety=args.inpaint_safety,
        inpaint_timeout=args.inpaint_timeout,
        random_seed=args.random_seed,
        enable_inpainting=args.enable_inpainting,
        dense_point_path=args.dense_point_path,
        uncertain_point_path=args.uncertain_point_path,
        pseudo_view_dir=args.pseudo_view_dir,
        pseudo_mask_dir=args.pseudo_mask_dir,
        inpaint_dir=args.inpaint_dir,
        pseudo_pose_count=args.pseudo_count,
        rgb_scale=args.rgb_scale,
        depth_source=args.depth_source,
        depth_encoder=args.depth_encoder,
        depth_checkpoint_path=args.depth_checkpoint,
        depth_input_size=args.depth_input_size,
        depth_device=args.depth_device,
        depth_valid_threshold=args.depth_valid_threshold,
        depth_invert=args.depth_invert,
        depth_visualize_dir=args.depth_visualize_dir,
    )
    cfg.prepose_count = args.prepose_count
    cfg.prepose_offset = args.prepose_offset
    cfg.clean_inpaint_mask = args.clean_inpaint_mask
    cfg.clean_mask_min_area = args.clean_mask_min_area
    cfg.pseudo_depth_from_render = args.pseudo_depth_from_render

    pipeline = SequentialInitializationPipeline(cfg)
    outputs = pipeline.run()
    print("\n[SeqInit] Outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
