
voxel_size=0.01
echo "voxel_size=$voxel_size"

for scene_name in scene_000 scene_001 scene_002 scene_003 scene_004 scene_005 scene_006 scene_007 scene_008 scene_009 scene_010 scene_011 scene_012 scene_013 scene_014 scene_015
do
    voxel_type=INIT_FINALLY

    python -m initialization_rs_points.run_initialization \
      --workspace YOU_PATH \
      --colmap-model /DATA_PATHLevir/$scene_name/sparse/0 \
      --rgb-dir /DATA_PATHLevir/$scene_name/images \
      --depth-source depth_anything_v2 \
      --depth-checkpoint /CHECOPOINT_PATH/checkpoints/depth_anything_v2_vitb.pth \
      --visible 000.png 007.png 015.png \
      --enable-inpainting \
      --inpaint-api-key you_key \
      --inpaint-prompt "Fill the masked area seamlessly based on surrounding visible pixels. Preserve the overall visual style,
      lighting, and tone of the original aerial/satellite image. Extend nearby textures, structures, and terrain naturally into the masked
      region, ensuring a coherent continuation of context without altering unmasked areas." \
      --depth-visualize-dir INITIALIZATION/$scene_name/depth_viz \
      --dense-point-path INITIALIZATION/$scene_name/points_dense_$voxel_type.ply \
      --uncertain-point-path INITIALIZATION/$scene_name/points_proposal_$voxel_type.ply \
      --pseudo-view-dir INITIALIZATION/$scene_name/pseudo_views \
      --pseudo-mask-dir INITIALIZATION/$scene_name/pseudo_masks \
      --inpaint-dir INITIALIZATION/$scene_name/inpainted \
      --pseudo-pose-mode interpolate \
      --pseudo-count 20 \
      --pseudo-depth-from-render \
      --prepose-offset 24 \
      --voxel-size $voxel_size \
      --max-points 250000

done

