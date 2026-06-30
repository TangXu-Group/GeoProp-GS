for scene_name in City/scene0 City/scene1  City/scene2 Country/scene0  Country/scene1  Country/scene2 Port/scene0  Port/scene1  Port/scene2
do

    voxel_type=INIT_FINALLY
    python -m initialization_rs_points.run_initialization \
      --workspace YOU_PATH \
      --colmap-model /DATA_PATHrsscene/$scene_name/sparse/0 \
      --rgb-dir /DATA_PATHrsscene/$scene_name/images \
      --depth-source depth_anything_v2 \
      --depth-checkpoint /CHECOPOINT_PATH/checkpoints/depth_anything_v2_vitb.pth \
      --visible frame_001.png frame_011.png frame_023.png frame_034.png frame_046.png frame_058.png frame_069.png \
      --enable-inpainting \
      --inpaint-api-key you_key \
      --inpaint-prompt "Fill the masked area seamlessly based on surrounding visible pixels. Preserve the overall visual style, lighting, and tone of the original aerial/satellite image. Extend nearby textures, structures, and terrain naturally into the masked region, ensuring a coherent continuation of context without altering unmasked areas." \
      --depth-visualize-dir INITIALIZATION/$scene_name/depth_viz \
      --dense-point-path INITIALIZATION/$scene_name/points_dense_$voxel_type.ply \
      --uncertain-point-path INITIALIZATION/$scene_name/points_proposal_$voxel_type.ply \
      --pseudo-view-dir INITIALIZATION/$scene_name/pseudo_views \
      --pseudo-mask-dir INITIALIZATION/$scene_name/pseudo_masks \
      --inpaint-dir INITIALIZATION/$scene_name/inpainted \
      --pseudo-pose-mode interpolate \
      --pseudo-depth-from-render \
      --pseudo-count 30 \
      --rgb-scale 1 \
      --voxel_size 0.01 \
      --prepose-offset 1.5
done
