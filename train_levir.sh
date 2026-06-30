



CARD=0
save_path="SAVE_PATH"
data_path="DATA_PATH"
few_type="test"
n_views=3
data_name="levir_shot"
colmap="real_large_voxel_points"
scenes=("scene_000" "scene_001" "scene_002" "scene_003" "scene_004" "scene_005" "scene_006" "scene_007" "scene_008" "scene_009" "scene_010" "scene_011" "scene_012" "scene_013" "scene_014" "scene_015")
for scene in ${scenes[@]}
do
  echo "This is scene: $data_path/$scene"
  CUDA_VISIBLE_DEVICES=$CARD python train_residual.py \
    --source_path "$data_path/$scene" \
    --model_path "$save_path/$few_type/$colmap/$scene" \
    --eval \
    --n_views "$n_views" \
    --colmap "$colmap" \
    --proposal_lr_multiplier 1 \
    --proposal_dc_mean_reg 3e-4 \
    --proposal_dc_var_reg 1e-5 \
    --debug_proposal_color_init \
    --use_confidence 
    CUDA_VISIBLE_DEVICES=$CARD python render_residual.py --source_path $data_path/$scene --model_path  $save_path/$few_type/$colmap/$scene --add_depth_loss False --iteration 10000  --n_views $n_views --colmap $colmap
    CUDA_VISIBLE_DEVICES=$CARD python metrics.py --source_path $data_path/$scene --model_path  $save_path/$few_type/$colmap/$scene 
done


