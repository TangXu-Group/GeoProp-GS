

CARD=0
save_path="SAVE_PATH"
data_path="DATA_PATH"
few_type="test"
n_views=7
data_name="rsscene_shot"
colmap="real_large_voxel_points"


scenes=( "City/scene0" "City/scene1" "City/scene2" "Country/scene0" "Country/scene1" "Country/scene2" "Port/scene0" "Port/scene1" "Port/scene2"  )
for scene in ${scenes[@]}
do
  echo "This is scene: $data_path/$scene"
  echo "This is save path: $save_path/$few_type/$colmap/$scene"
  CUDA_VISIBLE_DEVICES=$CARD python train_residual.py \
    --source_path "$data_path/$scene" \
    -r 4 \
    --model_path "$save_path/$few_type/$colmap/$scene" \
    --eval \
    --n_views "$n_views" \
    --colmap "$colmap" \
    --proposal_lr_multiplier 2.0 \
    --proposal_dc_mean_reg 3e-4 \
    --proposal_dc_var_reg 1e-5 \
    --debug_proposal_color_init \
    --use_confidence 
    CUDA_VISIBLE_DEVICES=$CARD python render_residual.py  --source_path $data_path/$scene --model_path  $save_path/$few_type/$colmap/$scene --add_depth_loss False --iteration 10000  --n_views $n_views --colmap $colmap
    CUDA_VISIBLE_DEVICES=$CARD python metrics.py --source_path $data_path/$scene --model_path  $save_path/$few_type/$colmap/$scene 
done


