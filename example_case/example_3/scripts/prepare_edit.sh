export CUDA_VISIBLE_DEVICES=0   #set GPU

python prepare_edit_tools/create_train_splits.py --base_path ./example_case/example_3/
  
python prepare_edit_tools/get_each_scale_code.py \
  --base_path example_case/example_3/ \
  --vae_path weights/infinity_vae_d32reg.pth 
