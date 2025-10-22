export CUDA_VISIBLE_DEVICES=0   #set GPU
export PYTHONPATH=$PYTHONPATH:""  #Set its path ./EditInfinity as a relative path

python prepare_edit_tools/create_train_splits.py --base_path ./EditInfinity/example_case/example_2/

python prepare_edit_tools/generate_weighted_tensor.py --case_path ./EditInfinity/example_case/example_2/

python prepare_edit_tools/get_each_scale_code.py \
  --base_path example_case/example_2/ \
  --vae_path weights/infinity_vae_d32_reg.pth 
