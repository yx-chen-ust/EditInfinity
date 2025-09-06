export CUDA_VISIBLE_DEVICES=2   #设置GPU
export PYTHONPATH=$PYTHONPATH:/data1/chenyuxin/code/Infinity_clone/  #设置相对路径

python prepare_edit_tools/create_train_splits.py --base_path /data1/chenyuxin/code/Infinity_clone/example_case/example_2/

python prepare_edit_tools/generate_weighted_tensor.py --case_path /data1/chenyuxin/code/Infinity_clone/example_case/example_2/

python prepare_edit_tools/get_each_scale_code.py \
  --base_path example_case/example_2/ \
  --vae_path weights/infinity_vae_d32_reg.pth 