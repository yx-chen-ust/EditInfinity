export CUDA_VISIBLE_DEVICES=2   #设置GPU
export PYTHONPATH=$PYTHONPATH:/data1/chenyuxin/code/Infinity_clone/  #设置相对路径

python prepare_edit_tools/visualize_attention_map.py \
    --base_path /data1/chenyuxin/code/Infinity_clone/example_case/example_3 \
    --threshold 0.5

python prepare_edit_tools/generate_weighted_tensor.py \
    --case_path /data1/chenyuxin/code/Infinity_clone/example_case/example_3