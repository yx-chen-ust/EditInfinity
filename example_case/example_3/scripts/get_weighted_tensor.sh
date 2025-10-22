export CUDA_VISIBLE_DEVICES=0   #set GPU
export PYTHONPATH=$PYTHONPATH:""  #Set its path ./EditInfinity as a relative path

python prepare_edit_tools/visualize_attention_map.py \
    --base_path ./EditInfinity/example_case/example_3 \
    --threshold 0.5

python prepare_edit_tools/generate_weighted_tensor.py \
    --case_path ./EditInfinity/example_case/example_3
