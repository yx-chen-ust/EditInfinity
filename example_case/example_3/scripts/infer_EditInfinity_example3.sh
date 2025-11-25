#!/bin/bash

# set arguments for inference
pn=1M
model_type=infinity_2b
use_scale_schedule_embedding=0
use_bit_label=1
checkpoint_type='torch'
infinity_model_path=weights/infinity_2b_reg.pth
vae_type=32
vae_path=weights/infinity_vae_d32reg.pth
cfg=4
tau=0.5
rope2d_normalized_by_hw=2
add_lvl_embeding_only_first_block=1
rope2d_each_sa_layer=1
text_encoder_ckpt=weights/flan-t5-xl
text_channels=2048
apply_spatial_patchify=0

#Select inference function
infer_function=2

# Whether to use concatenated language style embedding
use_concat_embedding=1
# Iteration count for the concatenated language style embedding to use
use_embedding_iter=10

# LoRA parameters
use_lora=1
lora_r=4    # LoRA rank
lora_alpha=32  # LoRA alpha
lora_dropout=0.1  # LoRA dropout rate
use_lora_iter=50

export CUDA_VISIBLE_DEVICES=0   #set GPU

#set inference data path
infer_root_dir="./example_case/"
infer_sub_dir="example_3/"

#set inference prompt
prompt_file="${infer_root_dir}${infer_sub_dir}prompt/edit_image_prompt.txt"
if [ -f "$prompt_file" ]; then
prompt="$(tr -d '\n' < "$prompt_file")"
else
echo "Prompt file not found: $prompt_file" >&2
exit 1
fi

save_file="${infer_root_dir}${infer_sub_dir}output/edit_result_example3_result.jpg"

# run inference
python3 tools/run_infinity.py \
--cfg ${cfg} \
--tau ${tau} \
--pn ${pn} \
--model_path ${infinity_model_path} \
--vae_type ${vae_type} \
--vae_path ${vae_path} \
--add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
--use_bit_label ${use_bit_label} \
--model_type ${model_type} \
--rope2d_each_sa_layer ${rope2d_each_sa_layer} \
--rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
--use_scale_schedule_embedding ${use_scale_schedule_embedding} \
--cfg ${cfg} \
--tau ${tau} \
--checkpoint_type ${checkpoint_type} \
--text_encoder_ckpt ${text_encoder_ckpt} \
--text_channels ${text_channels} \
--apply_spatial_patchify ${apply_spatial_patchify} \
--infer_function ${infer_function} \
--use_lora ${use_lora} \
--lora_r ${lora_r} \
--lora_alpha ${lora_alpha} \
--lora_dropout ${lora_dropout} \
--use_lora_iter ${use_lora_iter} \
--use_concat_embedding ${use_concat_embedding} \
--use_embedding_iter ${use_embedding_iter} \
--seed 42 \
--save_file "${save_file}" \
--infer_root_dir "${infer_root_dir}" \
--infer_sub_dir "${infer_sub_dir}" \
--prompt "${prompt}"
