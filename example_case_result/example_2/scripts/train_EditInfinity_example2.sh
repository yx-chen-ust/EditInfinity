#!/usr/bin/env bash

set -x

# set dist args
# SINGLE=1
nproc_per_node=${ARNOLD_WORKER_GPU}

if [ ! -z "$SINGLE" ] && [ "$SINGLE" != "0" ]; then
  echo "[single node alone] SINGLE=$SINGLE"
  nnodes=1
  node_rank=0
  nproc_per_node=1
  master_addr=127.0.0.1
  master_port=12345
else
  MASTER_NODE_ID=0
  nnodes=${ARNOLD_WORKER_NUM}
  node_rank=${ARNOLD_ID}
  master_addr="METIS_WORKER_${MASTER_NODE_ID}_HOST"
  master_addr=${!master_addr}
  master_port="METIS_WORKER_${MASTER_NODE_ID}_PORT"
  master_port=${!master_port}
  ports=(`echo $master_port | tr ',' ' '`)
  master_port=${ports[0]}
fi

echo "[nproc_per_node: ${nproc_per_node}]"
echo "[nnodes: ${nnodes}]"
echo "[node_rank: ${node_rank}]"
echo "[master_addr: ${master_addr}]"
echo "[master_port: ${master_port}]"

# set up envs
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eno1
export CUDA_VISIBLE_DEVICES=3,4

BED=checkpoints
LOCAL_OUT=local_output
mkdir -p $BED
mkdir -p $LOCAL_OUT


export COMPILE_GAN=0
export USE_TIMELINE_SDK=1
export CUDA_TIMER_STREAM_KAFKA_CLUSTER=bmq_data_va
export CUDA_TIMER_STREAM_KAFKA_TOPIC=megatron_cuda_timer_tracing_original_v2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

train_root_dir="/data1/chenyuxin/code/Infinity_clone/example_case/"
train_sub_dir="example_2/"


#是否启动语言风格embedding训练
train_textembedding=0
#需要训练的语言风格embedding iter数
train_textembedding_iter=0

# 使用的训练语言风格偏移量embedding的iter数
use_textembedding=1
use_textembedding_iter=10
#是否启动Lora训练
train_lora=1
#需要训练的lora iter数
train_lora_iter=50

# 断言：train_textembedding 和 train_lora 至多只能有一个为 1
if [ "${train_textembedding}" -eq 1 ] && [ "${train_lora}" -eq 1 ]; then
echo "Error: train_textembedding and train_lora cannot both be 1" >&2
exit 1
fi

#wandb offline
exp_name=debug_example1
bed_path=checkpoints/${exp_name}/
data_path="${train_root_dir}${train_sub_dir}splits"   # path to the data for only finetune one image
video_data_path=''
local_out_path=$LOCAL_OUT/${exp_name}


torchrun \
--nproc_per_node=2 \
--nnodes=1 \
--node_rank=0 \
--master_addr="10.249.189.212" \
--master_port=2988 \
train.py \
--ep=2 \
--opt=adamw \
--cum=3 \
--sche=lin0 \
--fp16=2 \
--ada=0.9_0.97 \
--tini=-1 \
--tclip=5 \
--flash=0 \
--alng=5e-06 \
--saln=1 \
--cos=1 \
--enable_checkpointing=full-block \
--local_out_path ${local_out_path} \
--task_type='t2i' \
--bed=${bed_path} \
--data_path=${data_path} \
--video_data_path=${video_data_path} \
--exp_name=${exp_name} \
--tblr=6e-3 \
--pn 1M \
--model=2bc8 \
--lbs=1 \
--workers=8 \
--short_cap_prob 0.5 \
--online_t5=1 \
--use_streaming_dataset 1 \
--iterable_data_buffersize 30000 \
--Ct5=2048 \
--t5_path=weights/flan-t5-xl \
--vae_type 32 \
--vae_ckpt=weights/infinity_vae_d32_reg.pth \
--rush_resume=weights/infinity_2b_reg.pth \
--wp 0.00000001 \
--wpe=1 \
--dynamic_resolution_across_gpus 1 \
--enable_dynamic_length_prompt 1 \
--reweight_loss_by_scale 1 \
--add_lvl_embeding_only_first_block 1 \
--rope2d_each_sa_layer 1 \
--rope2d_normalized_by_hw 2 \
--use_fsdp_model_ema 0 \
--always_training_scales 100 \
--use_bit_label 1 \
--zero=2 \
--save_model_iters_freq 10 \
--log_freq=10 \
--checkpoint_type='torch' \
--prefetch_factor=16 \
--noise_apply_strength 0.3 \
--noise_apply_layers 13 \
--apply_spatial_patchify 0 \
--use_flex_attn=True \
--pad=128 \
--train_textembedding=${train_textembedding} \
--train_textembedding_iter=${train_textembedding_iter} \
--train_lora=${train_lora} \
--train_lora_iter=${train_lora_iter} \
--lora_r=4 \
--lora_alpha=32 \
--lora_dropout=0.1 \
--use_kl_loss=1 \
--kl_weight=0.0000002 \
--train_root_dir=${train_root_dir} \
--train_sub_dir=${train_sub_dir} \
--use_textembedding=${use_textembedding} \
--use_textembedding_iter=${use_textembedding_iter} \

#lbs is batchsize
#--pn 1M means 1024*1024 pixels