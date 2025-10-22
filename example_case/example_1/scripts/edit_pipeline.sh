#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:""  #Set its path ./EditInfinity as a relative path

set -euo pipefail
set -x

# This pipeline automates the sequence described in pipeline.txt:
# 1) Prepare dataset splits and required tensors
# 2) Train text embedding (only)
# 3) Train LoRA (load the trained text embedding)
# 4) Run inference (load both text embedding and LoRA)

# Paths (adjust only if your tree layout differs)
EX1_ROOT="EditInfinity/example_case/example_1"
PREPARE_SH="${EX1_ROOT}/scripts/prepare_edit.sh"
TRAIN_SH="${EX1_ROOT}/scripts/train_EditInfinity_example1.sh"
INFER_SH="${EX1_ROOT}/scripts/infer_EditInfinity_example1.sh"

# Derive infer_root_dir and infer_sub_dir from EX1_ROOT
BASE_DIR="$(dirname "$EX1_ROOT")/"
SUB_DIR="$(basename "$EX1_ROOT")/"

# Helper: in-place update of a shell var assignment in a file, e.g. train_lora=1
update_var() {
  local file="$1"; shift
  local var="$1"; shift
  local val="$1"; shift
  # Replace lines that start with var=... (no leading spaces in our scripts)
  sed -E -i "s|^(${var}=).*|\1${val}|" "$file"
}

# Helper: in-place update of command line arguments in a file, e.g. --base_path /path/to/dir
update_arg() {
  local file="$1"; shift
  local arg="$1"; shift
  local val="$1"; shift
  local occurrence="${1:-1}"  # Which occurrence to replace (default: 1st)
  # Replace the argument value (handles both --arg value and --arg=value formats)
  sed -E -i "${occurrence}s|(${arg}[[:space:]]*=?[[:space:]]*)[^[:space:]]*|\1${val}|" "$file"
}

echo "[Step 1] Prepare dataset (splits, weights, etc.)"
# Update paths in prepare_edit.sh to use EX1_ROOT
update_arg "$PREPARE_SH" "--base_path" "$EX1_ROOT/"
# Also update the relative path for get_each_scale_code.py
RELATIVE_PATH="$(basename "$EX1_ROOT")/"
update_arg "$PREPARE_SH" "--base_path" "$RELATIVE_PATH" 2
bash "$PREPARE_SH"

echo "[Step 2] Train text embedding only"
update_var "$TRAIN_SH" train_textembedding 1
update_var "$TRAIN_SH" train_textembedding_iter 10
update_var "$TRAIN_SH" use_textembedding 0
update_var "$TRAIN_SH" use_textembedding_iter 0
update_var "$TRAIN_SH" train_lora 0
update_var "$TRAIN_SH" train_lora_iter 0
# Set train roots for scripts as required (derived from EX1_ROOT)
update_var "$TRAIN_SH" train_root_dir "\"$BASE_DIR\""
update_var "$TRAIN_SH" train_sub_dir "\"$SUB_DIR\""
bash "$TRAIN_SH"

echo "[Step 3] Train LoRA (load the text embedding)"
update_var "$TRAIN_SH" train_textembedding 0
update_var "$TRAIN_SH" train_textembedding_iter 0
update_var "$TRAIN_SH" use_textembedding 1
update_var "$TRAIN_SH" use_textembedding_iter 10
update_var "$TRAIN_SH" train_lora 1
update_var "$TRAIN_SH" train_lora_iter 50
# Ensure train roots are set consistently (derived again to be safe)
update_var "$TRAIN_SH" train_root_dir "\"$BASE_DIR\""
update_var "$TRAIN_SH" train_sub_dir "\"$SUB_DIR\""
bash "$TRAIN_SH"

echo "[Step 4] Inference (load text embedding and LoRA)"
# Set inference parameters for final image generation
update_var "$INFER_SH" infer_root_dir "\"$BASE_DIR\""
update_var "$INFER_SH" infer_sub_dir "\"$SUB_DIR\""
update_var "$INFER_SH" use_concat_embedding 1
update_var "$INFER_SH" use_embedding_iter 10
update_var "$INFER_SH" use_lora 1
update_var "$INFER_SH" use_lora_iter 50
bash "$INFER_SH"

echo "Pipeline finished successfully."
