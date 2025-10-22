Example2 Image Editing Pipeline Usage Guide

Functional Overview
Example2 is an intelligent image editing pipeline with the following core features:
• Background Preservation: Intelligently retains the original background by combining user-provided mask images
• Two-Stage Training: First trains text embeddings, then trains the LoRA model to ensure editing quality
• Automated Processing: Complete end-to-end processing flow minimizes manual intervention

Prerequisites
Before running the pipeline, ensure the following files are prepared:
Required Files
1. Original Image: image/original_image.jpg
2. Edit Prompt: prompt/edit_image_prompt.txt - describes how the image should be edited
3. Mask Image: A mask image is required to specify foreground and background regions

Environment Requirements
• Ensure all dependency environments are correctly installed
• Set up proper Python paths and CUDA environment

Pipeline Execution Flow

Step 1: Data Preparation
Script: prepare_edit.sh
Function:
• Generates segmentation files (JSONL format) for the training dataset
• Extracts multi-scale encoded features from images
• Prepares necessary data structures for subsequent training

Step 2: Text Embedding Training
Script: train_EditInfinity_example2.sh
Parameter Settings:
train_textembedding=1          # Enable text embedding training
train_textembedding_iter=10    # Train for 10 iterations
use_textembedding=0            # Do not use pre-trained embeddings during training
train_lora=0                   # Do not train LoRA in this stage


Step 3: LoRA Model Training
Script: train_EditInfinity_example2.sh
Parameter Settings:
train_textembedding=0          # Stop text embedding training
use_textembedding=1            # Use text embeddings trained in Step 2
use_textembedding_iter=10      # Use embedding weights from the 10th iteration
train_lora=1                   # Enable LoRA training
train_lora_iter=50             # Train for 50 iterations

Alternative Option: If you don't want to load text embeddings, set:
train_textembedding=0
use_textembedding=0            # Do not load text embeddings
train_lora=1
train_lora_iter=50


Step 4: Final Image Inference
Script: infer_EditInfinity_example2.sh
Parameter Settings:
infer_function=2               # For final image generation
use_concat_embedding=1         # Use concatenated text embeddings
use_embedding_iter=10          # Use embeddings from the 10th iteration
use_lora=1                     # Use LoRA model
use_lora_iter=50               # Use LoRA weights from the 50th iteration

Parameter Explanation

Training Parameters
• train_textembedding: Whether to start text embedding training (0/1)
• train_textembedding_iter: Number of iterations for text embedding training
• use_textembedding: Whether to use pre-trained text embeddings (0/1)
• use_textembedding_iter: Which iteration's text embedding weights to use
• train_lora: Whether to start LoRA training (0/1)
• train_lora_iter: Number of iterations for LoRA training

Inference Parameters
• infer_function: Inference function type (2: final image generation)
• use_concat_embedding: Whether to use concatenated text embeddings (0/1)
• use_embedding_iter: Which iteration's text embeddings to use
• use_lora: Whether to use LoRA model (0/1)
• use_lora_iter: Which iteration's LoRA weights to use

Quick Start
1. Prepare required files (image, prompt, and mask image)
2. Run edit_pipeline.sh to start automated processing
3. Wait for pipeline execution to complete
4. Check result files in the output directory

Output Results
After pipeline execution completes, the following will be generated:
• Trained model weight files
• Final edited images (with original background preserved)

Differences from Example3
• Example2: Requires users to provide mask images to specify foreground and background regions
• Example3: Automatically generates mask images through attention mechanism, no manual provision needed

Important Notes
• You can use only text embeddings or only LoRA by setting corresponding parameters to 0
• Ensure iteration settings are consistent across steps to avoid loading non-existent weights
• Recommended to use suggested parameter settings for initial runs, adjust as needed later
• Mask image quality directly affects background preservation results - ensure masks accurately mark foreground regions
