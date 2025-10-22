Example3 Image Editing Pipeline User Guide

Overview
Example3 is an intelligent image editing pipeline with the following key features:  
• Automatic Background Preservation: Automatically identifies foreground and background through attention mechanisms without requiring manual mask images  
• Smart Mask Generation: Automatically generates foreground masks and background preservation weights based on attention maps  
• Two-Stage Training: First trains text embeddings, then trains the LoRA model to ensure editing quality  

Prerequisites
Before running the pipeline, ensure the following files are prepared:  
Required Files
1. Original Image: image/original_image.jpg  
2. Original Image Description: prompt/original_image_prompt.txt – describes the content of the original image  
3. Edit Prompt: prompt/edit_image_prompt.txt – describes how the image should be edited  
4. Target Keyword: prompt/target_word.txt – specifies the target word to focus on  

Environment Requirements
• Ensure all dependency environments are properly installed  
• Set up the correct Python path and CUDA environment  

Pipeline Execution Flow

Step 1: Data Preparation
Script: prepare_edit.sh  
Function:  
• Generates segmentation files (JSONL format) for the training dataset  
• Extracts multi-scale encoded features from images  
• Prepares necessary data structures for subsequent training  

Step 2: Text Embedding Training
Script: train_EditInfinity_example3.sh  
Parameter Settings:  
train_textembedding=1          # Enable text embedding training  
train_textembedding_iter=10    # Train for 10 iterations  
use_textembedding=0            # Do not use pre-trained embeddings during training  
train_lora=0                   # Do not train LoRA in this stage  
  
Step 3: LoRA Model Training
Script: train_EditInfinity_example3.sh  
Parameter Settings:  
train_textembedding=0          # Stop text embedding training  
use_textembedding=1            # Use text embeddings trained in Step 2  
use_textembedding_iter=10      # Use embedding weights from the 10th iteration  
train_lora=1                   # Enable LoRA training  
train_lora_iter=20             # Train for 20 iterations  
  
Alternative Option: If you don't want to load text embeddings, set:  
train_textembedding=0  
use_textembedding=0            # Do not load text embeddings  
train_lora=1  
train_lora_iter=20  
  
Step 4: Attention Map Generation
Script: get_targetword_attentionmap_example3.sh  
Function:  
• Uses the trained model to generate attention maps for target words  
• Attention maps identify foreground regions in the image that need editing  
• Set infer_function=1 specifically for generating attention maps  

Step 5: Mask and Weight Tensor Generation
Script: get_weighted_tensor.sh  
Function:  
• Generates foreground mask images (mask.png) based on attention maps  
• Creates weight tensors for background preservation  
• Generates visual analysis images of attention maps  

Step 6: Final Image Inference
Script: infer_EditInfinity_example3.sh  
Parameter Settings:  
infer_function=2               # Used for final image generation  
use_concat_embedding=1         # Use concatenated text embeddings  
use_embedding_iter=10          # Use embeddings from the 10th iteration  
use_lora=1                     # Use LoRA model  
use_lora_iter=20               # Use LoRA weights from the 20th iteration  
  

Parameter Explanation

Training Parameters
• train_textembedding: Whether to start text embedding training (0/1)  
• train_textembedding_iter: Number of iterations for text embedding training  
• use_textembedding: Whether to use pre-trained text embeddings (0/1)  
• use_textembedding_iter: Which iteration's text embedding weights to use  
• train_lora: Whether to start LoRA training (0/1)  
• train_lora_iter: Number of iterations for LoRA training  

Inference Parameters
• infer_function:  
  • 1: Generate attention maps  
  • 2: Generate final edited images  
• use_concat_embedding: Whether to use concatenated text embeddings (0/1)  
• use_embedding_iter: Which iteration's text embeddings to use  
• use_lora: Whether to use LoRA model (0/1)  
• use_lora_iter: Which iteration's LoRA weights to use  

Quick Start
1. Prepare required files (images and prompt files)  
2. Run edit_pipeline.sh to start automated processing  
3. Wait for the pipeline execution to complete  
4. Check the result files in the output directory  

Output Results
After pipeline execution completes, the following will be generated:  
• Trained model weight files  
• Attention maps and analysis images  
• Foreground mask images (mask.png)  
• Final edited images (with original background preserved)  

Notes
• You can use only text embeddings or only LoRA by setting the corresponding parameters to 0  
• Ensure iteration settings are consistent across steps to avoid loading non-existent weights  
• It is recommended to use the suggested parameter settings for the first run, which can be adjusted as needed later
