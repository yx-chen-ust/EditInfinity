Example1 Image Editing Pipeline Documentation

Overview
Example1 is a basic image editing pipeline with the following core features:
• Basic editing capabilities: Focuses on fundamental image editing without complex background preservation
• Two-phase training: First trains language style embeddings, then LoRA models to ensure editing quality
• Automated processing: Complete end-to-end workflow with minimal manual intervention

Prerequisites
Before running the pipeline, ensure the following files are prepared:

Required Files
1. Original image: image/original_image.jpg
2. Edit prompt: prompt/edit_image_prompt.txt - Describes desired image modifications

Environment Requirements
• Verify all dependencies are properly installed
• Set up correct Python paths and CUDA environment

Pipeline Execution Flow

Step 1: Data Preparation
Script: prepare_edit.sh
Functionality:
• Generates segmented training dataset files (JSONL format)
• Extracts multi-scale encoded features from images
• Prepares necessary data structures for subsequent training

Step 2: Language Style Embedding Training
Script: train_EditInfinity_example1.sh
Parameter Settings:
train_textembedding=1          # Enable language style embedding training
train_textembedding_iter=10    # Train for 10 iterations
use_textembedding=0            # Don't use pretrained embeddings during training
train_lora=0                   # Disable LoRA training in this phase

Step 3: LoRA Model Training
Script: train_EditInfinity_example1.sh
Parameter Settings:
train_textembedding=0          # Disable embedding training
use_textembedding=1            # Use embeddings from Step 2
use_textembedding_iter=10      # Use weights from 10th iteration
train_lora=1                   # Enable LoRA training
train_lora_iter=50             # Train for 50 iterations

Alternative: To skip language style embeddings:
train_textembedding=0
use_textembedding=0            # Don't load language style embeddings
train_lora=1
train_lora_iter=50

Step 4: Final Image Inference
Script: infer_EditInfinity_example1.sh
Parameter Settings:
infer_function=0               # Basic image editing function
use_concat_embedding=1         # Use concatenated language style embeddings
use_embedding_iter=10          # Use weights from 10th iteration
use_lora=1                     # Use LoRA model
use_lora_iter=50               # Use weights from 50th iteration


Parameter Explanation

Training Parameters
• train_textembedding: Enable language style embedding training (0/1)
• train_textembedding_iter: Training iterations for embeddings
• use_textembedding: Use pretrained language style embeddings (0/1)
• use_textembedding_iter: Which iteration's embedding weights to use
• train_lora: Enable LoRA training (0/1)
• train_lora_iter: Training iterations for LoRA

Inference Parameters
• infer_function: Inference type (0: Basic image editing)
• use_concat_embedding: Use concatenated language style embeddings (0/1)
• use_embedding_iter: Which iteration's embeddings to use
• use_lora: Use LoRA model (0/1)
• use_lora_iter: Which iteration's LoRA weights to use

Quick Start
1. Prepare required files (image and prompt)
2. Run edit_pipeline.sh to begin automated processing
3. Wait for pipeline completion
4. Check results in output directory

Output
Upon completion, the pipeline generates:
• Trained model weights
• Final edited images

Notes
• Can use either language style embeddings or LoRA alone by setting corresponding parameters to 0
• Ensure iteration counts match across steps to avoid loading nonexistent weights
• Recommended to use suggested parameters for first run, then adjust as needed
• For background preservation features, use Example2 or Example3 instead
