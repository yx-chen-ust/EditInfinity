import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
from typing import List
import argparse

import numpy as np
import torch
torch._dynamo.config.cache_size_limit=64
import torch.nn.functional as F

from PIL import Image as PImage
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


def load_visual_tokenizer(vae_type, apply_spatial_patchify, vae_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load vae
    from infinity.models.bsq_vae.vae import vae_model
    schedule_mode = "dynamic"
    codebook_dim = vae_type
    codebook_size = 2**codebook_dim
    if apply_spatial_patchify:
        patch_size = 8
        encoder_ch_mult=[1, 2, 4, 4]
        decoder_ch_mult=[1, 2, 4, 4]
    else:
        patch_size = 16
        encoder_ch_mult=[1, 2, 4, 4, 4]
        decoder_ch_mult=[1, 2, 4, 4, 4]
    vae = vae_model(vae_path, schedule_mode, codebook_dim, codebook_size, patch_size=patch_size, 
                    encoder_ch_mult=encoder_ch_mult, decoder_ch_mult=decoder_ch_mult, test_mode=True).to(device)
    return vae


def process_image_to_code(image_path, vae, save_dir):
    """Process a single image to code and save the results"""
    transform = Compose([
        Resize((1024, 1024)),  
        ToTensor(),            
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
    ])
    
    with PImage.open(image_path) as img:
        img_tensor = transform(img)  # 1024x1024, [-1, 1] [C,H,W]
    
    img_tensor = img_tensor.unsqueeze(0)  # B, C, H, W
    img_tensor = img_tensor.to('cuda')  
    
    x_recon, vq_output = vae(img_tensor)
    
    # Create save directory if not exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save each scale code
    for i, all_bit_indices in enumerate(vq_output["encodings"]):
        codes = vae.quantizer.lfq.indices_to_codes(all_bit_indices, label_type='bit_label')
        each_scale_code = F.interpolate(codes, size=(1, 64, 64), mode=vae.quantizer.z_interplote_up)
        save_path = os.path.join(save_dir, f"each_scale_code_{i}.pt")
        torch.save(each_scale_code, save_path)
        print(f"Saved Layer {i} to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and save each-scale codes for a single image using the Infinity VAE.")
    parser.add_argument("--base_path", type=str, required=True,
                        help="Base directory of a sample, e.g., /path/to/example_X/. The image is expected at base_path/image/original_image.jpg, and outputs will be saved to base_path/each_scale_code/original_image.")
    parser.add_argument("--vae_path", type=str, default="/data1/chenyuxin/code/Infinity_clone/weights/infinity_vae_d32_reg.pth", help="Path to the Infinity VAE weights.")
    parser.add_argument("--vae_type", type=int, default=32, help="VAE codebook dimension (e.g., 32).")
    parser.add_argument("--apply_spatial_patchify", type=int, default=0, help="Whether to use spatial patchify (0/1).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Initialize VAE
    vae = load_visual_tokenizer(args.vae_type, args.apply_spatial_patchify, args.vae_path)

    # Derive image path and save directory from base_path
    base_path = args.base_path
    image_path = os.path.join(base_path, 'image', 'original_image.jpg')
    save_dir = os.path.join(base_path, 'each_scale_code')

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Process single image
    print(f"\nProcessing: {image_path}")
    print(f"Saving to: {save_dir}")
    ok = process_image_to_code(image_path, vae, save_dir)
    if ok is False:
        print("Processing failed.")