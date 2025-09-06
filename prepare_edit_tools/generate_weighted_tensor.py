"""
Generate Weighted Tensor from Mask Images

This script processes mask images to generate weighted tensors for training purposes.
The script can process either a single case or all cases in a directory.

Input Files:
- mask.png: Located in {case_path}/image/mask.png
  - Should be a binary mask image (0=white/non-mask, 255=black/mask)
  - Will be automatically converted to binary format

Output Files:
- weighted_tensor.pt: Saved in {case_path}/weight_tensor/weighted_tensor.pt
  - PyTorch tensor file (.pt) containing 64x64 weight matrix
  - Values: 0.0 (mask region), 0.0-1.0 (transition zone), 1.0 (background)
- weighted_tensor_heatmap.png: Saved in {case_path}/image/weighted_tensor_heatmap.png
  - Visual heatmap of the weight matrix for verification

Usage:
  # Process single case
  python generate_weighted_tensor.py --case_path /path/to/case/folder
  
  # Process all cases in directory
  python generate_weighted_tensor.py --base_dir /path/to/base/directory
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse
from PIL import Image
from scipy.spatial.distance import cdist


def mask_to_coord(mask, num_blocks, black_ratio_threshold=0.2):
    """
    Divide mask image into num_blocks x num_blocks blocks, count black pixel ratio in each block,
    record (i, j) coordinates if black ratio >= threshold.
    
    Args:
        mask: Binary mask array (0=white, 1=black)
        num_blocks: Number of blocks to divide the image
        black_ratio_threshold: Threshold for black pixel ratio
    
    Returns:
        List of (i, j) coordinates where black ratio exceeds threshold
    """
    height, width = mask.shape

    # Calculate block size, handle non-divisible cases
    block_height = round(height / num_blocks)
    block_width = round(width / num_blocks)

    output_coordinates = []
    for i in range(num_blocks):
        for j in range(num_blocks):
            # Calculate block start and end positions
            row_start = min(i * block_height, height - 1)
            row_end = min((i + 1) * block_height, height)
            col_start = min(j * block_width, width - 1)
            col_end = min((j + 1) * block_width, width)

            # Ensure at least one pixel
            if row_end <= row_start:
                row_end = row_start + 1
            if col_end <= col_start:
                col_end = col_start + 1

            block = mask[row_start:row_end, col_start:col_end]
            black_pixels = np.sum(block == 0)
            black_ratio = black_pixels / block.size

            if black_ratio >= black_ratio_threshold:
                output_coordinates.append((i, j))

    return output_coordinates


def generate_weight_matrix(coord_64, coord_4):
    """
    Generate weight matrix based on coordinate files using linear gradient approach
    
    Args:
        coord_64: List of coordinates for 64x64 blocks
        coord_4: List of coordinates for 4x4 blocks
    
    Returns:
        64x64 weight matrix with values:
        - 0.0: Mask region and adjacent areas (distance <= 1)
        - 0.0-1.0: Transition zone (distance 1-4, linear interpolation)
        - 1.0: Background areas
    """
    weighted_matrix = np.full((64, 64), -1, dtype=float)
    
    replaced_coordinates = [coord for coord in coord_64]
    replaced_block_coordinates = [coord for coord in coord_4]
    
    if len(replaced_coordinates) == 0:
        # If no coordinates, set all positions to 1
        weighted_matrix[:, :] = 1.0
        return weighted_matrix
    
    replaced_array = np.array(replaced_coordinates)
    all_positions = np.array(np.meshgrid(np.arange(64), np.arange(64))).T.reshape(-1, 2)
    distances = cdist(all_positions, replaced_array, metric='cityblock') 
    
    if len(replaced_block_coordinates) < 6:
        # Linear gradient logic
        close_to_mask = np.any(distances <= 1, axis=1)
        
        for (i, j) in all_positions[close_to_mask]:
            weighted_matrix[i, j] = 0.0

        # Calculate minimum distance from each pixel to mask region
        min_distances = np.min(distances, axis=1)
        
        # Handle transition zone with distance (1, 4]
        transition_zone = (min_distances > 1) & (min_distances <= 4)
        for idx, (i, j) in enumerate(all_positions[transition_zone]):
            d = min_distances[transition_zone][idx]
            weighted_matrix[i, j] = (d - 1) / 4  # Linear gradient formula
        
        # Set remaining positions to 1.0
        center = (31.5, 31.5)
        remaining_positions = np.argwhere(weighted_matrix == -1)
        for (i, j) in remaining_positions:
            distance_to_center = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            max_distance = np.sqrt((0 - center[0]) ** 2 + (0 - center[1]) ** 2)
            normalized_distance = distance_to_center / max_distance
            weighted_matrix[i, j] = 1.0 
    else:
        # Mask region and distance=1 areas set to 0, others set to 1
        close_to_mask = np.any(distances <= 1, axis=1)
        weighted_matrix[:, :] = 1.0  # Set all positions to 1
        for (i, j) in all_positions[close_to_mask]:
            weighted_matrix[i, j] = 0.0  
    
    return weighted_matrix


def process_single_case(base_dir, case_path):
    """
    Process a single case: complete workflow from mask.png to weighted tensor
    
    Args:
        base_dir: Base directory path
        case_path: Path to the specific case folder
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n==============================================")
    print(f"Processing: {case_path}")
    print(f"==============================================")
    
    # 1. Find mask.png file
    mask_path = os.path.join(case_path, "image", "mask.png")
    if not os.path.exists(mask_path):
        print(f"Error: Mask file not found: {mask_path}")
        return False
    
    try:
        # 2. Load mask image
        print("Step 1: Loading mask image...")
        with Image.open(mask_path) as img:
            # Convert to grayscale and convert to numpy array
            if img.mode != 'L':
                img = img.convert('L')
            mask = np.array(img)
        
        # Ensure mask is binary (0 and 255)
        mask = (mask > 128).astype(np.uint8) * 255
        # Convert to 0 and 1 (0=white, 1=black)
        # In mask: 0=white (non-mask region), 1=black (mask region)
        mask = (mask == 255).astype(np.uint8)
        
        print(f"Mask shape: {mask.shape}")
        print(f"Mask value range: {mask.min()} to {mask.max()}")
        
        # 3. Generate 64x64 and 4x4 coordinates
        print("Step 2: Generating coordinates...")
        coord_64 = mask_to_coord(mask, 64, black_ratio_threshold=0.2)
        coord_4 = mask_to_coord(mask, 4, black_ratio_threshold=0.2)
        print(f"Found {len(coord_64)} coordinates for 64x64, {len(coord_4)} coordinates for 4x4")
        
        # 4. Generate weight matrix
        print("Step 3: Generating weight matrix...")
        weighted_matrix = generate_weight_matrix(coord_64, coord_4)
        
        # 5. Create output directories
        weight_dir = os.path.join(case_path, "weight_tensor")
        image_dir = os.path.join(case_path, "image")
        os.makedirs(weight_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        
        # 6. Save weight tensor
        print("Step 4: Saving weighted tensor...")
        weighted_tensor = torch.tensor(weighted_matrix, dtype=torch.float32)
        tensor_path = os.path.join(weight_dir, "weighted_tensor.pt")
        torch.save(weighted_tensor, tensor_path)
        
        # 7. Save heatmap
        print("Step 5: Saving heatmap...")
        plt.figure(figsize=(8, 8))
        plt.imshow(weighted_matrix, cmap='coolwarm', vmin=0, vmax=1)
        plt.colorbar(label='Value')
        plt.title('Weight Matrix Heatmap')
        plt.axis('off')
        heatmap_path = os.path.join(image_dir, "weighted_tensor_heatmap.png")
        plt.savefig(heatmap_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Successfully processed: {case_path}")
        print(f"Output saved to: {weight_dir} and {image_dir}")
        return True
        
    except Exception as e:
        print(f"Error processing {case_path}: {str(e)}")
        return False


def process_all_cases(base_dir):
    """
    Process all cases in base_dir directory
    
    Args:
        base_dir: Base directory containing case folders
    """
    # Find all numeric folders (e.g., 000000000000, 000000000001, etc.)
    pattern = os.path.join(base_dir, "*")
    all_dirs = glob.glob(pattern)
    
    # Filter numeric folders
    cases_to_process = []
    for dir_path in all_dirs:
        if os.path.isdir(dir_path) and os.path.basename(dir_path).isdigit():
            mask_path = os.path.join(dir_path, "image", "mask.png")
            if os.path.exists(mask_path):
                cases_to_process.append(dir_path)
    
    print(f"Found {len(cases_to_process)} cases to process")
    
    success_count = 0
    for case_path in cases_to_process:
        if process_single_case(base_dir, case_path):
            success_count += 1
    
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {success_count}/{len(cases_to_process)} cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate weighted tensor from mask images.')
    parser.add_argument('--base_dir', type=str, help='Base directory containing case folders')
    parser.add_argument('--case_path', type=str, help='Specific case folder path to process')
    
    args = parser.parse_args()
    
    if args.case_path:
        # Process single case
        base_dir = os.path.dirname(args.case_path) if args.base_dir is None else args.base_dir
        process_single_case(base_dir, args.case_path)
    elif args.base_dir:
        # Process all cases
        process_all_cases(args.base_dir)
    else:
        print("Error: Must specify either --case_path or --base_dir")
        print("Usage:")
        print("  # Process single case")
        print("  python generate_weighted_tensor.py --case_path /path/to/case/folder")
        print("  # Process all cases in directory")
        print("  python generate_weighted_tensor.py --base_dir /path/to/base/directory")
        exit(1)