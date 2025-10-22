import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_attention_map(attention_map_path, save_path=None, threshold: float = 0.5):
    """
    Visualize attention map using red-blue heatmap
    
    Args:
        attention_map_path: Path to .pt file containing attention map
        save_path: Path to save the visualization image. If None, displays the image
    """
    # Load attention map
    attention_map = torch.load(attention_map_path)
    print(f"Attention map shape: {attention_map.shape}")
    print(f"Attention map dtype: {attention_map.dtype}")
    
    # Convert to numpy array if it's a torch tensor
    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.cpu().numpy()
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Use red-blue heatmap
    # Red indicates high attention, blue indicates low attention
    im = plt.imshow(attention_map, cmap='RdBu_r', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', rotation=270, labelpad=15)
    
    # Calculate the value corresponding to white in colorbar (threshold position between min and max, default 0.5 as midpoint)
    white_value = attention_map.min() + float(threshold) * (attention_map.max() - attention_map.min())
    print(f"Colorbar white value: {white_value:.6f}")
    print(f"Data range: [{attention_map.min():.6f}, {attention_map.max():.6f}]")
    
    # Set title
    word_name = os.path.basename(attention_map_path).replace('_final_attention_map.pt', '')
    plt.title(f'Cross-Attention Map for "{word_name}" (64x64)', fontsize=14, fontweight='bold')
    
    # Set axis labels
    plt.xlabel('Width', fontsize=12)
    plt.ylabel('Height', fontsize=12)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display the image
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def generate_attention_mask(attention_map_path, save_path=None, coord_save_path=None, threshold: float = 0.5):
    """
    Generate cross-attention mask: areas above white threshold value set to black (0), others to white (1)
    Optionally save coordinates of black positions
    """
    # Load attention map
    attention_map = torch.load(attention_map_path)
    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.cpu().numpy()
    
    # Calculate white threshold value (between 0-1, default 0.5 as midpoint)
    white_value = attention_map.min() + float(threshold) * (attention_map.max() - attention_map.min())
    
    # Generate mask: areas above white value set to black (0), others to white (1)
    mask = np.ones_like(attention_map)
    mask[attention_map > white_value] = 0
    
    # Save mask image
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"Saved attention mask to: {save_path}")
        print(f"Mask threshold (white value): {white_value:.6f}")
    else:
        plt.show()
    plt.close()

    return mask, white_value

def visualize_attention_map_with_original_image(attention_map_path, original_image_path=None, save_path=None, threshold: float = 0.5):
    """
    Visualize attention map with option to overlay on original image
    
    Args:
        attention_map_path: Path to .pt file containing attention map
        original_image_path: Path to original image (optional)
        save_path: Path to save the visualization image
    """
    # Load attention map
    attention_map = torch.load(attention_map_path)
    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.cpu().numpy()
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # First subplot: pure attention map
    im1 = axes[0].imshow(attention_map, cmap='RdBu_r', interpolation='nearest')
    axes[0].set_title('Cross-Attention Map', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Width')
    axes[0].set_ylabel('Height')
    axes[0].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Attention Weight')
    
    # Calculate white threshold value
    white_value = attention_map.min() + float(threshold) * (attention_map.max() - attention_map.min())
    print(f"Colorbar white value: {white_value:.6f}")
    print(f"Data range: [{attention_map.min():.6f}, {attention_map.max():.6f}]")
    
    # Second subplot: overlay with original image if available
    if original_image_path and os.path.exists(original_image_path):
        from PIL import Image
        import cv2
        
        # Load original image
        original_img = Image.open(original_image_path)
        original_img = np.array(original_img)
        
        # Resize attention map to match original image dimensions
        attention_map_resized = cv2.resize(attention_map, (original_img.shape[1], original_img.shape[0]))
        
        # Normalize attention map to [0,1] range
        attention_map_normalized = (attention_map_resized - attention_map_resized.min()) / (attention_map_resized.max() - attention_map_resized.min())
        
        # Create heatmap
        heatmap = plt.cm.RdBu_r(attention_map_normalized)[:, :, :3]  # Take only RGB channels
        
        # Overlay on original image
        alpha = 0.6
        overlay = original_img * (1 - alpha) + heatmap * 255 * alpha
        
        axes[1].imshow(overlay.astype(np.uint8))
        axes[1].set_title('Attention Map Overlay', fontsize=12, fontweight='bold')
        axes[1].axis('off')
    else:
        # If no original image, show attention map statistics
        axes[1].hist(attention_map.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[1].set_title('Attention Weight Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Attention Weight')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
    
    # Set main title
    word_name = os.path.basename(attention_map_path).replace('_final_attention_map.pt', '')
    fig.suptitle(f'Cross-Attention Analysis for "{word_name}"', fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display the image
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize and export attention map and mask for a given example base path.")
    parser.add_argument("--base_path", type=str, required=True,
                        help="Base directory, e.g., /data1/.../example_case/example_3")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold in [0,1] to set the 'white' reference value between min and max (default: 0.5).")
    args = parser.parse_args()

    base_path = args.base_path.rstrip('/')
    attention_map_path = os.path.join(base_path, 'cross_attention_maps', 'final_attention_map.pt')
    output_dir = os.path.join(base_path, 'cross_attention_maps')
    os.makedirs(output_dir, exist_ok=True)

    # Check if file exists
    if not os.path.exists(attention_map_path):
        print(f"Error: File not found: {attention_map_path}")
        exit(1)

    # If original image exists, prepare for overlay visualization
    candidate_img = os.path.join(base_path, 'image', 'original_image.jpg')
    original_image_path = candidate_img if os.path.exists(candidate_img) else None

    # Visualization: with statistics/overlay
    save_vis = os.path.join(output_dir, "attention_analysis.png")
    visualize_attention_map_with_original_image(attention_map_path, save_path=save_vis, original_image_path=original_image_path, threshold=args.threshold)

    # Generate attention mask
    mask_output_dir = os.path.join(base_path, 'image')
    save_mask = os.path.join(mask_output_dir, "mask.png")
    mask, threshold = generate_attention_mask(attention_map_path, save_mask, coord_save_path=None, threshold=args.threshold)

    print("Visualization completed!")
