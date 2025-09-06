import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_attention_map(attention_map_path, save_path=None, threshold: float = 0.5):
    """
    可视化attention map，使用红蓝热力图
    
    Args:
        attention_map_path: attention map的.pt文件路径
        save_path: 保存图片的路径，如果为None则显示图片
    """
    # 加载attention map
    attention_map = torch.load(attention_map_path)
    print(f"Attention map shape: {attention_map.shape}")
    print(f"Attention map dtype: {attention_map.dtype}")
    
    # 转换为numpy数组
    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.cpu().numpy()
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 使用红蓝热力图
    # 红色表示高注意力，蓝色表示低注意力
    im = plt.imshow(attention_map, cmap='RdBu_r', interpolation='nearest')
    
    # 添加颜色条
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', rotation=270, labelpad=15)
    
    # 获取颜色条中白色对应的数值（位于 [min, max] 的阈值位置，默认 0.5 为中点）
    white_value = attention_map.min() + float(threshold) * (attention_map.max() - attention_map.min())
    print(f"Colorbar white value: {white_value:.6f}")
    print(f"Data range: [{attention_map.min():.6f}, {attention_map.max():.6f}]")
    
    # 设置标题
    word_name = os.path.basename(attention_map_path).replace('_final_attention_map.pt', '')
    plt.title(f'Cross-Attention Map for "{word_name}" (64x64)', fontsize=14, fontweight='bold')
    
    # 设置坐标轴
    plt.xlabel('Width', fontsize=12)
    plt.ylabel('Height', fontsize=12)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def generate_attention_mask(attention_map_path, save_path=None, coord_save_path=None, threshold: float = 0.5):
    """
    生成cross-attention mask：大于白色对应值的部分设为黑色，其余为白色，并可保存黑色位置坐标
    """
    # 加载attention map
    attention_map = torch.load(attention_map_path)
    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.cpu().numpy()
    
    # 计算白色对应的数值（阈值 0-1，默认 0.5 为中点）
    white_value = attention_map.min() + float(threshold) * (attention_map.max() - attention_map.min())
    
    # 生成mask：大于白色值的设为黑色（0），其余为白色（1）
    mask = np.ones_like(attention_map)
    mask[attention_map > white_value] = 0
    
    # 保存mask图片
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
    可视化attention map，可以选择叠加到原始图像上
    
    Args:
        attention_map_path: attention map的.pt文件路径
        original_image_path: 原始图像路径（可选）
        save_path: 保存图片的路径
    """
    # 加载attention map
    attention_map = torch.load(attention_map_path)
    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.cpu().numpy()
    
    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 第一个子图：纯attention map
    im1 = axes[0].imshow(attention_map, cmap='RdBu_r', interpolation='nearest')
    axes[0].set_title('Cross-Attention Map', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Width')
    axes[0].set_ylabel('Height')
    axes[0].grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Attention Weight')
    
    # 获取颜色条中白色对应的数值
    white_value = attention_map.min() + float(threshold) * (attention_map.max() - attention_map.min())
    print(f"Colorbar white value: {white_value:.6f}")
    print(f"Data range: [{attention_map.min():.6f}, {attention_map.max():.6f}]")
    
    # 第二个子图：如果有原始图像，则叠加显示
    if original_image_path and os.path.exists(original_image_path):
        from PIL import Image
        import cv2
        
        # 加载原始图像
        original_img = Image.open(original_image_path)
        original_img = np.array(original_img)
        
        # 将attention map resize到原始图像大小
        attention_map_resized = cv2.resize(attention_map, (original_img.shape[1], original_img.shape[0]))
        
        # 归一化attention map到0-1范围
        attention_map_normalized = (attention_map_resized - attention_map_resized.min()) / (attention_map_resized.max() - attention_map_resized.min())
        
        # 创建热力图
        heatmap = plt.cm.RdBu_r(attention_map_normalized)[:, :, :3]  # 只取RGB通道
        
        # 叠加到原始图像上
        alpha = 0.6
        overlay = original_img * (1 - alpha) + heatmap * 255 * alpha
        
        axes[1].imshow(overlay.astype(np.uint8))
        axes[1].set_title('Attention Map Overlay', fontsize=12, fontweight='bold')
        axes[1].axis('off')
    else:
        # 如果没有原始图像，显示attention map的统计信息
        axes[1].hist(attention_map.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[1].set_title('Attention Weight Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Attention Weight')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
    
    # 设置总标题
    word_name = os.path.basename(attention_map_path).replace('_final_attention_map.pt', '')
    fig.suptitle(f'Cross-Attention Analysis for "{word_name}"', fontsize=16, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示图片
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

    # 检查文件是否存在
    if not os.path.exists(attention_map_path):
        print(f"Error: File not found: {attention_map_path}")
        exit(1)

    # 如果存在原始图像则叠加可视化
    candidate_img = os.path.join(base_path, 'image', 'original_image.jpg')
    original_image_path = candidate_img if os.path.exists(candidate_img) else None

    # 可视化：带统计信息/叠加原图
    save_vis = os.path.join(output_dir, "attention_analysis.png")
    visualize_attention_map_with_original_image(attention_map_path, save_path=save_vis, original_image_path=original_image_path, threshold=args.threshold)

    # 生成 attention mask
    mask_output_dir = os.path.join(base_path, 'image')
    save_mask = os.path.join(mask_output_dir, "mask.png")
    mask, threshold = generate_attention_mask(attention_map_path, save_mask, coord_save_path=None, threshold=args.threshold)

    print("Visualization completed!")