# Example3 图像编辑管道使用说明

## 功能概述
Example3 是一个智能图像编辑管道，具有以下核心特点：
- **自动背景保留**：无需人工提供遮罩图像，通过注意力机制自动识别前景和背景
- **智能遮罩生成**：基于注意力图自动生成前景遮罩和背景保留权重
- **两阶段训练**：先训练语言风格嵌入，再训练LoRA模型，确保编辑质量

## 前置准备
在运行管道前，请确保以下文件已准备就绪：

### 必需文件
1. **原始图像**：`image/original_image.jpg`
2. **原图描述**：`prompt/original_image_prompt.txt` - 描述原始图像的内容
3. **编辑提示词**：`prompt/edit_image_prompt.txt` - 描述希望如何编辑图像
4. **目标关键词**：`prompt/target_word.txt` - 指定需要关注的目标词汇

### 环境要求
- 确保所有依赖环境已正确安装
- 设置正确的Python路径和CUDA环境

## 管道执行流程

### 步骤1：数据准备
**脚本：** `prepare_edit.sh`
**功能：**
- 生成训练数据集的分割文件（JSONL格式）
- 提取图像的多尺度编码特征
- 为后续训练准备必要的数据结构

### 步骤2：语言风格嵌入训练
**脚本：** `train_EditInfinity_example3.sh`
**参数设置：**
```bash
train_textembedding=1          # 启用语言风格嵌入训练
train_textembedding_iter=10    # 训练10个迭代周期
use_textembedding=0            # 训练阶段不使用预训练嵌入
train_lora=0                   # 此阶段不训练LoRA
```

### 步骤3：LoRA模型训练
**脚本：** `train_EditInfinity_example3.sh`
**参数设置：**
```bash
train_textembedding=0          # 停止语言风格嵌入训练
use_textembedding=1            # 使用步骤2训练的语言风格嵌入
use_textembedding_iter=10      # 使用第10个迭代的嵌入权重
train_lora=1                   # 启用LoRA训练
train_lora_iter=20             # 训练20个迭代周期
```

**替代方案：** 如果不想加载语言风格嵌入，可以设置：
```bash
train_textembedding=0
use_textembedding=0            # 不加载语言风格嵌入
train_lora=1
train_lora_iter=20
```

### 步骤4：注意力图生成
**脚本：** `get_targetword_attentionmap_example3.sh`
**功能：**
- 使用训练好的模型生成目标词的注意力图
- 注意力图用于识别图像中需要编辑的前景区域
- 设置 `infer_function=1` 专门用于生成注意力图

### 步骤5：遮罩和权重张量生成
**脚本：** `get_weighted_tensor.sh`
**功能：**
- 基于注意力图生成前景遮罩图像（`mask.png`）
- 创建背景前景保留的权重张量
- 生成注意力图的可视化分析图像

### 步骤6：最终图像推理
**脚本：** `infer_EditInfinity_example3.sh`
**参数设置：**
```bash
infer_function=2               # 用于最终图像生成
use_concat_embedding=1         # 使用拼接的语言风格嵌入
use_embedding_iter=10          # 使用第10个迭代的嵌入
use_lora=1                     # 使用LoRA模型
use_lora_iter=20               # 使用第20个迭代的LoRA权重
```

## 参数说明

### 训练参数
- **`train_textembedding`**：是否启动语言风格嵌入训练（0/1）
- **`train_textembedding_iter`**：语言风格嵌入训练的迭代次数
- **`use_textembedding`**：是否使用预训练的语言风格嵌入（0/1）
- **`use_textembedding_iter`**：使用哪个迭代的语言风格嵌入权重
- **`train_lora`**：是否启动LoRA训练（0/1）
- **`train_lora_iter`**：LoRA训练的迭代次数

### 推理参数
- **`infer_function`**：
  - `1`：生成注意力图
  - `2`：生成最终编辑图像
- **`use_concat_embedding`**：是否使用拼接的语言风格嵌入（0/1）
- **`use_embedding_iter`**：使用哪个迭代的语言风格嵌入
- **`use_lora`**：是否使用LoRA模型（0/1）
- **`use_lora_iter`**：使用哪个迭代的LoRA权重

## 快速开始
1. 准备必需的文件（图像和提示词文件）
2. 运行 `edit_pipeline.sh` 开始自动化处理
3. 等待管道执行完成
4. 查看输出目录中的结果文件

## 输出结果
管道执行完成后，将生成：
- 训练好的模型权重文件
- 注意力图和分析图像
- 前景遮罩图像（`mask.png`）
- 最终编辑后的图像（保留原始背景）

## 注意事项
- 可以仅使用语言风格嵌入或仅使用LoRA，只需将对应的参数设置为0
- 确保各步骤的迭代次数设置一致，避免加载不存在的权重
- 建议按照推荐的参数设置进行首次运行，后续可根据需要调整