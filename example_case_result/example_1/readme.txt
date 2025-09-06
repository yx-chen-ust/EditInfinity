# Example1 图像编辑管道使用说明

## 功能概述
Example1 是一个基础图像编辑管道，具有以下核心特点：
- **基础编辑功能**：专注于基本的图像编辑，无需复杂的背景保留操作
- **两阶段训练**：先训练语言风格嵌入，再训练LoRA模型，确保编辑质量
- **自动化处理**：完整的端到端处理流程，减少人工干预

## 前置准备
在运行管道前，请确保以下文件已准备就绪：

### 必需文件
1. **原始图像**：`image/original_image.jpg`
2. **编辑提示词**：`prompt/edit_image_prompt.txt` - 描述希望如何编辑图像

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
**脚本：** `train_EditInfinity_example1.sh`
**参数设置：**
```bash
train_textembedding=1          # 启用语言风格嵌入训练
train_textembedding_iter=10    # 训练10个迭代周期
use_textembedding=0            # 训练阶段不使用预训练嵌入
train_lora=0                   # 此阶段不训练LoRA
```

### 步骤3：LoRA模型训练
**脚本：** `train_EditInfinity_example1.sh`
**参数设置：**
```bash
train_textembedding=0          # 停止语言风格嵌入训练
use_textembedding=1            # 使用步骤2训练的语言风格嵌入
use_textembedding_iter=10      # 使用第10个迭代的嵌入权重
train_lora=1                   # 启用LoRA训练
train_lora_iter=50             # 训练50个迭代周期
```

**替代方案：** 如果不想加载语言风格嵌入，可以设置：
```bash
train_textembedding=0
use_textembedding=0            # 不加载语言风格嵌入
train_lora=1
train_lora_iter=50
```

### 步骤4：最终图像推理
**脚本：** `infer_EditInfinity_example1.sh`
**参数设置：**
```bash
infer_function=0               # 基础图像编辑功能
use_concat_embedding=1         # 使用拼接的语言风格嵌入
use_embedding_iter=10          # 使用第10个迭代的嵌入
use_lora=1                     # 使用LoRA模型
use_lora_iter=50               # 使用第50个迭代的LoRA权重
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
- **`infer_function`**：推理功能类型（0：基础图像编辑）
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
- 最终编辑后的图像

## 与其他例子的区别
- **Example1**：基础图像编辑，无需背景保留操作
- **Example2**：需要用户提供遮罩图像进行背景保留
- **Example3**：通过注意力机制自动生成遮罩图像进行背景保留

## 注意事项
- 可以仅使用语言风格嵌入或仅使用LoRA，只需将对应的参数设置为0
- 确保各步骤的迭代次数设置一致，避免加载不存在的权重
- 建议按照推荐的参数设置进行首次运行，后续可根据需要调整
- 此管道专注于基本编辑功能，如需背景保留请使用Example2或Example3