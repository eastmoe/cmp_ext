# cmpext3 - PyTorch CUDA扩展

这是一个基于PyTorch的CUDA扩展，主要用于验证论文 **[《Instruction-Level Performance Analysis and Optimization Strategies for Constrained AI Accelerators A Case Study of CMP 170HX》](./paper/paper_20260208.pdf).** 中提出的计算原理。本扩展工程价值有限，主要作为原理性验证和实验使用。

## 包含的算子

扩展实现了以下算子在三种精度（FP16、FP32、BF16）下的CUDA实现：

### 核心计算算子
- **矩阵乘法** (MatMul): `fp16/bf16/fp32_matmul`
- **卷积** (Conv2d): `fp16/bf16/fp32_conv`
- **反卷积** (ConvTranspose2d): `fp16/bf16/fp32_ConvTranspose2d`
- **注意力机制** (Attention): `fp16/bf16/fp32_attention`

### 归一化层
- **层归一化** (LayerNorm): `fp16/bf16/fp32_layernorm`
- **组归一化** (GroupNorm): `fp16/bf16/fp32_groupnorm`
- **RMS归一化** (RMSNorm): `fp16/bf16/fp32_rmsnorm`

### 激活函数
- **GELU**: `fp16/bf16/fp32_gelu`
- **SiLU**: `fp16/bf16/fp32_silu`
- **Swish**: `fp16/bf16/fp32_swish`
- **Mish**: `fp16/bf16/fp32_mish`
- **Softmax系列**: Softmax, Softplus, Softsign, Softshrink

### 基础数学函数
- **Tanh**: `fp16/bf16/fp32_base_tanh`
- **Erf**: `fp16/bf16/fp32_base_erf`

### 嵌入层
- **Embedding**: `fp16/bf16/fp32_emb`

## 安装依赖

### 必需依赖
- **Python**
- **PyTorch** (需要与CUDA版本匹配)
- **CUDA Toolkit**
- **GPU**: NVIDIA GPU 

### 开发依赖
- **ninja** (可选，默认禁用)
- **setuptools**

## 安装方法

### 1. 前置准备
确保已安装正确版本的PyTorch和CUDA Toolkit：
```bash
# 示例：安装PyTorch (请根据您的CUDA版本选择)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 2.下载并安装扩展
```
# 从GitHub下载源码
git clone https://github.com/eastmoe/cmp_ext
# 从源码安装
cd cmp_ext
pip install -e . --no-build-isolation
```
### 3.验证安装
```
import torch
import cmpext3

# 检查CUDA是否可用
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Extension loaded: {cmpext3.__file__}")
```
或者运行测试工具
```
python tools/bench.py
```

## 实现目的

**主要目的**：本扩展是论文《XXXX》（请替换为实际论文名称）中计算原理的验证实现。重点在于验证CMP 170HX上的限制规避策略。

**工程价值**：本扩展主要用于研究验证，不适用于生产环境。可能存在以下限制：性能未充分优化（BF16\Attention等）。缺乏完整的错误处理。测试覆盖有限。

## 文件结构

```
.
├── setup.py              # 安装配置文件
├── src/
│   ├── main.cpp         # Python绑定入口
│   ├── cuda/            # 主算子实现
│   └── cuda-base/       # 基础数学函数
```

## 许可证
MIT

## ComfyUI 扩展
参照[ComfyUI-CMP-Extention](https://github.com/eastmoe/ComfyUI-CMP-Extention)以在ComfyUI中获得加速

