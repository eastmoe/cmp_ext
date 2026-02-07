
# cmpext3 - PyTorch CUDA Extension

[中文版](README_zh.md)

This is a CUDA extension based on PyTorch, primarily used to verify the computational principles proposed in the paper **"XXXX"**. The engineering value of this extension is limited; it is mainly intended for principle verification and experimental use.

## Included Operators

The extension implements CUDA versions of the following operators in three precisions (FP16, FP32, BF16):

### Core Computation Operators
- **Matrix Multiplication** (MatMul): `fp16/bf16/fp32_matmul`
- **Convolution** (Conv2d): `fp16/bf16/fp32_conv`
- **Transposed Convolution** (ConvTranspose2d): `fp16/bf16/fp32_ConvTranspose2d`
- **Attention Mechanism** (Attention): `fp16/bf16/fp32_attention`

### Normalization Layers
- **Layer Normalization** (LayerNorm): `fp16/bf16/fp32_layernorm`
- **Group Normalization** (GroupNorm): `fp16/bf16/fp32_groupnorm`
- **RMS Normalization** (RMSNorm): `fp16/bf16/fp32_rmsnorm`

### Activation Functions
- **GELU**: `fp16/bf16/fp32_gelu`
- **SiLU**: `fp16/bf16/fp32_silu`
- **Swish**: `fp16/bf16/fp32_swish`
- **Mish**: `fp16/bf16/fp32_mish`
- **Softmax Series**: Softmax, Softplus, Softsign, Softshrink

### Basic Mathematical Functions
- **Tanh**: `fp16/bf16/fp32_base_tanh`
- **Erf**: `fp16/bf16/fp32_base_erf`

### Embedding Layer
- **Embedding**: `fp16/bf16/fp32_emb`

## Installation Dependencies

### Required Dependencies
- **Python**
- **PyTorch** (Must match your CUDA version)
- **CUDA Toolkit**
- **GPU**: NVIDIA GPU 

### Development Dependencies
- **ninja** (Optional, disabled by default)
- **setuptools**

## Installation Instructions

### 1. Prerequisites
Ensure you have the correct versions of PyTorch and CUDA Toolkit installed:
```bash
# Example: Installing PyTorch (Please choose according to your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 2. Install Extension
```bash
# Install from source
pip install -e . --no-build-isolation
```

### 3. Verify Installation
```python
import torch
import cmpext3

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Extension loaded: {cmpext3.__file__}")
```
Or run the test tool:
```bash
python tools/bench.py
```

## Implementation Purpose

**Main Purpose**: This extension is a verification implementation of the computational principles in the paper **"XXXX"** (please replace with the actual paper title). The focus is on verifying restriction evasion strategies on the CMP 170HX.

**Engineering Value**: This extension is primarily for research verification and is not suitable for production environments. There may be the following limitations: Performance is not fully optimized (e.g., BF16/Attention, etc.). Lack of comprehensive error handling. Limited test coverage.

## File Structure

```
.
├── setup.py              # Installation configuration file
├── src/
│   ├── main.cpp         # Python binding entry point
│   ├── cuda/            # Main operator implementation
│   └── cuda-base/       # Basic mathematical functions
```

## License
MIT

## ComfyUI Extension
Refer to [ComfyUI-CMP-Extention](https://github.com/eastmoe/ComfyUI-CMP-Extention) to get acceleration in ComfyUI.
