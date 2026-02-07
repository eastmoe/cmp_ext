import multiprocessing
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# pip install -e . --no-build-isolation

# 获取CPU核心数
max_workers = multiprocessing.cpu_count()

setup(
    name='cmpext3',
    ext_modules=[
        CUDAExtension(
            name='cmpext3',
            sources=[
                'src/main.cpp',
                'src/cuda/fp16_matmul.cu',
                'src/cuda/fp32_matmul.cu', 
                'src/cuda/bf16_matmul.cu',
                'src/cuda/fp16_conv.cu',
                'src/cuda/fp32_conv.cu',
                'src/cuda/bf16_conv.cu',
                'src/cuda/fp16_emb.cu',
                'src/cuda/fp32_emb.cu',
                'src/cuda/bf16_emb.cu',
                'src/cuda/fp16_ConvTranspose2d.cu',
                'src/cuda/fp32_ConvTranspose2d.cu',
                'src/cuda/bf16_ConvTranspose2d.cu',
                'src/cuda/fp16_groupnorm.cu',
                'src/cuda/fp32_groupnorm.cu',
                'src/cuda/bf16_groupnorm.cu',
                'src/cuda/fp16_layernorm.cu',
                'src/cuda/fp32_layernorm.cu',
                'src/cuda/bf16_layernorm.cu',
                'src/cuda/fp16_rmsnorm.cu',
                'src/cuda/fp32_rmsnorm.cu',
                'src/cuda/bf16_rmsnorm.cu',
                'src/cuda/fp16_attention.cu', 
                'src/cuda/fp32_attention.cu', 
                'src/cuda/bf16_attention.cu',
                #'src/cuda/fp16_upsample.cu',
                #'src/cuda/fp32_upsample.cu',
                #'src/cuda/bf16_upsample.cu',
                'src/cuda/fp16_gelu.cu',
                'src/cuda/fp32_gelu.cu',
                'src/cuda/bf16_gelu.cu',
                'src/cuda/fp16_silu.cu',
                'src/cuda/fp32_silu.cu',
                'src/cuda/bf16_silu.cu',
                'src/cuda/fp16_swish.cu',
                'src/cuda/fp32_swish.cu',
                'src/cuda/bf16_swish.cu',
                'src/cuda/fp16_mish.cu',
                'src/cuda/fp32_mish.cu',
                'src/cuda/bf16_mish.cu',
                'src/cuda/fp16_softmax.cu',
                'src/cuda/fp32_softmax.cu',
                'src/cuda/bf16_softmax.cu',
                'src/cuda/fp16_softplus.cu',
                'src/cuda/fp32_softplus.cu',
                'src/cuda/bf16_softplus.cu',
                'src/cuda/fp16_softsign.cu',
                'src/cuda/fp32_softsign.cu',
                'src/cuda/bf16_softsign.cu',
                'src/cuda/fp16_softshrink.cu',
                'src/cuda/fp32_softshrink.cu',
                'src/cuda/bf16_softshrink.cu',
                'src/cuda-base/fp16_base_tanh.cu',
                'src/cuda-base/fp32_base_tanh.cu',
                'src/cuda-base/bf16_base_tanh.cu',
                'src/cuda-base/fp16_base_erf.cu',
                'src/cuda-base/fp32_base_erf.cu',
                'src/cuda-base/bf16_base_erf.cu'
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', 
                    #'-std=c++14', 
                    '--use_fast_math', 
                    '--ptxas-options=-v',
                    '--fmad=false',  # <--- 关键：全局禁止生成 FMA 指令
                    # 必须显式 "Undefine" PyTorch 自动添加的禁用 Half 的宏
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_HALF2_OPERATORS__',
                    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False, max_workers=max_workers)
    }
)
