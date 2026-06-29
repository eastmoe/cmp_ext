#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>     // 修复：包含 uintptr_t

__device__ __forceinline__ float reciprocal_positive_rsqrt(float x) {
    return rsqrtf(__fmul_rn(x, x));
}

// 辅助函数：计算单个 float4 向量的 SiLU
// 优化策略：
// 1. 使用 __expf (FP32)
// 2. 避免 FP32 FMA，使用 __fadd_rn / __fmul_rn
// 3. 正分母倒数使用 rsqrtf，避免半精度 RCP 转换路径
__device__ __forceinline__ float4 silu_vec4(float4 v) {
    float4 res;

    // --- 步骤 1: 计算分母 (1 + exp(-x))，保持 FP32 ---
    // 约束 4: 使用 __expf
    // 约束 1: 使用 __fadd_rn 避免 FMA
    // 注意：一元负号 -v.x 通常编译为简单的符号位翻转，不需要 intrinsic
    float d_x = __fadd_rn(1.0f, __expf(-v.x));
    float d_y = __fadd_rn(1.0f, __expf(-v.y));
    float d_z = __fadd_rn(1.0f, __expf(-v.z));
    float d_w = __fadd_rn(1.0f, __expf(-v.w));

    // --- 步骤 2: 最终乘法 (x * (1 / d))，保持 FP32 ---
    // 约束 1: 使用 __fmul_rn 避免 FMA
    res.x = __fmul_rn(v.x, reciprocal_positive_rsqrt(d_x));
    res.y = __fmul_rn(v.y, reciprocal_positive_rsqrt(d_y));
    res.z = __fmul_rn(v.z, reciprocal_positive_rsqrt(d_z));
    res.w = __fmul_rn(v.w, reciprocal_positive_rsqrt(d_w));

    return res;
}

// 辅助函数：计算标量 SiLU
__device__ __forceinline__ float silu_scalar(float x) {
    // 约束 4 & 1: FP32 exp 和 add
    float denom = __fadd_rn(1.0f, __expf(-x));

    // 约束 1: FP32 mul
    return __fmul_rn(x, reciprocal_positive_rsqrt(denom));
}

__global__ void silu_kernel_fp32(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 检查指针是否对齐到 16 字节
    bool is_aligned = (reinterpret_cast<std::uintptr_t>(input) % 16 == 0) && 
                      (reinterpret_cast<std::uintptr_t>(output) % 16 == 0);

    if (is_aligned) {
        // --- 向量化路径 ---
        int n_vec = n >> 2; // n / 4
        
        const float4* in_vec = reinterpret_cast<const float4*>(input);
        float4* out_vec = reinterpret_cast<float4*>(output);

        for (int i = idx; i < n_vec; i += stride) {
            out_vec[i] = silu_vec4(in_vec[i]);
        }

        // 处理尾部
        int tail_start = n_vec << 2; 
        for (int i = tail_start + idx; i < n; i += stride) {
            output[i] = silu_scalar(input[i]);
        }
    } else {
        // --- 标量路径 ---
        for (int i = idx; i < n; i += stride) {
            output[i] = silu_scalar(input[i]);
        }
    }
}

void launch_silu_fp32(const float* input, float* output, int total_elements) {
    int threads = 256;
    // 向上取整计算 blocks
    int n_vec = (total_elements + 3) / 4;
    int blocks = (n_vec + threads - 1) / threads;
    
    silu_kernel_fp32<<<blocks, threads>>>(input, output, total_elements);
}
//[8192x8192]：0.412 ms
