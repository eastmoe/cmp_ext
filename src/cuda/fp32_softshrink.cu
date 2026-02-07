#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// 核心计算逻辑：无分支实现，严格使用指定intrinsic
// y = sign(x) * max(|x| - lambda, 0)
__device__ __forceinline__ float softshrink_core_func(float x, float neg_lambd) {
    // 计算 |x|
    float abs_x = fabsf(x);
    
    // 计算 |x| - lambda，即 |x| + (-lambda)
    // 规则1: 必须拆分为 __fadd_rn，禁止 FMA
    float val = __fadd_rn(abs_x, neg_lambd);
    
    // max(val, 0)
    float clipped = fmaxf(val, 0.0f);
    
    // 获取符号 sign(x)
    // copysignf 是位操作，效率极高
    float sgn = copysignf(1.0f, x);
    
    // 最终结果 = clipped * sgn
    // 规则1: 必须拆分为 __fmul_rn
    return __fmul_rn(clipped, sgn);
}

__global__ void softshrink_kernel_fp32(const float* input, float* output, int n, float lambd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 预计算 -lambda
    // 规则1: 乘法使用 __fmul_rn
    float neg_lambd = __fmul_rn(lambd, -1.0f);

    // ---------------------------------------------------
    // 1. 向量化处理部分 (Vectorized Path using float4)
    // ---------------------------------------------------
    // 将指针转换为 float4 指针，一次处理4个元素 (128位加载)
    const float4* input_v = reinterpret_cast<const float4*>(input);
    float4* output_v = reinterpret_cast<float4*>(output);
    
    // 计算向量化循环的上限
    int num_vec = n / 4;

    for (int i = idx; i < num_vec; i += stride) {
        float4 in_val = input_v[i]; // 128-bit Load
        float4 out_val;

        // 分别处理4个分量
        out_val.x = softshrink_core_func(in_val.x, neg_lambd);
        out_val.y = softshrink_core_func(in_val.y, neg_lambd);
        out_val.z = softshrink_core_func(in_val.z, neg_lambd);
        out_val.w = softshrink_core_func(in_val.w, neg_lambd);

        output_v[i] = out_val; // 128-bit Store
    }

    // ---------------------------------------------------
    // 2. 标量处理尾部 (Scalar Path for Tail)
    // ---------------------------------------------------
    // 处理剩余不足4个的元素，或者 n < 4 的情况
    int tail_start = num_vec * 4;
    
    // 依然使用 grid-stride 模式处理尾部
    for (int i = tail_start + idx; i < n; i += stride) {
        output[i] = softshrink_core_func(input[i], neg_lambd);
    }
}

void launch_softshrink_fp32(const float* input, float* output, int total_elements, float lambd) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    softshrink_kernel_fp32<<<blocks, threads>>>(input, output, total_elements, lambd);
}