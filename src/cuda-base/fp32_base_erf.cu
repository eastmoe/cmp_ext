#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// Abramowitz and Stegun 7.1.26 近似参数
#define ERF_P  0.3275911f
#define ERF_A1 0.254829592f
#define ERF_A2 -0.284496736f
#define ERF_A3 1.421413741f
#define ERF_A4 -1.453152027f
#define ERF_A5 1.061405429f

// 辅助函数：遵守约束 7 (强制使用 FP16 倒数指令)
__device__ __forceinline__ float custom_rcp_fp32_via_fp16(float x) {
    // 将 FP32 转为 FP16
    __half h = __float2half(x);
    // 使用 FP16 的倒数指令 (hrcp)
    __half r = hrcp(h);
    // 转回 FP32
    return __half2float(r);
}

// 辅助函数：ERF 核心数学逻辑
// 遵守约束 1 (无 FMA, 使用 __fmul_rn, __fadd_rn)
// 遵守约束 4 (Exp 使用 __expf)
__device__ __forceinline__ float compute_erf_core_fp32(float x) {
    // 符号处理
    float sign = (x >= 0.0f) ? 1.0f : -1.0f;
    float abs_x = fabsf(x);

    // t = 1.0 / (1.0 + p * x)
    // 必须使用 explicit mul/add
    float denom = __fadd_rn(1.0f, __fmul_rn(ERF_P, abs_x));
    
    // 必须使用 hrcp 路径计算倒数
    float t = custom_rcp_fp32_via_fp16(denom);

    // 多项式计算 (Horner's method, explicit mul/add)
    // poly = ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t
    
    float poly = __fadd_rn(__fmul_rn(ERF_A5, t), ERF_A4);
    poly = __fadd_rn(__fmul_rn(poly, t), ERF_A3);
    poly = __fadd_rn(__fmul_rn(poly, t), ERF_A2);
    poly = __fadd_rn(__fmul_rn(poly, t), ERF_A1);
    poly = __fmul_rn(poly, t); // 最后乘 t

    // exp_part = exp(-x * x)
    // 展开平方
    float x_sq = __fmul_rn(x, x);
    // 取负号 (0 - x^2)
    float neg_x_sq = __fsub_rn(0.0f, x_sq);
    // 使用 __expf (约束 4)
    float e_val = __expf(neg_x_sq);

    // result = 1 - poly * exp
    float term = __fmul_rn(poly, e_val);
    float res = __fsub_rn(1.0f, term);

    // 恢复符号
    return __fmul_rn(res, sign);
}

__global__ void erf_kernel_fp32(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = compute_erf_core_fp32(x);
    }
}

void launch_erf_fp32(const float* input, float* output, int total_elements) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    erf_kernel_fp32<<<blocks, threads>>>(input, output, total_elements);
}