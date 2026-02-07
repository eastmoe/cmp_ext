#include <cuda_runtime.h>
#include <cuda_fp16.h> // 必须包含，用于 __float2half, hrcp, __half2float

// 预计算常量
// 原始公式系数: sqrt(2/pi) ≈ 0.7978845608
// Sigmoid 优化形式系数: -2 * sqrt(2/pi) ≈ -1.5957691216
#define NEG_2_SQRT_2_OVER_PI -1.5957691216f
#define GELU_COEF 0.044715f

__global__ void gelu_fp32_optimized_kernel(const float* __restrict__ input, float* __restrict__ output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];

        // -----------------------------------------------------------
        // 算法: GELU(x) ≈ x * sigmoid(1.702 * x) 的高精度变体
        // 原始 Tanh 近似: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // 展开消去 Tanh: x / (1 + exp( -2 * sqrt(2/pi) * (x + 0.044715 * x^3) ))
        // -----------------------------------------------------------

        // 1. 计算 x^3
        // [Constraint 1] 不要用 FP32 FMA，必须拆分为 __fmul_rn 和 __fadd_rn
        float x_sq = __fmul_rn(x, x);
        float x_cub = __fmul_rn(x_sq, x);

        // 2. 计算多项式内部: inner = x + 0.044715 * x^3
        float poly_term = __fmul_rn(GELU_COEF, x_cub);
        float inner = __fadd_rn(x, poly_term);

        // 3. 计算 exp 的指数部分: exp_arg = -1.59577 * inner
        float exp_arg = __fmul_rn(NEG_2_SQRT_2_OVER_PI, inner);

        // 4. 计算 Exp
        // [Constraint 4] 不要用 FP16 hexp，必须转为 FP32 使用 __expf
        float exp_val = __expf(exp_arg);

        // 5. 计算分母: den = 1.0 + exp_val
        float den = __fadd_rn(1.0f, exp_val);

        // 6. 计算倒数 (核心优化点)
        // [Constraint 7] 不要用 FP32 __frcp_rn，必须转为 FP16 使用 h2rcp (此处使用 scalar 版 hrcp)
        // [Constraint 2] 不使用 BF16，使用标准 FP16 (__half)
        __half h_den = __float2half(den);
        __half h_rcp = hrcp(h_den); // 使用硬件 FP16 倒数指令
        float rcp = __half2float(h_rcp);

        // 7. 最终结果: result = x * rcp
        float result = __fmul_rn(x, rcp);

        output[idx] = result;
    }
}

void launch_gelu_fp32(const float* input, float* output, int total_elements) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    gelu_fp32_optimized_kernel<<<blocks, threads>>>(input, output, total_elements);
}