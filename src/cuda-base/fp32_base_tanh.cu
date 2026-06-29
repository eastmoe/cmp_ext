#include <cuda_runtime.h>
// ----------------------------------------------------------------------
// FP32 Tanh Kernel
// 约束：
// 1. 不使用 FP32 FMA (分开 mul, add)
// 2. 基于 exp 展开: tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
// 3. 使用 rsqrtf 计算正分母倒数
// ----------------------------------------------------------------------

__global__ void tanh_kernel_fp32(const float* input, float* output, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float x = input[idx];

        // 1. 计算 2x
        // 禁止 FMA，使用 __fmul_rn
        float val_2x = __fmul_rn(x, 2.0f);

        // 2. 计算 e^(2x)
        // 使用 __expf (FP32 intrinsic)
        float exp_val = __expf(val_2x);

        // 3. 计算分子: e^(2x) - 1
        // 使用 __fadd_rn
        float numerator = __fadd_rn(exp_val, -1.0f);

        // 4. 计算分母: e^(2x) + 1
        // 使用 __fadd_rn
        float denominator = __fadd_rn(exp_val, 1.0f);

        float rcp_den = rsqrtf(__fmul_rn(denominator, denominator));

        // 6. 计算结果: numerator * rcp_den
        // 禁止 FMA
        float result = __fmul_rn(numerator, rcp_den);

        output[idx] = result;
    }
}

void launch_tanh_fp32(const float* input, float* output, int total_elements) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    tanh_kernel_fp32<<<blocks, threads>>>(input, output, total_elements);
}
