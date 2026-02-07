#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ----------------------------------------------------------------------
// FP16 Tanh Kernel
// 约束：
// 1. 不要使用 hexp, hsin 等，必须转为 FP32 使用 __expf
// 2. 不要使用 FP32 RCP，必须基于 FP16 倒数
// 3. 展开公式
// ----------------------------------------------------------------------

__global__ void tanh_kernel_fp16(const __half* input, __half* output, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        __half h_in = input[idx];
        
        // 转换到 FP32 进行核心数学运算
        float x = __half2float(h_in);

        // 1. 计算 2x (FP32)
        float val_2x = __fmul_rn(x, 2.0f);

        // 2. 计算 e^(2x) (FP32)
        float exp_val = __expf(val_2x);

        // 3. 分子 (FP32)
        float numerator = __fadd_rn(exp_val, -1.0f);

        // 4. 分母 (FP32)
        float denominator = __fadd_rn(exp_val, 1.0f);

        // 5. 倒数处理 (Constraint 7)
        // 既然我们在 FP16 kernel 里，也要求 "必须基于 FP16 计算倒数"
        __half h_den = __float2half(denominator);
        __half h_rcp = hrcp(h_den);
        // 虽然 h_rcp 已经是 FP16，但为了符合"全程用 float __fmul 计算"的逻辑，
        // 我们将其转回 float 与 numerator 相乘
        float rcp_den = __half2float(h_rcp);

        // 6. 结果 (FP32)
        float res_f = __fmul_rn(numerator, rcp_den);

        // 转换回 FP16 输出
        output[idx] = __float2half(res_f);
    }
}

void launch_tanh_fp16(const void* input, void* output, int total_elements) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    // 修正错误：增加了 <<<blocks, threads>>>
    tanh_kernel_fp16<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(input), 
        reinterpret_cast<__half*>(output), 
        total_elements
    );
}