#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h> 

// ----------------------------------------------------------------------
// BF16 Tanh Kernel
// 约束：
// 1. 不要使用 BF16 原生算术指令 (转为 FP32)
// 2. 倒数必须基于 FP16 计算 (BF16 -> FP32 -> FP16 -> RCP -> FP32 -> Result -> BF16)
// ----------------------------------------------------------------------

__global__ void tanh_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        __nv_bfloat16 bf_in = input[idx];
        
        // 转换到 FP32
        float x = __bfloat162float(bf_in);

        // 1. 2x (FP32)
        float val_2x = __fmul_rn(x, 2.0f);

        // 2. exp (FP32)
        float exp_val = __expf(val_2x);

        // 3. 分子 (FP32)
        float numerator = __fadd_rn(exp_val, -1.0f);

        // 4. 分母 (FP32)
        float denominator = __fadd_rn(exp_val, 1.0f);

        // 5. 倒数 (Constraint 7 & Constraint 2)
        // 必须基于 FP16 计算倒数
        // BF16 本身没有硬件 rcp 指令等价物，通常转 float。
        // 但根据要求，必须转为 FP16 使用 hrcp。
        // float -> half
        __half h_den = __float2half(denominator);
        // half rcp
        __half h_rcp = hrcp(h_den);
        // half -> float
        float rcp_den = __half2float(h_rcp);

        // 6. 结果 (FP32)
        float res_f = __fmul_rn(numerator, rcp_den);

        // 转换回 BF16 输出
        output[idx] = __float2bfloat16(res_f);
    }
}

void launch_tanh_bf16(const void* input, void* output, int total_elements) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    // 修正错误：增加了 <<<blocks, threads>>>
    tanh_kernel_bf16<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(input), 
        reinterpret_cast<__nv_bfloat16*>(output), 
        total_elements
    );
}