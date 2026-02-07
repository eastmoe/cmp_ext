#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h> 

#define ERF_P  0.3275911f
#define ERF_A1 0.254829592f
#define ERF_A2 -0.284496736f
#define ERF_A3 1.421413741f
#define ERF_A4 -1.453152027f
#define ERF_A5 1.061405429f

// 约束: 倒数必须走 FP16 hrcp
__device__ __forceinline__ float custom_rcp_bf16_via_fp16(float x) {
    __half h = __float2half(x);
    __half r = hrcp(h);
    return __half2float(r);
}

// 核心逻辑：在 FP32 域计算，但严禁 FMA
__device__ __forceinline__ float compute_erf_logic_bf16(float x) {
    float sign = (x >= 0.0f) ? 1.0f : -1.0f;
    float abs_x = fabsf(x);

    // denom = 1 + p*|x|
    // 禁止 FMA: 使用 __fmul_rn + __fadd_rn
    float denom = __fadd_rn(1.0f, __fmul_rn(ERF_P, abs_x));
    
    // 倒数转 FP16
    float t = custom_rcp_bf16_via_fp16(denom);

    // Poly (Explicit mul/add)
    float p = __fadd_rn(__fmul_rn(ERF_A5, t), ERF_A4);
    p = __fadd_rn(__fmul_rn(p, t), ERF_A3);
    p = __fadd_rn(__fmul_rn(p, t), ERF_A2);
    p = __fadd_rn(__fmul_rn(p, t), ERF_A1);
    p = __fmul_rn(p, t);

    // Exp
    float x2 = __fmul_rn(x, x);
    float neg_x2 = __fsub_rn(0.0f, x2);
    float e = __expf(neg_x2); // FP32 exp

    // Result
    float term = __fmul_rn(p, e);
    float res = __fsub_rn(1.0f, term);
    
    return __fmul_rn(res, sign);
}

__global__ void erf_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Load BF16 -> Float
        float x = __bfloat162float(input[idx]);
        // Compute (Strict rules)
        float res = compute_erf_logic_bf16(x);
        // Store Float -> BF16
        output[idx] = __float2bfloat16(res);
    }
}

void launch_erf_bf16(const void* input, void* output, int total_elements) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    erf_kernel_bf16<<<blocks, threads>>>(
        (const __nv_bfloat16*)input, 
        (__nv_bfloat16*)output, 
        total_elements
    );
}