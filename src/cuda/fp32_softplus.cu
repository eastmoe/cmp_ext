#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// 辅助函数：严格遵守 Constraint 7
// 目标：计算 1/beta
// 要求：不要使用 FP32 的 __frcp_rn，必须转换为向量化的 FP16 使用 h2rcp
__device__ __forceinline__ float get_strict_inv_beta(float beta) {
    // 1. 构造向量 (Vectorize)
    float2 beta_vec = make_float2(beta, beta);
    
    // 2. 转换为 FP16 向量 (half2)
    half2 h_beta_vec = __float22half2_rn(beta_vec);
    
    // 3. 使用硬件 FP16 向量倒数指令
    half2 h_rcp_vec = h2rcp(h_beta_vec);
    
    // 4. 转回 FP32 向量
    float2 rcp_vec = __half22float2(h_rcp_vec);
    
    // 5. 返回标量结果 (取其中一个分量即可)
    return rcp_vec.x;
}

// 标量 Softplus 核心：用于处理尾部元素
// 严格遵守 Constraint 1, 4, 8
__device__ __forceinline__ float softplus_scalar_strict(float x, float beta, float inv_beta, float threshold) {
    // Constraint 1: 禁止 FP32 FMA，使用 __fmul_rn
    float bx = __fmul_rn(x, beta);

    if (bx > threshold) {
        return x;
    } else {
        // Constraint 4: 禁止 FP16 hexp，必须使用 FP32 __expf
        float e = __expf(bx);
        
        // Constraint 1: 禁止 FP32 FMA，使用 __fadd_rn
        float term = __fadd_rn(1.0f, e);
        
        // Constraint 8: 禁止 FP32 log，必须转换为向量化的 FP16 使用 h2log
        // 为了满足 "向量化" 要求，即使是标量也复制一份进行 SIMD 计算
        float2 term_vec = make_float2(term, term);
        half2 h_term = __float22half2_rn(term_vec);
        half2 h_log = h2log(h_term); // 使用 vector FP16 log
        float2 log_vec = __half22float2(h_log);
        float log_val = log_vec.x;
        
        // Constraint 1: 禁止 FP32 FMA
        return __fmul_rn(log_val, inv_beta);
    }
}

// 向量化 Softplus 核心 (针对 float2)
// 适配 float4 数据流 (处理一半)，最大化利用 h2log
__device__ __forceinline__ float2 softplus_v2_strict(float2 val, float beta, float inv_beta, float threshold) {
    float2 out;

    // Constraint 1: 独立乘法
    float bx_x = __fmul_rn(val.x, beta);
    float bx_y = __fmul_rn(val.y, beta);

    // Constraint 4: 必须使用 FP32 exp
    float e_x = __expf(bx_x);
    float e_y = __expf(bx_y);

    // Constraint 1: 独立加法
    float term_x = __fadd_rn(1.0f, e_x);
    float term_y = __fadd_rn(1.0f, e_y);

    // Constraint 8: 必须使用向量化 FP16 h2log
    // 将两个 float 打包为 half2
    half2 h_term = __float22half2_rn(make_float2(term_x, term_y));
    
    // 执行 FP16 向量对数
    half2 h_log = h2log(h_term);
    
    // 转回 float2
    float2 log_vec = __half22float2(h_log);

    // 计算结果 (FP32)
    float res_x = __fmul_rn(log_vec.x, inv_beta);
    float res_y = __fmul_rn(log_vec.y, inv_beta);

    // 阈值判断与选择
    // 注意：如果 bx > threshold，exp(bx) 可能溢出或很大，
    // 但根据 Softplus 定义，超过阈值直接返回 x 即可，无需理会中间计算的 Inf/NaN (非 trap 模式下)
    out.x = (bx_x > threshold) ? val.x : res_x;
    out.y = (bx_y > threshold) ? val.y : res_y;

    return out;
}

__global__ void softplus_kernel_fp32_opt(const float* __restrict__ input, float* __restrict__ output, int total_elements, float beta, float threshold) {
    // 预计算倒数，满足 Constraint 7
    const float inv_beta = get_strict_inv_beta(beta);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // GA100 优化：使用 float4 进行向量化加载和存储
    int vec_limit = total_elements / 4;
    
    // 强转指针用于 float4 访问
    const float4* inp_vec = reinterpret_cast<const float4*>(input);
    float4* out_vec = reinterpret_cast<float4*>(output);

    // 1. 向量化循环
    for (int i = idx; i < vec_limit; i += stride) {
        float4 in_val = inp_vec[i];
        float4 out_val;

        // 将 float4 拆分为两个 float2 组，以便利用 h2log 处理 Constraint 8
        float2 v1 = make_float2(in_val.x, in_val.y);
        float2 v2 = make_float2(in_val.z, in_val.w);

        // 调用向量化核心函数
        float2 r1 = softplus_v2_strict(v1, beta, inv_beta, threshold);
        float2 r2 = softplus_v2_strict(v2, beta, inv_beta, threshold);

        // 组装回 float4
        out_val.x = r1.x;
        out_val.y = r1.y;
        out_val.z = r2.x;
        out_val.w = r2.y;

        out_vec[i] = out_val;
    }

    // 2. 标量尾部循环 (处理不能被4整除的剩余部分)
    int scalar_start = vec_limit * 4;
    for (int i = scalar_start + idx; i < total_elements; i += stride) {
        output[i] = softplus_scalar_strict(input[i], beta, inv_beta, threshold);
    }
}

void launch_softplus_fp32(const float* input, float* output, int total_elements, float beta, float threshold) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    softplus_kernel_fp32_opt<<<blocks, threads>>>(input, output, total_elements, beta, threshold);
}