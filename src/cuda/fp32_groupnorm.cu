// =========================================================
// 文件名: groupnorm_scaled_h2rcp.cu
// 方案: 使用 h2rcp，但结合 Scaling Trick 绕过 FP16 范围限制
// =========================================================

#include <cstdio>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h> 

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = __fadd_rn(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// 保留原始实现
__device__ __forceinline__ float compute_rcp_constrained(float v) {
    __half2 v_h2 = __float2half2_rn(v);
    __half2 rcp_h2 = h2rcp(v_h2);
    return __low2float(rcp_h2);
}

__global__ void __launch_bounds__(BLOCK_SIZE) GroupNormKernelFP32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int N, int C, int HxW, int groups, float eps) {

    __shared__ float s_mem[2];
    if (threadIdx.x == 0) {
        s_mem[0] = 0.0f;
        s_mem[1] = 0.0f;
    }
    __syncthreads();

    int n = blockIdx.x; 
    int g = blockIdx.y; 

    int channels_per_group = C / groups;
    float f_num_elements = (float)(channels_per_group * HxW);

    int c_start = g * channels_per_group;
    int batch_offset = n * C * HxW;

    // --- Pass 1: Mean & Variance ---
    
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    for (int c = 0; c < channels_per_group; ++c) {
        const float* input_ptr = input + batch_offset + (c_start + c) * HxW;

        int idx = threadIdx.x * 4;
        int limit = (HxW / 4) * 4;

        while (idx < limit) {
            float4 v = *reinterpret_cast<const float4*>(&input_ptr[idx]);
            
            float sum_01 = __fadd_rn(v.x, v.y);
            float sum_23 = __fadd_rn(v.z, v.w);
            local_sum = __fadd_rn(local_sum, __fadd_rn(sum_01, sum_23));

            float sq_0 = __fmul_rn(v.x, v.x);
            float sq_1 = __fmul_rn(v.y, v.y);
            float sq_2 = __fmul_rn(v.z, v.z);
            float sq_3 = __fmul_rn(v.w, v.w);
            
            float sum_sq_01 = __fadd_rn(sq_0, sq_1);
            float sum_sq_23 = __fadd_rn(sq_2, sq_3);
            local_sum_sq = __fadd_rn(local_sum_sq, __fadd_rn(sum_sq_01, sum_sq_23));

            idx += blockDim.x * 4;
        }

        idx = limit + threadIdx.x;
        if (idx < HxW) {
            float val = input_ptr[idx];
            local_sum = __fadd_rn(local_sum, val);
            local_sum_sq = __fadd_rn(local_sum_sq, __fmul_rn(val, val));
        }
    }

    local_sum = warpReduceSum(local_sum);
    local_sum_sq = warpReduceSum(local_sum_sq);

    if ((threadIdx.x & 0x1f) == 0) {
        atomicAdd(&s_mem[0], local_sum);
        atomicAdd(&s_mem[1], local_sum_sq);
    }

    __syncthreads();

    // ---------------------------------------------------------
    // 【修改点】Scaling Trick + h2rcp
    // ---------------------------------------------------------
    
    // 1. 定义缩放系数 1/256 (2的幂次，无损)
    // 100万元素缩小后约为 3906，安全处于 FP16 范围内
    float scale_factor = 0.00390625f; 

    // 2. 先缩小 N
    float n_scaled = __fmul_rn(f_num_elements, scale_factor);

    // 3. 计算 1 / (N_scaled) = 1 / (N * scale) = 1/N / scale
    float rcp_n_scaled = compute_rcp_constrained(n_scaled);

    // 4. 还原结果: (1/N_scaled) * scale = 1/N
    float rcp_N = __fmul_rn(rcp_n_scaled, scale_factor);

    // ---------------------------------------------------------

    float mean = __fmul_rn(s_mem[0], rcp_N);
    float mean_sq = __fmul_rn(mean, mean);
    float avg_sum_sq = __fmul_rn(s_mem[1], rcp_N);
    float var = __fsub_rn(avg_sum_sq, mean_sq);
    float rstd = rsqrtf(__fadd_rn(fmaxf(var, 0.0f), eps));

    // --- Pass 2: Normalize ---

    for (int c = 0; c < channels_per_group; ++c) {
        int current_c = c_start + c;
        int offset = batch_offset + current_c * HxW;
        
        const float* input_ptr = input + offset;
        float* output_ptr = output + offset;

        float g_val = (gamma != nullptr) ? gamma[current_c] : 1.0f;
        float b_val = (beta != nullptr) ? beta[current_c] : 0.0f;

        float scale = __fmul_rn(rstd, g_val);
        float shift = __fsub_rn(b_val, __fmul_rn(mean, scale));

        int idx = threadIdx.x * 4;
        int limit = (HxW / 4) * 4;

        while (idx < limit) {
            float4 v = *reinterpret_cast<const float4*>(&input_ptr[idx]);
            float4 out_v;

            out_v.x = __fadd_rn(__fmul_rn(v.x, scale), shift);
            out_v.y = __fadd_rn(__fmul_rn(v.y, scale), shift);
            out_v.z = __fadd_rn(__fmul_rn(v.z, scale), shift);
            out_v.w = __fadd_rn(__fmul_rn(v.w, scale), shift);

            *reinterpret_cast<float4*>(&output_ptr[idx]) = out_v;
            idx += blockDim.x * 4;
        }

        idx = limit + threadIdx.x;
        if (idx < HxW) {
            float val = input_ptr[idx];
            output_ptr[idx] = __fadd_rn(__fmul_rn(val, scale), shift);
        }
    }
}

void launch_groupnorm_fp32(float* output, const float* input, const float* weight, const float* bias, int N, int C, int HxW, int groups, float eps) {
    dim3 grid(N, groups);
    dim3 block(BLOCK_SIZE);
    GroupNormKernelFP32<<<grid, block>>>(output, input, weight, bias, N, C, HxW, groups, eps);
}
//[N=64, C=512, 128x128]:5.068 ms