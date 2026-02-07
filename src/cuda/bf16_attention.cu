#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>


// 复用 FP16 里的 softmax 逻辑 (因为 logits 都是转成 float 算的)
__device__ void softmax_block_bf16(float* s_data, int len) {
    int tid = threadIdx.x;
    
    __shared__ float s_max;
    if (tid == 0) s_max = -1e20f;
    __syncthreads();
    
    if (tid == 0) {
        float local_max = -1e20f;
        for(int i=0; i<len; ++i) local_max = fmaxf(local_max, s_data[i]);
        s_max = local_max;
    }
    __syncthreads();

    __shared__ float s_sum;
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();

    float max_v = s_max;
    if (tid == 0) {
        float local_sum = 0.0f;
        for (int i = 0; i < len; ++i) {
            float val = expf(s_data[i] - max_v);
            s_data[i] = val;
            local_sum += val;
        }
        s_sum = local_sum;
    }
    __syncthreads();

    float inv_sum = 1.0f / (s_sum + 1e-6f);
    for (int i = tid; i < len; i += blockDim.x) {
        s_data[i] *= inv_sum;
    }
}

__global__ void attention_bf16_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ Output,
    int S, int D, float scale) 
{
    int q_idx = blockIdx.x; 
    int bh_idx = blockIdx.y; 
    int tid = threadIdx.x;
    
    int batch_head_offset = bh_idx * S * D;
    const __nv_bfloat16* q_ptr = Q + batch_head_offset + q_idx * D;
    const __nv_bfloat16* k_base = K + batch_head_offset;
    const __nv_bfloat16* v_base = V + batch_head_offset;
    __nv_bfloat16* out_ptr = Output + batch_head_offset + q_idx * D;

    extern __shared__ float s_logits[]; 

    // 1. Q * K^T
    for (int i = 0; i < S; ++i) {
        float dot = 0.0f;
        const __nv_bfloat16* k_ptr = k_base + i * D;
        
        for (int d = tid; d < D; d += blockDim.x) {
            dot += __bfloat162float(q_ptr[d]) * __bfloat162float(k_ptr[d]);
        }
        
        __shared__ float row_dot;
        if (tid == 0) row_dot = 0.0f;
        __syncthreads();
        
        atomicAdd(&row_dot, dot);
        __syncthreads();
        
        if (tid == 0) {
            s_logits[i] = row_dot * scale;
        }
        __syncthreads();
    }

    // 2. Softmax
    __syncthreads();
    softmax_block_bf16(s_logits, S);
    __syncthreads();

    // 3. Prob * V
    for (int d = tid; d < D; d += blockDim.x) {
        float val = 0.0f;
        for (int i = 0; i < S; ++i) {
            val += s_logits[i] * __bfloat162float(v_base[i * D + d]);
        }
        out_ptr[d] = __float2bfloat16(val);
    }
}

void launch_attention_bf16(const void* q, const void* k, const void* v, void* output, int B, int H, int S, int D, float scale) {
    dim3 grid(S, B * H);
    dim3 block(128);
    int shared_mem_size = S * sizeof(float);
    attention_bf16_kernel<<<grid, block, shared_mem_size>>>(
        (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v, (__nv_bfloat16*)output, S, D, scale);
}

