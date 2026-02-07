#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>

__global__ void embedding_bf16_kernel(
    const int64_t* indices,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int num_indices,
    int embedding_dim,
    int padding_idx,
    int num_embeddings) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_indices * embedding_dim;

    if (idx < total_elements) {
        int row_idx = idx / embedding_dim;
        int col_idx = idx % embedding_dim;

        int64_t target_idx = indices[row_idx];

        if (target_idx == padding_idx) {
            output[idx] = __float2bfloat16(0.0f);
        } else if (target_idx >= 0 && target_idx < num_embeddings) {
            output[idx] = weight[target_idx * embedding_dim + col_idx];
        } else {
            output[idx] = __float2bfloat16(0.0f);
        }
    }
}

void launch_embedding_bf16(const int64_t* indices, const void* weight, void* output, int num_indices, int embedding_dim, int padding_idx, int num_embeddings) {
    int total_elements = num_indices * embedding_dim;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    embedding_bf16_kernel<<<blocks, threads>>>(
        indices, 
        reinterpret_cast<const __nv_bfloat16*>(weight), 
        reinterpret_cast<__nv_bfloat16*>(output), 
        num_indices, embedding_dim, padding_idx, num_embeddings
    );
}