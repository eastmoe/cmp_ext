#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cassert>

// 定义向量化类型，int4 用于搬运 128位 (16字节) 数据，即 8 个 __half
using vec_t = int4;
constexpr int VEC_SIZE = sizeof(vec_t) / sizeof(__half); // 8

/**
 * 优化策略：
 * 1. 向量化：使用 int4 (128-bit) 一次搬运 8 个 half。
 * 2. 避免除法：每个 Block 处理若干行，ThreadIdx.x 直接映射列，消除 idx / dim 计算。
 * 3. 访存合并：确保同一个 Warp 内的线程读取连续内存。
 * 4. 分支外提：Padding/越界判断在行级别处理，而非元素级别。
 */
__global__ void __launch_bounds__(256) embedding_fp16_opt_kernel(
    const int64_t* __restrict__ indices,      // 只读，使用 __restrict__ 提示编译器
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    int num_indices,
    int embedding_dim,
    int padding_idx,
    int num_embeddings,
    int num_vecs_per_row) // 预计算每行有多少个 int4 向量
{
    // -----------------------------------------------------------------
    // 1. Grid-Stride Loop over Rows (Indices)
    //    blockIdx.x 负责行维度的遍历，使得一个 Block 完整处理某一行(或多行)
    // -----------------------------------------------------------------
    for (int row_idx = blockIdx.x; row_idx < num_indices; row_idx += gridDim.x) {
        
        // 提前读取 index，减少全局内存延迟影响
        int64_t target_idx = indices[row_idx];
        
        // 计算输出的起始偏移量
        int64_t out_row_offset = (int64_t)row_idx * embedding_dim;
        
        // -------------------------------------------------------------
        // 2. 逻辑分支优化：将 Padding 判断移出内层拷贝循环
        // -------------------------------------------------------------
        bool is_valid = (target_idx != padding_idx) && (target_idx >= 0) && (target_idx < num_embeddings);
        
        // 针对 A100 优化：利用 reinterpret_cast 进行向量化指针操作
        vec_t* out_vec_ptr = reinterpret_cast<vec_t*>(output + out_row_offset);
        
        if (is_valid) {
            // == 有效 Index：执行拷贝 ==
            int64_t weight_row_offset = target_idx * embedding_dim;
            const vec_t* weight_vec_ptr = reinterpret_cast<const vec_t*>(weight + weight_row_offset);

            // ---------------------------------------------------------
            // 3. 向量化拷贝循环 (Vectorized Loop)
            //    线程并行处理这一行中的不同 int4 块
            // ---------------------------------------------------------
            for (int i = threadIdx.x; i < num_vecs_per_row; i += blockDim.x) {
                // 128-bit Load & Store
                out_vec_ptr[i] = weight_vec_ptr[i]; 
            }

            // 处理无法被 8 整除的尾部元素 (Scalar Peeling)
            // 绝大多数情况 embedding_dim 是 8 的倍数，此部分不执行
            int remaining_start = num_vecs_per_row * VEC_SIZE;
            for (int i = remaining_start + threadIdx.x; i < embedding_dim; i += blockDim.x) {
                output[out_row_offset + i] = weight[weight_row_offset + i];
            }

        } else {
            // == 无效 Index (Padding/OOB)：执行清零 ==
            
            // 构造全 0 的 128-bit 寄存器
            vec_t zero_vec;
            zero_vec.x = 0; zero_vec.y = 0; zero_vec.z = 0; zero_vec.w = 0;

            for (int i = threadIdx.x; i < num_vecs_per_row; i += blockDim.x) {
                out_vec_ptr[i] = zero_vec;
            }

            // 尾部清零
            int remaining_start = num_vecs_per_row * VEC_SIZE;
            for (int i = remaining_start + threadIdx.x; i < embedding_dim; i += blockDim.x) {
                output[out_row_offset + i] = __float2half(0.0f);
            }
        }
    }
}

void launch_embedding_fp16(const int64_t* indices, const void* weight, void* output, int num_indices, int embedding_dim, int padding_idx, int num_embeddings) {
    // 硬件相关配置
    const int thread_per_block = 256; 
    
    // 计算向量化参数
    // 我们假设 output 和 weight 内存地址是 16字节对齐的 (cudaMalloc 默认 256字节对齐)
    // 如果 embedding_dim 不是 8 的倍数，向量化部分只能处理前面部分，尾部走 scalar 循环
    int num_vecs_per_row = embedding_dim / 8;

    // Grid 配置策略：
    // 我们希望有足够的 Block 填满 GPU 的 SM。
    // A100 有 108 个 SM。每个 SM 可以驻留多个 Block。
    // 简单的启发式：行数较多时，使用较大的 Grid；行数较少时，Grid = num_indices。
    // 为了防止 Grid 过大，设置上限。
    int min_blocks = (num_indices + 3) / 4; // 至少保证一定的并行度
    int max_blocks = 108 * 8; // A100 SM count * ~saturation
    int blocks = std::min(max_blocks, std::max(1, num_indices));
    
    // 如果 embedding_dim 非常大（例如 > 4096），可以考虑增加每个 Row 的 Block 数，
    // 但通常 embedding lookup 是 Memory Bound，一个 Block 处理一行足以跑满带宽。

    embedding_fp16_opt_kernel<<<blocks, thread_per_block>>>(
        indices, 
        reinterpret_cast<const __half*>(weight), 
        reinterpret_cast<__half*>(output), 
        num_indices, embedding_dim, padding_idx, num_embeddings,
        num_vecs_per_row
    );
    
    // 检查 kernel 启动错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}