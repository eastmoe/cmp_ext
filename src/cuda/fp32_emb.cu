#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>

// ------------------------------------------------------------------
// 辅助工具：强制内联的向量读写
// ------------------------------------------------------------------

// 强制内联的向量加载
__device__ __forceinline__ float4 load_float4(const float* ptr, int offset) {
    return reinterpret_cast<const float4*>(ptr)[offset];
}

// 强制内联的向量存储
__device__ __forceinline__ void store_float4(float* ptr, int offset, float4 v) {
    reinterpret_cast<float4*>(ptr)[offset] = v;
}

// ------------------------------------------------------------------
// Kernel: 极致优化的 Embedding Lookup (ILP + Vectorized)
// ------------------------------------------------------------------
// 模板参数 UNROLL_FACTOR: 每个线程每轮循环处理多少个 float4 (推荐 4)
template <int UNROLL_FACTOR>
__global__ void __launch_bounds__(256) embedding_fp32_ilp_kernel(
    const int64_t* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int num_indices,
    const int embedding_dim,
    const int padding_idx,
    const int num_embeddings) 
{
    // -----------------------------------------------------------
    // 1. 索引计算与 Warp 分配
    // -----------------------------------------------------------
    const int warp_size = 32;
    int lane_id = threadIdx.x % warp_size;
    int warp_id = threadIdx.x / warp_size;
    int warps_per_block = blockDim.x / warp_size;
    int global_warp_id = blockIdx.x * warps_per_block + warp_id;
    int total_warps = gridDim.x * warps_per_block;

    // 预计算 float4 的数量
    int dim_vec = embedding_dim >> 2; // embedding_dim / 4
    int dim_tail = embedding_dim & 3; // embedding_dim % 4

    // -----------------------------------------------------------
    // 2. Grid-Stride Loop (处理每一行)
    // -----------------------------------------------------------
    for (int i = global_warp_id; i < num_indices; i += total_warps) {
        
        // --- 读取 Index (Lane 0 读取并广播) ---
        int64_t target_idx;
        if (lane_id == 0) {
            target_idx = indices[i];
        }
        // 广播给 Warp 内所有线程
        target_idx = __shfl_sync(0xffffffff, (long long)target_idx, 0);

        float* out_row = output + i * embedding_dim;

        // --- 路径分支：Padding/越界 vs 正常读取 ---
        if (target_idx == padding_idx || target_idx < 0 || target_idx >= num_embeddings) {
            // ============ 快速置零路径 ============
            #pragma unroll
            for (int k = lane_id; k < dim_vec; k += warp_size) {
                store_float4(out_row, k, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
            }
            // 处理尾部
            if (dim_tail > 0) {
                int tail_start = dim_vec * 4;
                for (int k = tail_start + lane_id; k < embedding_dim; k += warp_size) {
                    out_row[k] = 0.0f;
                }
            }
        } 
        else {
            // ============ 极致带宽读取路径 (ILP) ============
            const float* w_row = weight + target_idx * embedding_dim;
            
            int k = lane_id;
            
            // 每次 Warp 步进 stride = warp_size * UNROLL_FACTOR
            const int stride = warp_size * UNROLL_FACTOR;
            
            // 主体部分：多路循环展开，隐藏内存延迟
            for (; k + (stride - warp_size) < dim_vec; k += stride) {
                // 1. 批量发射 Load 请求 (放入寄存器)
                float4 v[UNROLL_FACTOR]; 
                
                #pragma unroll
                for (int u = 0; u < UNROLL_FACTOR; u++) {
                    v[u] = load_float4(w_row, k + u * warp_size);
                }

                // 2. 批量执行 Store
                #pragma unroll
                for (int u = 0; u < UNROLL_FACTOR; u++) {
                    store_float4(out_row, k + u * warp_size, v[u]);
                }
            }

            // 处理剩余的 float4 (不够一次展开的部分)
            for (; k < dim_vec; k += warp_size) {
                float4 v = load_float4(w_row, k);
                store_float4(out_row, k, v);
            }

            // 处理非 4 倍数的尾部标量 (Scalar fallback)
            if (dim_tail > 0) {
                int tail_start = dim_vec * 4;
                for (int t = tail_start + lane_id; t < embedding_dim; t += warp_size) {
                    out_row[t] = w_row[t];
                }
            }
        }
    }
}

// ------------------------------------------------------------------
// Host Launcher (入口函数，保持原有签名)
// ------------------------------------------------------------------
void launch_embedding_fp32(
    const int64_t* indices, 
    const float* weight, 
    float* output, 
    int num_indices, 
    int embedding_dim, 
    int padding_idx, 
    int num_embeddings) 
{
    // A100 优化参数
    const int threads_per_block = 256; 
    const int warps_per_block = threads_per_block / 32;
    
    // 计算 Block 数量
    // 保持 Grid 足够大以利用 A100 的 108 个 SM，但限制上限以防超大输入
    int min_blocks = (num_indices + warps_per_block - 1) / warps_per_block;
    // 限制最大 block 数为 65535 (足以跑满 GPU 且不会造成调度过载)
    int blocks = std::min(65535, min_blocks); 
    
    // 实例化模板 Kernel，UNROLL_FACTOR=4 (平衡寄存器使用和带宽)
    embedding_fp32_ilp_kernel<4><<<blocks, threads_per_block>>>(
        indices, weight, output, num_indices, embedding_dim, padding_idx, num_embeddings
    );
}
//[Vocab=32000, Dim=1024, Input=(512, 1024)]：3.265 ms, Avg Power: 189.10 W