#include <cstdio>
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// =========================================================================
// A100 CUDA CORE 极限优化配置 (Native BF16 SIMT)
// =========================================================================
// 块大小配置：每个Block计算 128x128 的 C
#define BM 128
#define BN 128
#define BK 32   

// 线程分块配置：每个线程计算 8x8 的元素
#define TM 8
#define TN 8

// Padding 修正为 8，确保每一行的字节数 (BK+PAD)*2 是 16 的倍数 (128-bit对齐)
// (32 + 8) * 2 = 80 bytes (aligned to 16)
#define PAD 8   

//using namespace nvcuda;

// =========================================================================
// 辅助函数：安全的向量化加载
// =========================================================================
__device__ __forceinline__ void load_gmem_vectorized(const __nv_bfloat16* ptr, int4& dest, bool pred) {
    if (pred) {
        // 检查地址是否 16 字节对齐
        if ((reinterpret_cast<uintptr_t>(ptr) % 16) == 0) {
            dest = *reinterpret_cast<const int4*>(ptr);
        } else {
            // 不对齐时的回退路径：逐元素加载
            __nv_bfloat16* dst_ptr = reinterpret_cast<__nv_bfloat16*>(&dest);
            #pragma unroll
            for(int i=0; i<8; ++i) dst_ptr[i] = ptr[i];
        }
    } else {
        // 越界填充 0
        __nv_bfloat16* dst_ptr = reinterpret_cast<__nv_bfloat16*>(&dest);
        #pragma unroll
        for(int i=0; i<8; ++i) dst_ptr[i] = __float2bfloat16(0.0f);
    }
}

// =========================================================================
// 核心 Kernel (CUDA Core Native BF16)
// =========================================================================

__global__ void __launch_bounds__(256) gemm_bf16_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc) 
{
    // Shared Memory 配置 (Double Buffering)
    // 增加 alignas 确保基地址对齐
    // Padding=8 保证每行 Stride 对齐到 16 bytes
    __shared__ alignas(16) __nv_bfloat16 smem_a[2][BM][BK + PAD];
    __shared__ alignas(16) __nv_bfloat16 smem_b[2][BK][BN + PAD];

    // 寄存器分配
    __nv_bfloat162 c_reg[TM][TN / 2];
    __nv_bfloat16  a_frag[TM];      
    __nv_bfloat162 b_frag[TN / 2];  

    int4 ldg_a_reg[2]; 
    int4 ldg_b_reg[2];

    // 初始化 C 寄存器
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN / 2; ++j) {
            c_reg[i][j] = __float2bfloat162_rn(0.0f);
        }
    }

    int tid = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int ty = tid / 16; 
    int tx = tid % 16;

    const __nv_bfloat16* A_ptr = A; 
    const __nv_bfloat16* B_ptr = B; 

    // 主循环
    int num_tiles = (K + BK - 1) / BK;
    int write_stage = 0;
    int compute_stage = 0;

    // Prologue: 预加载
    {
        // Load A
        #pragma unroll
        for(int i=0; i<2; ++i) {
            int r = (tid / 4) + i * 64; 
            int c = (tid % 4) * 8;
            int g_r = by * BM + r;
            int g_c = c; // k=0
            
            bool pred = (r < 128 && c < 32 && g_r < M && g_c < K);
            // 使用安全加载函数
            load_gmem_vectorized(A_ptr + g_r * lda + g_c, ldg_a_reg[i], pred);
        }

        // Load B
        #pragma unroll
        for(int i=0; i<2; ++i) {
            int r = (tid / 16) + i * 16;
            int c = (tid % 16) * 8;
            int g_r = r; // k=0
            int g_c = bx * BN + c;
            
            bool pred = (r < 32 && c < 128 && g_r < K && g_c < N);
            load_gmem_vectorized(B_ptr + g_r * ldb + g_c, ldg_b_reg[i], pred);
        }
    }

    for (int k = 0; k < num_tiles; ++k) {
        // 1. Store Global-loaded Registers to Shared Memory
        // SMEM 指针是对齐的 (由 alignas 和 PAD=8 保证)，直接强转 int4
        #pragma unroll
        for(int i=0; i<2; ++i) {
            int r = (tid / 4) + i * 64;
            int c = (tid % 4) * 8;
            *reinterpret_cast<int4*>(&smem_a[write_stage][r][c]) = ldg_a_reg[i];
        }
        #pragma unroll
        for(int i=0; i<2; ++i) {
            int r = (tid / 16) + i * 16;
            int c = (tid % 16) * 8;
            *reinterpret_cast<int4*>(&smem_b[write_stage][r][c]) = ldg_b_reg[i];
        }

        __syncthreads();

        // 2. Start Loading Next Tile
        if (k < num_tiles - 1) {
            int next_k = (k + 1) * BK;
            
             // Load A
            #pragma unroll
            for(int i=0; i<2; ++i) {
                int r = (tid / 4) + i * 64;
                int c = (tid % 4) * 8;
                int g_r = by * BM + r;
                int g_c = next_k + c;
                
                bool pred = (r < 128 && c < 32 && g_r < M && g_c < K);
                load_gmem_vectorized(A_ptr + g_r * lda + g_c, ldg_a_reg[i], pred);
            }
            // Load B
            #pragma unroll
            for(int i=0; i<2; ++i) {
                int r = (tid / 16) + i * 16;
                int c = (tid % 16) * 8;
                int g_r = next_k + r;
                int g_c = bx * BN + c;
                
                bool pred = (r < 32 && c < 128 && g_r < K && g_c < N);
                load_gmem_vectorized(B_ptr + g_r * ldb + g_c, ldg_b_reg[i], pred);
            }
        }

        // 3. Compute
        #pragma unroll
        for (int k_step = 0; k_step < BK; ++k_step) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                // SMEM load: scalar is fine for A row-major
                a_frag[i] = smem_a[compute_stage][ty * TM + i][k_step];
            }

            // SMEM load B: packed int4. smem_b layout guarantees alignment
            int4 b_vec = *reinterpret_cast<int4*>(&smem_b[compute_stage][k_step][tx * TN]);
            __nv_bfloat16* b_ptr = reinterpret_cast<__nv_bfloat16*>(&b_vec);

            #pragma unroll
            for(int j=0; j < TN/2; ++j) {
                __nv_bfloat162 b_packed;
                b_packed.x = b_ptr[j*2];
                b_packed.y = b_ptr[j*2+1];
                b_frag[j] = b_packed;
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                __nv_bfloat162 a_val = __bfloat162bfloat162(a_frag[i]);
                #pragma unroll
                for (int j = 0; j < TN / 2; ++j) {
                    c_reg[i][j] = __hfma2(a_val, b_frag[j], c_reg[i][j]);
                }
            }
        }

        write_stage ^= 1;
        compute_stage ^= 1;
        __syncthreads();
    }

    // Store Result
    int global_row_start = by * BM + ty * TM;
    int global_col_start = bx * BN + tx * TN;

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int g_r = global_row_start + i;
        if (g_r < M) {
            #pragma unroll
            for (int j = 0; j < TN / 2; ++j) {
                int g_c = global_col_start + j * 2;
                if (g_c < N) {
                    C[g_r * ldc + g_c] = c_reg[i][j].x;
                    if (g_c + 1 < N) {
                        C[g_r * ldc + g_c + 1] = c_reg[i][j].y;
                    }
                }
            }
        }
    }
}

// =========================================================================
// Matmul Launcher
// =========================================================================

void launch_matmul_bf16(
    const void* input_ptr, 
    const void* weight_ptr, 
    void* output_ptr, 
    int m, int n, int k) {

    const __nv_bfloat16* input = reinterpret_cast<const __nv_bfloat16*>(input_ptr);
    const __nv_bfloat16* weight = reinterpret_cast<const __nv_bfloat16*>(weight_ptr);
    __nv_bfloat16* output = reinterpret_cast<__nv_bfloat16*>(output_ptr);

    int lda = k;
    int ldb = n;
    int ldc = n;

    dim3 block_size(256); 
    dim3 grid_size((n + BM - 1) / BM, (m + BM - 1) / BM);
    
    cudaFuncSetAttribute(gemm_bf16_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    gemm_bf16_kernel<<<grid_size, block_size>>>(
        input, weight, output, m, n, k, lda, ldb, ldc
    );
}

// =========================================================================
// Bias Kernel & Launcher (修复对齐问题)
// =========================================================================

__global__ void add_bias_bf16_optimized(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ bias,
    int n, int total_elements) 
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8; 
    
    // 向量化路径前提：索引在范围内 且 输出地址对齐
    if (idx + 8 <= total_elements && (reinterpret_cast<uintptr_t>(output + idx) % 16 == 0)) {
        int4* out_vec_ptr = reinterpret_cast<int4*>(output + idx);
        int4 out_data = *out_vec_ptr;
        __nv_bfloat16* out_bf16 = reinterpret_cast<__nv_bfloat16*>(&out_data);

        #pragma unroll
        for(int i=0; i<8; i+=2) {
            int curr_idx = idx + i;
            int col = curr_idx % n;
            
            __nv_bfloat162 out_val;
            out_val.x = out_bf16[i];
            out_val.y = out_bf16[i+1];

            __nv_bfloat162 bias_val;
            bias_val.x = bias[col];
            // 处理 bias 环绕
            bias_val.y = bias[(col + 1) % n]; 

            out_val = __hadd2(out_val, bias_val);

            out_bf16[i] = out_val.x;
            out_bf16[i+1] = out_val.y;
        }
        *out_vec_ptr = out_data;
    } 
    else if (idx < total_elements) {
        // 标量回退路径 (处理尾部或不对齐部分)
        for(int i=0; i<8; ++i) {
            int curr_idx = idx + i;
            if (curr_idx < total_elements) {
                int col = curr_idx % n;
                __nv_bfloat16 val = output[curr_idx];
                __nv_bfloat16 b = bias[col];
                output[curr_idx] = __hadd(val, b);
            }
        }
    }
}

void launch_add_bias_bf16(void* output_ptr, const void* bias_ptr, int rows, int cols) {
    __nv_bfloat16* output = reinterpret_cast<__nv_bfloat16*>(output_ptr);
    const __nv_bfloat16* bias = reinterpret_cast<const __nv_bfloat16*>(bias_ptr);
    
    int total_elements = rows * cols;
    int threads = 256;
    int blocks = (total_elements + (threads * 8) - 1) / (threads * 8);
    
    add_bias_bf16_optimized<<<blocks, threads>>>(output, bias, cols, total_elements);
}
//[4096x4096] @ [4096x4096].T： 6.723 ms, Avg Power: 150.14 W  3.6 x
