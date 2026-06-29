#include <cstdio>
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

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
__device__ __forceinline__ void load_gmem_vectorized(const __nv_bfloat16* ptr, int4& dest, bool row_valid, int valid_count) {
    __nv_bfloat16* dst_ptr = reinterpret_cast<__nv_bfloat16*>(&dest);

    if (row_valid && valid_count >= 8) {
        // 检查地址是否 16 字节对齐
        if ((reinterpret_cast<uintptr_t>(ptr) % 16) == 0) {
            dest = *reinterpret_cast<const int4*>(ptr);
        } else {
            // 不对齐时的回退路径：逐元素加载
            #pragma unroll
            for(int i=0; i<8; ++i) dst_ptr[i] = ptr[i];
        }
    } else {
        // 越界或尾块填充 0，避免 128-bit 读取跨过有效行尾。
        #pragma unroll
        for(int i=0; i<8; ++i) {
            dst_ptr[i] = (row_valid && i < valid_count) ? ptr[i] : __float2bfloat16(0.0f);
        }
    }
}

__device__ __forceinline__ half2 bf16x2_to_half2(__nv_bfloat162 value) {
    return __float22half2_rn(__bfloat1622float2(value));
}

__device__ __forceinline__ half2 bf16_to_half2(__nv_bfloat16 value) {
    return __float2half2_rn(__bfloat162float(value));
}

__device__ __forceinline__ __nv_bfloat162 half2_to_bf16x2(half2 value) {
    return __float22bfloat162_rn(__half22float2(value));
}

__device__ __forceinline__ uint32_t get_smem_offset_bf16(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint32_t ptx_ld_shared_u16_bf16(const void* ptr) {
    uint16_t out;
    uint32_t smem_int_ptr = get_smem_offset_bf16(ptr);
    asm volatile("ld.shared.u16 %0, [%1];\n\t" : "=h"(out) : "r"(smem_int_ptr));
    return static_cast<uint32_t>(out);
}

__device__ __forceinline__ int4 ptx_ld_shared_v4_b32_bf16(const void* ptr) {
    int4 out;
    uint32_t smem_int_ptr = get_smem_offset_bf16(ptr);
    asm volatile(
        "ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n\t"
        : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
        : "r"(smem_int_ptr)
    );
    return out;
}

__device__ __forceinline__ uint32_t ptx_dup_low_bf16(uint32_t value) {
    uint32_t out;
    asm volatile("prmt.b32 %0, %1, %1, 0x1010;\n\t" : "=r"(out) : "r"(value));
    return out;
}

__device__ __forceinline__ uint32_t ptx_dup_high_bf16(uint32_t value) {
    uint32_t out;
    asm volatile("prmt.b32 %0, %1, %1, 0x3232;\n\t" : "=r"(out) : "r"(value));
    return out;
}

__device__ __forceinline__ uint32_t ptx_fma_rn_bf16x2(uint32_t a, uint32_t b, uint32_t acc) {
    uint32_t out;
    asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n\t" : "=&r"(out) : "r"(a), "r"(b), "r"(acc));
    return out;
}

__device__ __forceinline__ float scalar_ptx_mul_rn_no_fma_bf16(float lhs_value, float rhs_value) {
    uint32_t a_bits = __float_as_uint(lhs_value);
    uint32_t b_bits = __float_as_uint(rhs_value);
    uint32_t out_bits;
    asm volatile(
        "{ .reg .f32 fa, fb, fo;\n\t"
        "mov.b32 fa, %1;\n\t"
        "mov.b32 fb, %2;\n\t"
        "mul.rn.f32 fo, fa, fb;\n\t"
        "mov.b32 %0, fo;\n\t"
        "}\n\t"
        : "=r"(out_bits)
        : "r"(a_bits), "r"(b_bits)
    );
    return __uint_as_float(out_bits);
}

__device__ __forceinline__ float scalar_ptx_add_rn_no_fma_bf16(float lhs_value, float rhs_value) {
    uint32_t a_bits = __float_as_uint(lhs_value);
    uint32_t b_bits = __float_as_uint(rhs_value);
    uint32_t out_bits;
    asm volatile(
        "{ .reg .f32 fa, fb, fo;\n\t"
        "mov.b32 fa, %1;\n\t"
        "mov.b32 fb, %2;\n\t"
        "add.rn.f32 fo, fa, fb;\n\t"
        "mov.b32 %0, fo;\n\t"
        "}\n\t"
        : "=r"(out_bits)
        : "r"(a_bits), "r"(b_bits)
    );
    return __uint_as_float(out_bits);
}

__device__ __forceinline__ __nv_bfloat16 bf16_from_u16(uint32_t bits) {
    union {
        uint16_t u;
        __nv_bfloat16 b;
    } value;
    value.u = static_cast<uint16_t>(bits);
    return value.b;
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
    uint32_t c_reg[TM][TN / 2];
    uint32_t a_frag[TM];
    uint32_t b_frag[TN / 2];

    int4 ldg_a_reg[2]; 
    int4 ldg_b_reg[2];

    // 初始化 C 寄存器
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN / 2; ++j) {
            c_reg[i][j] = 0u;
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
            
            bool row_valid = (r < 128 && c < 32 && g_r < M);
            int remain = K - g_c;
            int valid_count = remain >= 8 ? 8 : (remain > 0 ? remain : 0);
            // 使用安全加载函数
            load_gmem_vectorized(A_ptr + g_r * lda + g_c, ldg_a_reg[i], row_valid, valid_count);
        }

        // Load B
        #pragma unroll
        for(int i=0; i<2; ++i) {
            int r = (tid / 16) + i * 16;
            int c = (tid % 16) * 8;
            int g_r = r; // k=0
            int g_c = bx * BN + c;
            
            bool row_valid = (r < 32 && c < 128 && g_r < K);
            int remain = N - g_c;
            int valid_count = remain >= 8 ? 8 : (remain > 0 ? remain : 0);
            load_gmem_vectorized(B_ptr + g_r * ldb + g_c, ldg_b_reg[i], row_valid, valid_count);
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
                
                bool row_valid = (r < 128 && c < 32 && g_r < M);
                int remain = K - g_c;
                int valid_count = remain >= 8 ? 8 : (remain > 0 ? remain : 0);
                load_gmem_vectorized(A_ptr + g_r * lda + g_c, ldg_a_reg[i], row_valid, valid_count);
            }
            // Load B
            #pragma unroll
            for(int i=0; i<2; ++i) {
                int r = (tid / 16) + i * 16;
                int c = (tid % 16) * 8;
                int g_r = next_k + r;
                int g_c = bx * BN + c;
                
                bool row_valid = (r < 32 && c < 128 && g_r < K);
                int remain = N - g_c;
                int valid_count = remain >= 8 ? 8 : (remain > 0 ? remain : 0);
                load_gmem_vectorized(B_ptr + g_r * ldb + g_c, ldg_b_reg[i], row_valid, valid_count);
            }
        }

        // 3. Compute
        #pragma unroll
        for (int k_step = 0; k_step < BK; ++k_step) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_frag[i] = ptx_ld_shared_u16_bf16(&smem_a[compute_stage][ty * TM + i][k_step]);
            }

            // SMEM load B: packed int4. smem_b layout guarantees alignment
            int4 b_vec = ptx_ld_shared_v4_b32_bf16(&smem_b[compute_stage][k_step][tx * TN]);
            b_frag[0] = static_cast<uint32_t>(b_vec.x);
            b_frag[1] = static_cast<uint32_t>(b_vec.y);
            b_frag[2] = static_cast<uint32_t>(b_vec.z);
            b_frag[3] = static_cast<uint32_t>(b_vec.w);

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                uint32_t a_val = ptx_dup_low_bf16(a_frag[i]);
                #pragma unroll
                for (int j = 0; j < TN / 2; ++j) {
                    c_reg[i][j] = ptx_fma_rn_bf16x2(a_val, b_frag[j], c_reg[i][j]);
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
                    uint32_t out = c_reg[i][j];
                    C[g_r * ldc + g_c] = bf16_from_u16(out);
                    if (g_c + 1 < N) {
                        C[g_r * ldc + g_c + 1] = bf16_from_u16(out >> 16);
                    }
                }
            }
        }
    }
}

__global__ void __launch_bounds__(256) gemm_bf16_safe_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc)
{
    __shared__ alignas(16) __nv_bfloat16 smem_a[2][BM][BK + PAD];
    __shared__ alignas(16) __nv_bfloat16 smem_b[2][BK][BN + PAD];

    float c_reg[TM][TN];
    __nv_bfloat16 a_frag[TM];
    __nv_bfloat16 b_frag[TN];

    int4 ldg_a_reg[2];
    int4 ldg_b_reg[2];

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            c_reg[i][j] = 0.0f;
        }
    }

    int tid = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int ty = tid / 16;
    int tx = tid % 16;

    const __nv_bfloat16* A_ptr = A;
    const __nv_bfloat16* B_ptr = B;

    int num_tiles = (K + BK - 1) / BK;
    int write_stage = 0;
    int compute_stage = 0;

    {
        #pragma unroll
        for(int i=0; i<2; ++i) {
            int r = (tid / 4) + i * 64;
            int c = (tid % 4) * 8;
            int g_r = by * BM + r;
            int g_c = c;

            bool row_valid = (r < 128 && c < 32 && g_r < M);
            int remain = K - g_c;
            int valid_count = remain >= 8 ? 8 : (remain > 0 ? remain : 0);
            load_gmem_vectorized(A_ptr + g_r * lda + g_c, ldg_a_reg[i], row_valid, valid_count);
        }

        #pragma unroll
        for(int i=0; i<2; ++i) {
            int r = (tid / 16) + i * 16;
            int c = (tid % 16) * 8;
            int g_r = r;
            int g_c = bx * BN + c;

            bool row_valid = (r < 32 && c < 128 && g_r < K);
            int remain = N - g_c;
            int valid_count = remain >= 8 ? 8 : (remain > 0 ? remain : 0);
            load_gmem_vectorized(B_ptr + g_r * ldb + g_c, ldg_b_reg[i], row_valid, valid_count);
        }
    }

    for (int k = 0; k < num_tiles; ++k) {
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

        if (k < num_tiles - 1) {
            int next_k = (k + 1) * BK;

            #pragma unroll
            for(int i=0; i<2; ++i) {
                int r = (tid / 4) + i * 64;
                int c = (tid % 4) * 8;
                int g_r = by * BM + r;
                int g_c = next_k + c;

                bool row_valid = (r < 128 && c < 32 && g_r < M);
                int remain = K - g_c;
                int valid_count = remain >= 8 ? 8 : (remain > 0 ? remain : 0);
                load_gmem_vectorized(A_ptr + g_r * lda + g_c, ldg_a_reg[i], row_valid, valid_count);
            }

            #pragma unroll
            for(int i=0; i<2; ++i) {
                int r = (tid / 16) + i * 16;
                int c = (tid % 16) * 8;
                int g_r = next_k + r;
                int g_c = bx * BN + c;

                bool row_valid = (r < 32 && c < 128 && g_r < K);
                int remain = N - g_c;
                int valid_count = remain >= 8 ? 8 : (remain > 0 ? remain : 0);
                load_gmem_vectorized(B_ptr + g_r * ldb + g_c, ldg_b_reg[i], row_valid, valid_count);
            }
        }

        #pragma unroll
        for (int k_step = 0; k_step < BK; ++k_step) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_frag[i] = smem_a[compute_stage][ty * TM + i][k_step];
            }

            int4 b_vec = ptx_ld_shared_v4_b32_bf16(&smem_b[compute_stage][k_step][tx * TN]);
            __nv_bfloat16* b_ptr = reinterpret_cast<__nv_bfloat16*>(&b_vec);

            #pragma unroll
            for(int j=0; j < TN; ++j) {
                b_frag[j] = b_ptr[j];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                float a_val = __bfloat162float(a_frag[i]);
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    float prod = scalar_ptx_mul_rn_no_fma_bf16(a_val, __bfloat162float(b_frag[j]));
                    c_reg[i][j] = scalar_ptx_add_rn_no_fma_bf16(c_reg[i][j], prod);
                }
            }
        }

        write_stage ^= 1;
        compute_stage ^= 1;
        __syncthreads();
    }

    int global_row_start = by * BM + ty * TM;
    int global_col_start = bx * BN + tx * TN;

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int g_r = global_row_start + i;
        if (g_r < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int g_c = global_col_start + j;
                if (g_c < N) {
                    C[g_r * ldc + g_c] = __float2bfloat16_rn(c_reg[i][j]);
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

void launch_matmul_bf16_safe(
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

    cudaFuncSetAttribute(gemm_bf16_safe_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    gemm_bf16_safe_kernel<<<grid_size, block_size>>>(
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
