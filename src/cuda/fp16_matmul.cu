#include <cstdio>
#include <cstdint>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// =================================================================================
// Part 1: Configuration & PTX Helpers
// =================================================================================

// 配置参数
#define BLOCK_SIZE 256
#define BM 128
#define BN 128
#define BK 64          
#define STAGES 3       

#define TM 8
#define TN 8

// Shared Memory Padding (8 halves = 16 bytes) to reduce bank conflicts
#define PAD 8 

using int4_copy_t = int4;

// 辅助：获取 Shared Memory 偏移量 (Safe cast)
__device__ __forceinline__ uint32_t get_smem_offset(void* ptr) {
    // 显式先转为 uintptr_t 避免 64-bit 指针截断警告，再取低32位作为 offset
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// 异步拷贝: Global -> Shared (16 bytes, ZFILL 模式)
__device__ __forceinline__ void cp_async_pred_zfill(void* smem_ptr, const void* glob_ptr, bool src_valid) {
    uint32_t smem_int_ptr = get_smem_offset(smem_ptr);
    if (src_valid) {
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"(smem_int_ptr), "l"(glob_ptr)
        );
    } else {
        // ZFILL: 越界部分填充 0
        int4* s = reinterpret_cast<int4*>(smem_ptr);
        *s = make_int4(0,0,0,0);
    }
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_group(int n) {
    // n 必须是编译期常量，这里通过 switch 展开
    // Wait until N groups are left pending.
    if (n == 0) asm volatile("cp.async.wait_group 0;\n" ::);
    else if (n == 1) asm volatile("cp.async.wait_group 1;\n" ::);
    else if (n == 2) asm volatile("cp.async.wait_group 2;\n" ::);
}

// =================================================================================
// Part 2: Optimized Kernel (Triple Buffering, CUDA Cores FP16)
// =================================================================================

__global__ void __launch_bounds__(BLOCK_SIZE) gemm_fp16_ampere_optimized_v3(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m, int n, int k,
    int lda, int ldb, int ldc) 
{
    // 动态 Shared Memory
    extern __shared__ char smem_raw[];
    
    // 指针算术：计算 As 和 Bs 在 smem 中的位置
    // As: [STAGES][BM][BK + PAD]
    // Bs: [STAGES][BK][BN + PAD]
    // 注意：Bs 为了优化加载，通常需要在写回 SMEM 时考虑布局，
    // 但此处为了保持代码简洁且不依赖外部库，采用了直接加载，计算时利用 Cache Line 连续性。
    
    half (*As)[BM][BK + PAD] = reinterpret_cast<half (*)[BM][BK + PAD]>(smem_raw);
    
    half (*Bs)[BK][BN + PAD] = reinterpret_cast<half (*)[BK][BN + PAD]>(
        smem_raw + STAGES * BM * (BK + PAD) * sizeof(half)
    );

    // 寄存器累加器
    half2 accum[TM][TN / 2];
    // 寄存器片段
    half2 frag_a[TM / 2]; 
    half2 frag_b[TN / 2]; 

    // 初始化累加器 (Constraint 1: 避免 FP32 FMA，初始化为 half2 0.0)
    #pragma unroll
    for(int i = 0; i < TM; i++) {
        #pragma unroll
        for(int j = 0; j < TN / 2; j++) {
            accum[i][j] = __float2half2_rn(0.0f);
        }
    }

    int tid = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int block_row_start = by * BM;
    int block_col_start = bx * BN;

    // --- 线程映射计算 (Load) ---
    // 每个线程负责加载 128 bit (16 bytes = 8 halves)
    int load_a_row = tid / 8;        
    int load_a_col = (tid % 8) * 8;  

    int load_b_row = tid / 16;       
    int load_b_col = (tid % 16) * 8; 

    // --- 线程映射计算 (Compute) ---
    int ty = tid / 16; 
    int tx = tid % 16; 
    int thread_row_start = ty * TM; 
    int thread_col_start = tx * TN; 

    // =========================================================================
    // Prologue: Fill Pipeline
    // =========================================================================
    
    int max_k_tiles = (k + BK - 1) / BK;

    // 预加载 Stage 0
    if (max_k_tiles > 0) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = load_a_row + i * 32; 
            int c = load_a_col;
            bool valid = (block_row_start + r < m) && (c < k);
            cp_async_pred_zfill(&As[0][r][c], a + (block_row_start + r) * lda + c, valid);
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = load_b_row + i * 16; 
            int c = load_b_col;
            bool valid = (r < k) && (block_col_start + c < n);
            cp_async_pred_zfill(&Bs[0][r][c], b + r * ldb + (block_col_start + c), valid);
        }
        cp_async_commit();
    }

    // 预加载 Stage 1
    if (max_k_tiles > 1) {
        int k_offset = BK;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = load_a_row + i * 32; 
            int c = load_a_col;
            bool valid = (block_row_start + r < m) && (k_offset + c < k);
            cp_async_pred_zfill(&As[1][r][c], a + (block_row_start + r) * lda + (k_offset + c), valid);
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = load_b_row + i * 16; 
            int c = load_b_col;
            bool valid = (k_offset + r < k) && (block_col_start + c < n);
            cp_async_pred_zfill(&Bs[1][r][c], b + (k_offset + r) * ldb + (block_col_start + c), valid);
        }
        cp_async_commit();
    }

    // =========================================================================
    // Main Loop
    // =========================================================================
    
    for (int k_step = 0; k_step < max_k_tiles; k_step++) {
        
        int compute_stage = k_step % 3;
        int load_stage = (k_step + 2) % 3; 

        // 1. 等待数据准备就绪
        // 标准 3-stage 流水线：我们希望保持 (STAGES - 1) 个组在飞行中。
        // 等待直到只剩下 (STAGES - 2) = 1 个组未完成（即当前计算所需的组必须完成）。
        // 修正逻辑：如果总 Tile 很小 (<=2)，我们发出的组可能不够，必须使用 wait_group(0) 保证安全。
        if (max_k_tiles > 2) {
             cp_async_wait_group(1); 
        } else {
             cp_async_wait_group(0);
        }
        
        __syncthreads();

        // 2. Compute (Fully compliant with Constraints)
        // No FP32 FMA here, only integer index calc and __hfma2
        #pragma unroll
        for (int k_inner = 0; k_inner < BK; k_inner++) {
            
            // Load A to Reg (Scalar load broadcasted across warp)
            // As is [Row][Col]. We access [Row][Fixed_Col].
            // Threads in warp have different Row, same Col.
            #pragma unroll
            for (int i = 0; i < TM; i+=2) {
                // 读取两个 half，组合成 half2。Constraint 1: 不使用 float 中间量。
                half a1 = As[compute_stage][thread_row_start + i][k_inner];
                half a2 = As[compute_stage][thread_row_start + i + 1][k_inner];
                frag_a[i/2] = __halves2half2(a1, a2);
            }

            // Load B to Reg (Vectorized load)
            // Bs is [K][N]. We access [Fixed_K][Vector_Col].
            // 确保指针对齐: Bs 起始地址是对齐的，N维度步长(BN+PAD)也是偶数。
            int4 b_vec = *reinterpret_cast<int4*>(&Bs[compute_stage][k_inner][thread_col_start]);
            const half2* b_h2_ptr = reinterpret_cast<const half2*>(&b_vec);
            
            frag_b[0] = b_h2_ptr[0];
            frag_b[1] = b_h2_ptr[1];
            frag_b[2] = b_h2_ptr[2];
            frag_b[3] = b_h2_ptr[3];

            // Outer Product (Constraint 1 & 2: Use FP16 FMA, No Tensor Core)
            #pragma unroll
            for (int i = 0; i < TM / 2; i++) {
                // 将 A 的向量拆分为两个标量进行广播计算
                // __low2half2 将 x 复制到 xy, __high2half2 将 y 复制到 xy
                half2 a_top = __half2half2(frag_a[i].x); 
                half2 a_bot = __half2half2(frag_a[i].y); 
                
                #pragma unroll
                for (int j = 0; j < TN / 2; j++) {
                    // C = A * B + C using FP16 HW instruction
                    accum[i*2][j]   = __hfma2(a_top, frag_b[j], accum[i*2][j]);
                    accum[i*2+1][j] = __hfma2(a_bot, frag_b[j], accum[i*2+1][j]);
                }
            }
        }

        // 3. Issue Load for (k + 2)
        int next_k_idx = k_step + 2;
        if (next_k_idx < max_k_tiles) {
            int k_offset = next_k_idx * BK;
            
            // Load A
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int r = load_a_row + i * 32;
                int c = load_a_col;
                bool valid = (block_row_start + r < m) && (k_offset + c < k);
                cp_async_pred_zfill(&As[load_stage][r][c], a + (block_row_start + r) * lda + (k_offset + c), valid);
            }
            // Load B
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int r = load_b_row + i * 16;
                int c = load_b_col;
                bool valid = (k_offset + r < k) && (block_col_start + c < n);
                cp_async_pred_zfill(&Bs[load_stage][r][c], b + (k_offset + r) * ldb + (block_col_start + c), valid);
            }
            // Commit group
            cp_async_commit();
        }
    }

    // Epilogue: Wait for all pending copies (Constraint: cleanup)
    cp_async_wait_group(0);
    __syncthreads();

    // =========================================================================
    // Store Result
    // =========================================================================
    int c_row = block_row_start + thread_row_start;
    int c_col = block_col_start + thread_col_start;

    // Constraint 1: 这里只是存储，不涉及计算。
    // 如果需要 Alpha/Beta 混合，必须注意不要使用 FP32 乘法，除非拆分为 __fmul_rn
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        if (c_row + i < m) {
            #pragma unroll
            for (int j = 0; j < TN / 2; j++) {
                if (c_col + j * 2 < n) {
                    // 直接存储 FP16
                    c[(c_row + i) * ldc + (c_col + j * 2)]     = accum[i][j].x;
                    if (c_col + j * 2 + 1 < n)
                        c[(c_row + i) * ldc + (c_col + j * 2 + 1)] = accum[i][j].y;
                }
            }
        }
    }
}

// =================================================================================
// Part 3: Helper Functions & Launcher
// =================================================================================

static void set_l2_persistence(cudaStream_t stream, const void* ptr, size_t size_bytes) {
#if CUDART_VERSION >= 11000
    cudaStreamAttrValue stream_attribute;
    stream_attribute.accessPolicyWindow.base_ptr  = const_cast<void*>(ptr);
    stream_attribute.accessPolicyWindow.num_bytes = size_bytes;
    stream_attribute.accessPolicyWindow.hitRatio  = 1.0; 
    stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
#endif
}

void launch_matmul_fp16(
    const void* input_ptr, 
    const void* weight_ptr, 
    void* output_ptr, 
    int m, int n, int k) 
{
    const half* input = reinterpret_cast<const half*>(input_ptr);
    const half* weight = reinterpret_cast<const half*>(weight_ptr);
    half* output = reinterpret_cast<half*>(output_ptr);

    int lda = k;
    int ldb = n; 
    int ldc = n;

    cudaStream_t stream = 0;

    dim3 block(BLOCK_SIZE, 1);
    dim3 grid(
        (n + BN - 1) / BN, 
        (m + BM - 1) / BM
    );

    // 计算动态 Shared Memory 大小
    int as_bytes = STAGES * BM * (BK + PAD) * sizeof(half);
    int bs_bytes = STAGES * BK * (BN + PAD) * sizeof(half);
    int total_smem_bytes = as_bytes + bs_bytes; 

    // 设置 Kernel 属性允许 > 48KB SMEM
    cudaFuncSetAttribute(gemm_fp16_ampere_optimized_v3, 
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 
                         total_smem_bytes);
    
    // 设置 L2 Cache 驻留 (Weight Stationary 优化)
    size_t b_size = (size_t)k * n * sizeof(half);
    set_l2_persistence(stream, weight_ptr, b_size);

    gemm_fp16_ampere_optimized_v3<<<grid, block, total_smem_bytes, stream>>>(
        input, weight, output, m, n, k, lda, ldb, ldc
    );
}

// Bias Helper (符合 Constraint 1, 3-7: 只使用 __hadd2)
__global__ void add_bias_fp16_vectorized(
    half2* __restrict__ output, 
    const half2* __restrict__ bias,
    int total_vec_elements, 
    int width_h2 
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_vec_elements) {
        int col = idx % width_h2;
        // Constraint 1: 使用 hardware FP16 add, 不是 FP32 FMA
        output[idx] = __hadd2(output[idx], bias[col]);
    }
}

void launch_add_bias_fp16(void* output_ptr, const void* bias_ptr, int rows, int cols) {
    if (cols % 2 != 0) return; // 简单保护
    half2* output = reinterpret_cast<half2*>(output_ptr);
    const half2* bias = reinterpret_cast<const half2*>(bias_ptr);
    int total_vec_elements = (rows * cols) / 2;
    int width_h2 = cols / 2;
    int threads = 256;
    int blocks = (total_vec_elements + threads - 1) / threads;
    add_bias_fp16_vectorized<<<blocks, threads>>>(output, bias, total_vec_elements, width_h2);
}