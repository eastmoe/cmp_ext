#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>


// ==========================================
// GA100 (Ampere) 针对性优化参数
// ==========================================
// 使用 128x128 的大分块以利用 A100 的 L2 Cache 和 Shared Memory
#define BM 128
#define BN 128
#define BK 16   // K 维度一次步进 16，增加指令流水线深度
#define TM 8    // 每个线程负责 C 的 8 行
#define TN 8    // 每个线程负责 C 的 8 列

// 线程块大小：(BM/TM) * (BN/TN) = 16 * 16 = 256 线程
// 这与之前的 block_size 数量不同，但这是内部实现细节，外部无需感知

// 强制不使用 FMA (Fused Multiply-Add)
__device__ __forceinline__ float mul_no_fma(float a, float b) {
    return __fmul_rn(a, b);
}

__device__ __forceinline__ float add_no_fma(float a, float b) {
    return __fadd_rn(a, b);
}

// ==========================================
// GA100 FP32 Kernel
// ==========================================
__global__ void gemm_fp32_ga100_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m, int n, int k,
    int lda, int ldb, int ldc) 
{
    // Shared Memory 声明
    // s_a: 存储 A 的分块。为了让计算核心能连续读取(LDS.128)，
    // 我们在这里逻辑上将 A 转置存储为 [BK][BM]。
    // +4 padding 防止 bank conflict
    __shared__ float s_a[BK][BM + 4]; 
    
    // s_b: 存储 B 的分块。B 本身是 KxN，取块为 BKxBN。
    // 为了计算时连续读取，存储为 [BK][BN]。
    __shared__ float s_b[BK][BN + 4];

    // 寄存器分块
    float r_c[TM][TN] = {0.0f}; // 8x8 = 64 个累加寄存器
    float r_a[TM];              // 缓存 A
    float r_b[TN];              // 缓存 B

    // 线程索引计算
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int tid = ty * 16 + tx; // 0 ~ 255

    // ----------------------------------------------------
    // 1. 预计算全局内存加载的偏移量
    // ----------------------------------------------------
    
    // 我们需要用 256 个线程加载 Tile_A (128行 x 16列) 和 Tile_B (16行 x 128列)
    // 每个线程加载 2048 / 256 = 8 个 float
    
    // --- 加载 A 的映射 (Global: Row-Major) ---
    // A 的 Tile 是 [128, 16]。我们将其展平为 2048 个元素。
    // 为了 Global Memory 合并访问 (Coalescing)，tid 相邻应访问相邻地址。
    // A 的每一行有 16 个元素。所以 16 个线程正好读完一行。
    // load_a_row: 线程负责读哪一行
    // load_a_col: 线程负责读哪一列
    // tid >> 4 等价于 tid / 16; tid & 15 等价于 tid % 16
    int load_a_row = tid >> 4;   // 0 ~ 15 (但这只覆盖了前16行? 不，这只是偏移基准)
    int load_a_col = tid & 15;   // 0 ~ 15
    // 注意：256 个线程一次只能读 256 个数，而我们需要读 2048 个。
    // 所以需要循环 8 次 (stride = 32行)。
    // 每次迭代加载 offset: 0, 16, 32... 
    
    // --- 加载 B 的映射 (Global: Row-Major) ---
    // B 的 Tile 是 [16, 128]。
    // 连续的 128 个元素在一行。
    // 我们可以让 tid 连续读取。
    int load_b_row = tid >> 7;   // tid / 128 (0 or 1)
    int load_b_col = tid & 127;  // tid % 128
    // 同样需要循环加载，步长为 2行 (256/128=2)

    // ----------------------------------------------------
    // 2. 主循环：K 维度迭代
    // ----------------------------------------------------
    for (int k_idx = 0; k_idx < k; k_idx += BK) {
        
        // --- 加载 A 到 Shared Memory (逻辑转置: Global[row][k] -> Shared[k][row]) ---
        // 我们需要把 A 转置存入 s_a[col][row]，这样计算时按列取就是连续的
        #pragma unroll
        for (int i = 0; i < 8; ++i) { // 8 次迭代覆盖 128 行 (16 * 8 = 128)
            int r = load_a_row + i * 16; // 0, 16, 32 ... 112 + (0..15)
            int c = load_a_col;          // 0..15
            
            // 全局坐标
            int global_r = by * BM + r;
            int global_c = k_idx + c;
            
            float val = 0.0f;
            if (global_r < m && global_c < k) {
                val = a[global_r * lda + global_c];
            }
            // 转置写入: s_a[k_inner][row_inner]
            s_a[c][r] = val;
        }

        // --- 加载 B 到 Shared Memory (直接存储: Global[row][col] -> Shared[row][col]) ---
        // B 存为 s_b[k_inner][col_inner]
        #pragma unroll
        for (int i = 0; i < 8; ++i) { // 8 次迭代覆盖 16 行 (2 * 8 = 16)
            int r = load_b_row + i * 2; // 0, 2, 4 ... 14 + (0..1)
            int c = load_b_col;         // 0..127
            
            int global_r = k_idx + r;
            int global_c = bx * BN + c;

            float val = 0.0f;
            if (global_r < k && global_c < n) {
                val = b[global_r * ldb + global_c];
            }
            s_b[r][c] = val;
        }

        __syncthreads();

        // --- 计算核心 (8x8 Register Tile) ---
        // 循环 BK (16) 次
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            // 从 s_a 读取 8 个 float (A 的一列，对应 C 的 8 行)
            // 因为 s_a 已经转置存储，这里读取 s_a[kk][ty*8 ... ty*8+7] 是连续的！
            // 使用 float4 向量化读取
            float4* ptr_sa = (float4*)&s_a[kk][ty * TM];
            float4 va1 = ptr_sa[0];
            float4 va2 = ptr_sa[1];
            
            r_a[0] = va1.x; r_a[1] = va1.y; r_a[2] = va1.z; r_a[3] = va1.w;
            r_a[4] = va2.x; r_a[5] = va2.y; r_a[6] = va2.z; r_a[7] = va2.w;

            // 从 s_b 读取 8 个 float (B 的一行，对应 C 的 8 列)
            // s_b[kk][tx*8 ... tx*8+7] 也是连续的
            float4* ptr_sb = (float4*)&s_b[kk][tx * TN];
            float4 vb1 = ptr_sb[0];
            float4 vb2 = ptr_sb[1];

            r_b[0] = vb1.x; r_b[1] = vb1.y; r_b[2] = vb1.z; r_b[3] = vb1.w;
            r_b[4] = vb2.x; r_b[5] = vb2.y; r_b[6] = vb2.z; r_b[7] = vb2.w;

            // 外积计算：64 次乘加，严格禁止 FMA
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    float prod = mul_no_fma(r_a[i], r_b[j]);
                    r_c[i][j] = add_no_fma(r_c[i][j], prod);
                }
            }
        }

        __syncthreads();
    }

    // ----------------------------------------------------
    // 3. 结果写回
    // ----------------------------------------------------
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int global_r = by * BM + ty * TM + i;
        if (global_r < m) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int global_c = bx * BN + tx * TN + j;
                if (global_c < n) {
                    c[global_r * ldc + global_c] = r_c[i][j];
                }
            }
        }
    }
}

// ==========================================
// 向量化 Bias Kernel (保持高性能)
// ==========================================
__global__ void add_bias_fp32_vec4_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    int rows, int cols) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int vec_cols = cols / 4;
    int vec_size = rows * vec_cols; 

    // 使用 float4 向量化处理
    float4* out_vec = reinterpret_cast<float4*>(output);
    const float4* bias_vec = reinterpret_cast<const float4*>(bias);
    
    for (int i = tid; i < vec_size; i += stride) {
        int col_vec_idx = i % vec_cols;
        float4 b_val = bias_vec[col_vec_idx];
        float4 o_val = out_vec[i];
        
        o_val.x = add_no_fma(o_val.x, b_val.x);
        o_val.y = add_no_fma(o_val.y, b_val.y);
        o_val.z = add_no_fma(o_val.z, b_val.z);
        o_val.w = add_no_fma(o_val.w, b_val.w);
        
        out_vec[i] = o_val;
    }

    // 处理非 4 倍数尾部
    int tail_start = vec_cols * 4;
    if (tail_start < cols) {
        int total_elements = rows * cols;
        for (int i = tid; i < total_elements; i += stride) {
            int c = i % cols;
            if (c >= tail_start) {
                output[i] = add_no_fma(output[i], bias[c]);
            }
        }
    }
}

// ==========================================
// Launchers (函数签名与原有 C++ 接口保持一致)
// ==========================================

// 您原有的 C++ 接口要求
// void launch_matmul_fp32(const float* input, const float* weight, float* output, int M, int N, int K);

void launch_matmul_fp32(
    const float* input, 
    const float* weight,
    float* output,
    int m, int n, int k) 
{
    // 配置 Stride (假设 Row-Major 且 packed)
    int lda = k;
    int ldb = n;
    int ldc = n;

    // 使用针对 GA100 优化的配置
    // Block: 16x16 线程 (256 threads)
    // Grid:  基于 BM=128, BN=128 进行切分
    dim3 block_size(16, 16);
    dim3 grid_size(
        (n + BM - 1) / BM,
        (m + BN - 1) / BN
    );

    gemm_fp32_ga100_kernel<<<grid_size, block_size>>>(
        input, weight, output, m, n, k, lda, ldb, ldc
    );

    // 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA MatMul FP32 Error: %s\n", cudaGetErrorString(err));
    }
}

// Bias Launcher (签名保持一致)
// void launch_add_bias_fp32(float* output, const float* bias, int rows, int cols); // 假设您的头文件里是这个定义

// 注意：如果您的头文件里定义了这个函数，请确保这里的参数匹配。
// 基于您之前的代码，我保留这个实现。
void launch_add_bias_fp32(float* output, const float* bias, int rows, int cols) {
    int num_threads = 256;
    // 使用 1D Grid 处理整个矩阵，比 2D Grid + 取模运算更高效
    int total_vecs = (rows * cols) / 4;
    // 至少给一个 block
    int num_blocks = (total_vecs + num_threads - 1) / num_threads;
    if (num_blocks == 0) num_blocks = 1;
    if (num_blocks > 65535) num_blocks = 65535;

    add_bias_fp32_vec4_kernel<<<num_blocks, num_threads>>>(output, bias, rows, cols);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
         printf("CUDA Bias FP32 Error: %s\n", cudaGetErrorString(err));
    }
}
//A100特化，5.6T
//内部 Block 调整为 128x128:
//Register Tiling: 每个线程现在计算 8x8 (64个) 结果，而不是 1x1
//Shared Memory Layout: s_a 被设计为在加载时进行逻辑转置。这样在计算核心循环中，线程可以连续读取内存（Coalesced Shared Memory Access），从而利用向量化加载指令
//Loop Unrolling: 关键循环都加上了 #pragma unroll，帮助编译器在 strict no-FMA 模式下更好地调度乘法和加法指令，掩盖延迟。