#include <cstdio>
#include <cstdint>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// =================================================================================
// Part 1: Configuration
// =================================================================================

// Configuration compliant with User Requirements:
// 1. No FP32 FMA (Use __fmul_rn, __fadd_rn if needed, but here we use Int arithmetic).
// 2. No Tensor Cores / Libraries.
// 3-7. No forbidden FP16 math functions.

#define BLOCK_SIZE 1024
#define BM 256      // Block M (Pixels)
#define BN 128      // Block N (Channels)
#define BK 32       // Block K (Accumulation axis)
#define STAGES 3    // Pipeline stages

// Register Tile per thread: 4x8
#define TM 4
#define TN 8

// Shared Memory Padding to reduce bank conflicts
#define PAD 8 

// =================================================================================
// Part 2: Optimized Conv2D Kernel (SIMT / FP16 Core)
// =================================================================================

__global__ void __launch_bounds__(BLOCK_SIZE) conv2d_fp16_ampere_optimized(
    const half* __restrict__ input,     // [B, C_in, H_in, W_in]
    const half* __restrict__ weight,    // [C_out, C_in, KH, KW]
    const half* __restrict__ bias,
    half* __restrict__ output,          // [B, C_out, H_out, W_out]
    int B, int C_in, int H_in, int W_in,
    int C_out, int K_H, int K_W,
    int H_out, int W_out,
    int s_h, int s_w, int p_h, int p_w, int d_h, int d_w) 
{
    // 动态 Shared Memory
    extern __shared__ char smem_raw[];
    
    // Matrix A (Input): [BM, BK] -> [256, 32]
    // Layout: [STAGES][BM][BK + PAD]
    half (*As)[BM][BK + PAD] = reinterpret_cast<half (*)[BM][BK + PAD]>(smem_raw);
    
    // Matrix B (Weight): [BK, BN] -> [32, 128]
    // Layout: [STAGES][BK][BN + PAD]
    // Smem offset based on As size
    half (*Bs)[BK][BN + PAD] = reinterpret_cast<half (*)[BK][BN + PAD]>(
        smem_raw + STAGES * BM * (BK + PAD) * sizeof(half)
    );

    // 寄存器累加器: [TM][TN] = [4][8] elements
    // 使用 half2 存储，所以是 [TM][TN/2] = [4][4] half2s
    half2 accum[TM][TN / 2];
    
    // 片段寄存器
    // Frag A: TM=4 elements -> 2 half2s
    half2 frag_a[TM / 2]; 
    // Frag B: TN=8 elements -> 4 half2s
    half2 frag_b[TN / 2]; 

    // 初始化累加器
    #pragma unroll
    for(int i = 0; i < TM; i++) {
        #pragma unroll
        for(int j = 0; j < TN / 2; j++) {
            accum[i][j] = __float2half2_rn(0.0f);
        }
    }

    int tid = threadIdx.x;
    int bx = blockIdx.x; // Block Index for N (Output Channels)
    int by = blockIdx.y; // Block Index for M (Pixels)

    // -----------------------------------------------------------
    // 线程映射 (Thread Mapping)
    // -----------------------------------------------------------
    // Warp Tile: 32x32. Block Tile: 256x128.
    
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int warp_m = warp_id / 4; // 0..7
    int warp_n = warp_id % 4; // 0..3

    int lane_m = lane_id / 4; // 0..7
    int lane_n = lane_id % 4; // 0..3

    // 线程负责计算的全局坐标偏移 (相对于 Block Start)
    int thread_m_start = warp_m * 32 + lane_m * TM; // TM=4
    int thread_n_start = warp_n * 32 + lane_n * TN; // TN=8

    // 全局坐标基准
    int block_pixel_start = by * BM;
    int block_channel_start = bx * BN;
    
    // 预计算常量
    int total_pixels = B * H_out * W_out;
    int total_k = C_in * K_H * K_W;
    int stride_hw = H_out * W_out;
    int stride_chw = C_out * stride_hw; // For output calculation

    // -----------------------------------------------------------
    // 加载映射 (Loading Mapping)
    // -----------------------------------------------------------
    
    // Load A: [256, 32]. 1024 threads. 1 int4 (8 halves) per thread.
    int load_a_row = tid / 4;       // 0..255
    int load_a_col = (tid % 4) * 8; // 0, 8, 16, 24

    // Optimization: Pre-calculate coordinates for Input loading
    // Since load_a_row is fixed for the thread, pixel_idx is fixed.
    int load_a_pixel_idx = block_pixel_start + load_a_row;
    int load_a_b_idx = 0, load_a_h_out = 0, load_a_w_out = 0;
    bool load_a_valid_pixel = (load_a_pixel_idx < total_pixels);

    if (load_a_valid_pixel) {
        load_a_b_idx = load_a_pixel_idx / stride_hw;
        int rem = load_a_pixel_idx % stride_hw;
        load_a_h_out = rem / W_out;
        load_a_w_out = rem % W_out;
    }

    // Load B: [32, 128]. 4096 elements.
    // 512 threads active. 1 int4 per thread.
    int load_b_row = tid / 16;      // 0..31
    int load_b_col = (tid % 16) * 8;// 0..120 (step 8)
    bool load_b_active = (tid < 512);

    int max_k_tiles = (total_k + BK - 1) / BK;

    // =========================================================================
    // Pipeline Prologue
    // =========================================================================

    if (max_k_tiles > 0) {
        // --- Load A (Input) ---
        {
            // Register buffer for loading
            int4 load_val = make_int4(0,0,0,0);
            half* vec_data = reinterpret_cast<half*>(&load_val);
            
            // Current K position in the GEMM K dimension
            int k_idx_base = 0 + load_a_col; 

            if (load_a_valid_pixel && k_idx_base < total_k) {
                #pragma unroll
                for (int v = 0; v < 8; ++v) {
                    int curr_k = k_idx_base + v;
                    if (curr_k < total_k) {
                        // Unpack K to (C, R, S)
                        int tmp = curr_k;
                        int kv = tmp % K_W; tmp /= K_W;
                        int ku = tmp % K_H; 
                        int kc = tmp / K_H;
                        
                        // Calculate Input Coordinates
                        int h_in = load_a_h_out * s_h - p_h + ku * d_h;
                        int w_in = load_a_w_out * s_w - p_w + kv * d_w;

                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            // Address Math: Int only, no FMA
                            int offset = ((load_a_b_idx * C_in + kc) * H_in + h_in) * W_in + w_in;
                            vec_data[v] = input[offset];
                        }
                    }
                }
            }
            // Write to Shared Memory
            *reinterpret_cast<int4*>(&As[0][load_a_row][load_a_col]) = load_val;
        }

        // --- Load B (Weight) ---
        if (load_b_active) {
            int k = load_b_row; // K dim of GEMM
            int n = load_b_col; // N dim of GEMM
            
            int global_n = block_channel_start + n;
            int global_k = 0 + k;

            int4 load_val = make_int4(0,0,0,0);
            half* vec_data = reinterpret_cast<half*>(&load_val);
            
            if (global_n < C_out && global_k < total_k) {
                 #pragma unroll
                 for(int v=0; v<8; ++v) {
                     int curr_n = global_n + v; // Load 8 channels (N)
                     if (curr_n < C_out) {
                         // Weight Layout assumed [C_out, K] (Flattended from [C_out, C_in, KH, KW])
                         // Address = n * total_k + k
                         vec_data[v] = weight[curr_n * total_k + global_k];
                     }
                 }
            }
            *reinterpret_cast<int4*>(&Bs[0][k][n]) = load_val;
        }
        
        __syncthreads(); 
    }

    // =========================================================================
    // Main Loop
    // =========================================================================
    
    for (int k_step = 0; k_step < max_k_tiles; k_step++) {
        int compute_idx = k_step % STAGES;
        int load_idx = (k_step + 1) % STAGES;

        // 1. Prefetch Next Tile
        if (k_step + 1 < max_k_tiles) {
             int next_k_base = (k_step + 1) * BK;
             
             // --- Load A (Next) ---
             {
                int k_idx_base = next_k_base + load_a_col;
                int4 load_val = make_int4(0,0,0,0);
                half* vec_data = reinterpret_cast<half*>(&load_val);
                
                if (load_a_valid_pixel && k_idx_base < total_k) {
                    #pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        int curr_k = k_idx_base + v;
                        if (curr_k < total_k) {
                            int tmp = curr_k;
                            int kv = tmp % K_W; tmp /= K_W;
                            int ku = tmp % K_H; 
                            int kc = tmp / K_H;
                            int h_in = load_a_h_out * s_h - p_h + ku * d_h;
                            int w_in = load_a_w_out * s_w - p_w + kv * d_w;
                            if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                                int offset = ((load_a_b_idx * C_in + kc) * H_in + h_in) * W_in + w_in;
                                vec_data[v] = input[offset];
                            }
                        }
                    }
                }
                *reinterpret_cast<int4*>(&As[load_idx][load_a_row][load_a_col]) = load_val;
             }

             // --- Load B (Next) ---
             if (load_b_active) {
                int k = load_b_row;
                int n = load_b_col;
                int global_n = block_channel_start + n;
                int global_k = next_k_base + k;
                
                int4 load_val = make_int4(0,0,0,0);
                half* vec_data = reinterpret_cast<half*>(&load_val);
                
                if (global_n < C_out && global_k < total_k) {
                     #pragma unroll
                     for(int v=0; v<8; ++v) {
                         int curr_n = global_n + v;
                         if (curr_n < C_out) {
                             vec_data[v] = weight[curr_n * total_k + global_k];
                         }
                     }
                }
                *reinterpret_cast<int4*>(&Bs[load_idx][k][n]) = load_val;
             }
        }
        
        __syncthreads();

        // 2. Compute (Outer Product)
        // Tile size 4x8. Iterate k_inner.
        #pragma unroll
        for (int k_inner = 0; k_inner < BK; k_inner++) {
            
            // Load A Vector (M dimension): 4 elements -> 2 half2s
            // Matrix A is [BM][BK]. We need [row][k].
            half a0 = As[compute_idx][thread_m_start + 0][k_inner];
            half a1 = As[compute_idx][thread_m_start + 1][k_inner];
            half a2 = As[compute_idx][thread_m_start + 2][k_inner];
            half a3 = As[compute_idx][thread_m_start + 3][k_inner];
            
            frag_a[0] = __halves2half2(a0, a1);
            frag_a[1] = __halves2half2(a2, a3);

            // Load B Vector (N dimension): 8 elements -> 4 half2s
            // Matrix B is [BK][BN]. We need [k][col].
            int4 b_vec = *reinterpret_cast<int4*>(&Bs[compute_idx][k_inner][thread_n_start]);
            const half2* b_h2_ptr = reinterpret_cast<const half2*>(&b_vec);
            
            frag_b[0] = b_h2_ptr[0];
            frag_b[1] = b_h2_ptr[1];
            frag_b[2] = b_h2_ptr[2];
            frag_b[3] = b_h2_ptr[3];

            // Outer Product (4x8) using __hfma2
            
            // Row 0 & 1
            half2 a_val0 = __half2half2(frag_a[0].x); // Broadcast a0
            half2 a_val1 = __half2half2(frag_a[0].y); // Broadcast a1
            
            #pragma unroll
            for (int j = 0; j < 4; j++) { 
                accum[0][j] = __hfma2(a_val0, frag_b[j], accum[0][j]);
                accum[1][j] = __hfma2(a_val1, frag_b[j], accum[1][j]);
            }

            // Row 2 & 3
            half2 a_val2 = __half2half2(frag_a[1].x); // Broadcast a2
            half2 a_val3 = __half2half2(frag_a[1].y); // Broadcast a3
            
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                accum[2][j] = __hfma2(a_val2, frag_b[j], accum[2][j]);
                accum[3][j] = __hfma2(a_val3, frag_b[j], accum[3][j]);
            }
        }
        
        __syncthreads();
    }

    // =========================================================================
    // Store Result
    // =========================================================================
    
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int m_idx = block_pixel_start + thread_m_start + i;
        
        if (m_idx < total_pixels) {
            // Re-calculate M coords for output (register pressure is low here)
            int b_idx = m_idx / stride_hw;
            int rem   = m_idx % stride_hw; 

            // Processing 8 columns (4 half2s) per row
            #pragma unroll
            for (int j = 0; j < TN / 2; j++) {
                 int n_idx_base = block_channel_start + thread_n_start + j * 2;
                 
                 half val0 = accum[i][j].x;
                 half val1 = accum[i][j].y;
                 
                 // 处理第一个通道 (n_idx_base)
                 if (n_idx_base < C_out) {
                     if (bias) val0 = __hadd(val0, bias[n_idx_base]);
                     // Int arithmetic strictly
                     int out_offset = b_idx * stride_chw + n_idx_base * stride_hw + rem;
                     output[out_offset] = val0;
                 }

                 // 处理第二个通道 (n_idx_base + 1)
                 if (n_idx_base + 1 < C_out) {
                     if (bias) val1 = __hadd(val1, bias[n_idx_base + 1]);
                     int out_offset = b_idx * stride_chw + (n_idx_base + 1) * stride_hw + rem;
                     output[out_offset] = val1;
                 }
            }
        }
    }
}

// =================================================================================
// Part 3: Launcher
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

void launch_conv2d_fp16(
    const void* input, const void* weight, const void* bias, void* output,
    int B, int C_in, int H_in, int W_in, 
    int C_out, int K_H, int K_W,
    int H_out, int W_out,
    int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w
) {
    const half* input_ptr = reinterpret_cast<const half*>(input);
    const half* weight_ptr = reinterpret_cast<const half*>(weight);
    const half* bias_ptr = reinterpret_cast<const half*>(bias);
    half* output_ptr = reinterpret_cast<half*>(output);

    int m = B * H_out * W_out;
    int n = C_out;
    // int k = C_in * K_H * K_W; // Unused in grid calc

    dim3 block(BLOCK_SIZE, 1);
    
    // Grid adjustment for BM=256, BN=128
    dim3 grid(
        (n + BN - 1) / BN, 
        (m + BM - 1) / BM
    );

    // Shared Memory Calculation
    // As: [STAGES][BM][BK+PAD]
    // Bs: [STAGES][BK][BN+PAD]
    int as_bytes = STAGES * BM * (BK + PAD) * sizeof(half);
    int bs_bytes = STAGES * BK * (BN + PAD) * sizeof(half);
    int total_smem_bytes = as_bytes + bs_bytes; 
    
    // As = 3 * 256 * 40 * 2 = 61440 bytes
    // Bs = 3 * 32 * 136 * 2 = 26112 bytes
    // Total ~= 87KB. Needs configuration.

    // Set L2 persistence for Weights (Reuse heavy)
    size_t k_dim = (size_t)C_in * K_H * K_W;
    size_t w_size = (size_t)C_out * k_dim * sizeof(half);
    set_l2_persistence(0, weight_ptr, w_size);

    // Maximize Shared Memory
    cudaFuncSetAttribute(conv2d_fp16_ampere_optimized, 
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 
                         total_smem_bytes);

    conv2d_fp16_ampere_optimized<<<grid, block, total_smem_bytes, 0>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        B, C_in, H_in, W_in, C_out, K_H, K_W, H_out, W_out,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w
    );
}