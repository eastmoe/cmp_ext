#include <cstdio>
#include <cstdint>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// =================================================================================
// Configuration
// =================================================================================

#define TR 16         // Query Tile Size (Rows) -> BlockDim.y
#define TC 64         // Key/Value Tile Size (Rows)
#define MAX_D 128     // Head Dimension
#define WARP_SIZE 32

// Smem Padding: 8 halves (16 bytes) 避免 Bank Conflict
#define SMEM_PAD 8    
#define SMEM_STRIDE (MAX_D + SMEM_PAD)

// =================================================================================
// PTX Helpers & Math Compliance Wrappers
// =================================================================================

__device__ __forceinline__ void cp_async_pred_zfill(void* smem_ptr, const void* glob_ptr, bool src_valid) {
    uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    if (src_valid) {
        // 16 bytes = 128 bits = 8 halves
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"(smem_int_ptr), "l"(glob_ptr)
        );
    } else {
        int4* s = reinterpret_cast<int4*>(smem_ptr);
        *s = make_int4(0,0,0,0);
    }
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_group(int n) {
    if (n == 0) asm volatile("cp.async.wait_group 0;\n" ::);
    else if (n == 1) asm volatile("cp.async.wait_group 1;\n" ::);
}

// Constraint 4: 严禁 hexp, 必须转 FP32 __expf 再转回来
__device__ __forceinline__ half hexp_compliant(half h) {
    float f = __half2float(h);
    // Constraint 1: __expf 是数学函数，不是 FMA，允许使用
    f = __expf(f);
    return __float2half(f);
}

// Constraint 7: 严禁 FP32 div/rcp, 必须用 h2rcp (FP16 intrinsic)
__device__ __forceinline__ half2 h2rcp_compliant(half2 h) {
    // 使用 PTX 指令 rcp.approx.ftz.f16x2
    return h2rcp(h);
}

// Warp Reduce Sum (half)
__device__ __forceinline__ half warp_reduce_sum_half(half val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        // 使用位操作进行 shuffle，避免 FP32 转换
        unsigned short bits = *reinterpret_cast<unsigned short*>(&val);
        bits = __shfl_xor_sync(0xffffffff, bits, offset);
        val = __hadd(val, *reinterpret_cast<half*>(&bits));
    }
    return val;
}

// =================================================================================
// Flash Attention Kernel
// =================================================================================

__global__ void flash_attention_fp16_optimized(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    const int B, const int H, const int S, const int D,
    const float scale_float
) {
    // -------------------------------------------------------------------------
    // Shared Memory Setup (Double Buffer)
    // -------------------------------------------------------------------------
    extern __shared__ half smem[];
    half (*K_smem)[TC][SMEM_STRIDE] = reinterpret_cast<half (*)[TC][SMEM_STRIDE]>(smem);
    half (*V_smem)[TC][SMEM_STRIDE] = reinterpret_cast<half (*)[TC][SMEM_STRIDE]>(smem + 2 * TC * SMEM_STRIDE);

    // -------------------------------------------------------------------------
    // Indexing
    // -------------------------------------------------------------------------
    const int tx = threadIdx.x; // Lane ID (0-31)
    const int ty = threadIdx.y; // Query Row in Block (0-TR-1)
    
    const int batch_head_idx = blockIdx.y; 
    const int q_chunk_idx = blockIdx.x;
    
    const int batch_idx = batch_head_idx / H;
    const int head_idx  = batch_head_idx % H;
    
    const long long qkv_offset = ((long long)batch_idx * H * S * D) + ((long long)head_idx * S * D);
    const half* Q_base = Q + qkv_offset;
    const half* K_base = K + qkv_offset;
    const half* V_base = V + qkv_offset;
    half* O_base       = O + qkv_offset;

    const int q_global_row = q_chunk_idx * TR + ty;
    const bool is_valid_q = (q_global_row < S);

    // -------------------------------------------------------------------------
    // Registers
    // -------------------------------------------------------------------------
    half2 q_frag[2]; 
    half2 o_frag[2]; 
    
    // 初始化 Accumulator (FP16)
    o_frag[0] = __float2half2_rn(0.0f);
    o_frag[1] = __float2half2_rn(0.0f);

    half m_i = __float2half(-65504.0f); 
    half l_i = __float2half(0.0f);      

    // 转换 Scale 为 FP16，后续计算全在 FP16 进行，避免 FP32 FMA
    const half scale = __float2half(scale_float);
    const half2 scale2 = __half2half2(scale);

    // -------------------------------------------------------------------------
    // 1. Load Q -> Registers & Scale
    // -------------------------------------------------------------------------
    if (is_valid_q) {
        int d_idx_0 = tx * 2;
        int d_idx_1 = tx * 2 + 64;

        half2 q_val0 = *reinterpret_cast<const half2*>(&Q_base[q_global_row * D + d_idx_0]);
        // 使用 FP16 乘法，符合约束 1
        q_frag[0] = __hmul2(q_val0, scale2); 

        half2 q_val1 = *reinterpret_cast<const half2*>(&Q_base[q_global_row * D + d_idx_1]);
        q_frag[1] = __hmul2(q_val1, scale2);
    }

    // -------------------------------------------------------------------------
    // 2. Helper for Loading K/V Tiles
    // -------------------------------------------------------------------------
    auto load_tile_async = [&](int stage, int k_step) {
        int k_start = k_step * TC;
        int tid_global = ty * 32 + tx; 
        
        #pragma unroll
        for(int i=0; i<2; ++i) {
            int pack_idx = tid_global + i * 512;
            int row = pack_idx >> 4; 
            int col_pack = pack_idx & 15; 
            int col = col_pack << 3; 
            
            bool valid = (k_start + row < S);
            
            cp_async_pred_zfill(
                &K_smem[stage][row][col],
                K_base + (k_start + row) * D + col,
                valid
            );
            
            cp_async_pred_zfill(
                &V_smem[stage][row][col],
                V_base + (k_start + row) * D + col,
                valid
            );
        }
    };

    // -------------------------------------------------------------------------
    // 3. Main Loop
    // -------------------------------------------------------------------------
    int num_k_tiles = (S + TC - 1) / TC;

    load_tile_async(0, 0);
    cp_async_commit();

    for (int k_step = 0; k_step < num_k_tiles; ++k_step) {
        int cur_stage = k_step % 2;
        int next_stage = (k_step + 1) % 2;
        
        if (k_step + 1 < num_k_tiles) {
            load_tile_async(next_stage, k_step + 1);
        }
        cp_async_commit();

        cp_async_wait_group(1);
        __syncthreads();

        if (is_valid_q) {
            int k_start_curr = k_step * TC;
            int valid_k_rows = (S - k_start_curr);
            if (valid_k_rows > TC) valid_k_rows = TC;

            for (int k_sub = 0; k_sub < TC; ++k_sub) {
                if (k_sub >= valid_k_rows) break;

                // --- A. Compute Dot Product (Q * K^T) ---
                int d_idx_0 = tx * 2;
                int d_idx_1 = tx * 2 + 64;
                
                half2 k_val0 = *reinterpret_cast<half2*>(&K_smem[cur_stage][k_sub][d_idx_0]);
                half2 k_val1 = *reinterpret_cast<half2*>(&K_smem[cur_stage][k_sub][d_idx_1]);

                // Constraint 1: 使用硬件级 FP16 FMA (__hfma2)
                half2 dot2 = __hmul2(q_frag[0], k_val0);
                dot2 = __hfma2(q_frag[1], k_val1, dot2);
                
                half dot_val = __hadd(dot2.x, dot2.y);
                half score = warp_reduce_sum_half(dot_val);

                // --- B. Softmax Update ---
                
                half m_prev = m_i;
                m_i = __hmax(m_prev, score); // FP16 Max
                
                // Constraint 4: 使用 hexp_compliant (内部转FP32 __expf)
                half score_diff = __hsub(score, m_i);
                half p = hexp_compliant(score_diff); 
                
                half m_diff = __hsub(m_prev, m_i);
                half alpha = hexp_compliant(m_diff); 
                
                // FP16 FMA
                l_i = __hfma(l_i, alpha, p);

                // --- C. Update Accumulator (O) ---
                half2 p2 = __half2half2(p);
                half2 alpha2 = __half2half2(alpha);
                
                half2 v_val0 = *reinterpret_cast<half2*>(&V_smem[cur_stage][k_sub][d_idx_0]);
                half2 v_val1 = *reinterpret_cast<half2*>(&V_smem[cur_stage][k_sub][d_idx_1]);

                // Update O: O = O * alpha + P * V
                // Constraint 1: 使用 FP16 指令，无 FP32 FMA
                o_frag[0] = __hmul2(o_frag[0], alpha2);
                o_frag[0] = __hfma2(p2, v_val0, o_frag[0]);
                
                o_frag[1] = __hmul2(o_frag[1], alpha2);
                o_frag[1] = __hfma2(p2, v_val1, o_frag[1]);
            }
        }
        
        __syncthreads();
    }
    
    cp_async_wait_group(0);

    // -------------------------------------------------------------------------
    // 4. Epilogue
    // -------------------------------------------------------------------------
    if (is_valid_q) {
        // Constraint 7: 使用 h2rcp 进行向量化 FP16 倒数计算
        half2 l_vec = __half2half2(l_i);
        half2 inv_l2 = h2rcp_compliant(l_vec); 

        int d_idx_0 = tx * 2;
        int d_idx_1 = tx * 2 + 64;

        half2 res0 = __hmul2(o_frag[0], inv_l2);
        *reinterpret_cast<half2*>(&O_base[q_global_row * D + d_idx_0]) = res0;

        half2 res1 = __hmul2(o_frag[1], inv_l2);
        *reinterpret_cast<half2*>(&O_base[q_global_row * D + d_idx_1]) = res1;
    }
}

// =================================================================================
// Launcher
// =================================================================================

void launch_attention_fp16(
    const void* q, 
    const void* k, 
    const void* v, 
    void* output, 
    int B, int H, int S, int D, 
    float scale
) {
    assert(D == 128);

    const half* d_q = static_cast<const half*>(q);
    const half* d_k = static_cast<const half*>(k);
    const half* d_v = static_cast<const half*>(v);
    half* d_o       = static_cast<half*>(output);

    dim3 block(32, TR);
    dim3 grid((S + TR - 1) / TR, B * H);

    size_t smem_bytes = 2 * TC * SMEM_STRIDE * sizeof(half) * 2;

    cudaFuncSetAttribute(flash_attention_fp16_optimized, 
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 
                         smem_bytes);

    flash_attention_fp16_optimized<<<grid, block, smem_bytes>>>(
        d_q, d_k, d_v, d_o,
        B, H, S, D, scale
    );
}
//[B=8, H=8, L=1024, D=128]：9.689 ms