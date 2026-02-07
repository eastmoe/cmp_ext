#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h> 
#include <algorithm>
#include <cstdint> 

#define MAX_DIM 128 

// 辅助：向量化读写
__device__ __forceinline__ float4 load_float4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

// 【关键修复】
// 满足要求7：通过FP16计算倒数
// 修复说明：
// 之前的内联汇编 rcp.approx.f16 在 sm_80 架构下导致 ptxas 报错 "Unexpected instruction types"。
// 现改为使用 CUDA 标准内置函数 __hdiv 计算 (1.0h / x_h)。
// 1. 仍然完全在 FP16 域内执行倒数计算，满足精度和逻辑要求。
// 2. 避免了手写 PTX 带来的寄存器类型兼容性问题。
__device__ __forceinline__ float fp32_rcp_via_fp16(float x) {
    // 1. Float -> Half (使用 _rn 版本确保明确)
    __half h_x = __float2half_rn(x);
    
    // 2. 在 FP16 精度下计算倒数 (1.0 / x)
    // __hdiv 是 device intrinsic，确保生成半精度除法指令
    __half h_one = __float2half_rn(1.0f);
    __half h_res = __hdiv(h_one, h_x);
    
    // 3. Half -> Float
    return __half2float(h_res);
}

__global__ void attention_fp32_strict_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Output,
    int S, int D, float scale) 
{
    // TR: BlockDim.x (Query 分块), TC: Key/Value 分块
    const int TR = blockDim.x; 
    const int TC = 32; 

    int tx = threadIdx.x;
    int bx = blockIdx.x; 
    int by = blockIdx.y; 

    // 全局 Query 索引
    int q_global_idx = bx * TR + tx;

    // 指针偏移
    int batch_head_offset = by * S * D;
    const float* q_ptr = Q + batch_head_offset;
    const float* k_ptr = K + batch_head_offset;
    const float* v_ptr = V + batch_head_offset;
    float* out_ptr = Output + batch_head_offset;

    // Shared Memory: 存储 K 和 V 的 Tile [TC, D]
    extern __shared__ float smem[];
    float* s_k = smem;              // [TC, D]
    float* s_v = smem + TC * D;     // [TC, D]

    float r_q[MAX_DIM];     
    float r_o[MAX_DIM];     
    
    float m_i = -1e20f; // max val
    float l_i = 0.0f;   // sum exp

    // 1. 加载 Query 到寄存器 & 初始化 Accumulator
    bool is_valid_q = (q_global_idx < S);
    if (is_valid_q) {
        for (int i = 0; i < D; i+=4) {
            if (i + 3 < D) {
                float4 qv = load_float4(&q_ptr[q_global_idx * D + i]);
                r_q[i] = qv.x; r_q[i+1] = qv.y; r_q[i+2] = qv.z; r_q[i+3] = qv.w;
            } else {
                for (int k=i; k<D; ++k) r_q[k] = q_ptr[q_global_idx * D + k];
            }
        }
        for (int i = 0; i < D; ++i) r_o[i] = 0.0f;
    } else {
        for (int i = 0; i < D; ++i) { r_q[i] = 0.0f; r_o[i] = 0.0f; }
    }

    // 2. 循环遍历 K, V Tiles
    int num_tiles = (S + TC - 1) / TC;

    for (int t = 0; t < num_tiles; ++t) {
        int t_start = t * TC;
        int valid_tc = min(TC, S - t_start);

        // 2.1 协作加载 K, V 到 Shared Memory
        int total_elems = valid_tc * D; 
        
        for (int i = tx * 4; i < total_elems; i += TR * 4) {
            int row = i / D;
            int col = i % D;
            int global_row = t_start + row;

            // Load K
            float4 kv = make_float4(0.f,0.f,0.f,0.f);
            if (col + 3 < D) {
                kv = load_float4(&k_ptr[global_row * D + col]);
                *reinterpret_cast<float4*>(&s_k[row * D + col]) = kv;
            } else {
                for(int k=0; k<4 && col+k<D; ++k) 
                    s_k[row * D + col + k] = k_ptr[global_row * D + col + k];
            }

            // Load V
            float4 vv = make_float4(0.f,0.f,0.f,0.f);
            if (col + 3 < D) {
                vv = load_float4(&v_ptr[global_row * D + col]);
                *reinterpret_cast<float4*>(&s_v[row * D + col]) = vv;
            } else {
                for(int k=0; k<4 && col+k<D; ++k) 
                    s_v[row * D + col + k] = v_ptr[global_row * D + col + k];
            }
        }
        
        __syncthreads();

        // 2.2 Flash Attention Computation
        if (is_valid_q) {
            for (int j = 0; j < valid_tc; ++j) {
                // Dot Product Q * K_j
                // 【要求1】禁止 FMA，拆分为 __fmul_rn 和 __fadd_rn
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) {
                    float prod = __fmul_rn(r_q[d], s_k[j * D + d]);
                    dot = __fadd_rn(dot, prod);
                }
                
                // Scale
                dot = __fmul_rn(dot, scale);

                // Online Softmax Logic
                // m_new = fmaxf(m_i, dot);
                float m_new = fmaxf(m_i, dot);
                
                // exp_diff = expf(m_i - m_new);
                // 【要求1】减法使用 __fsub_rn
                // 【要求4】使用 __expf (FP32)
                float diff_m = __fsub_rn(m_i, m_new);
                float exp_diff = __expf(diff_m);

                // exp_val = expf(dot - m_new);
                float diff_d = __fsub_rn(dot, m_new);
                float exp_val = __expf(diff_d);

                // l_new = l_i * exp_diff + exp_val;
                // 【要求1】禁止 FMA
                float l_term = __fmul_rn(l_i, exp_diff);
                float l_new = __fadd_rn(l_term, exp_val);

                // 更新 Output accumulator
                for (int d = 0; d < D; ++d) {
                    // r_o[d] = r_o[d] * exp_diff + s_v[j * D + d] * exp_val;
                    // 【要求1】禁止 FMA
                    float term_o = __fmul_rn(r_o[d], exp_diff);
                    float term_v = __fmul_rn(s_v[j * D + d], exp_val);
                    r_o[d] = __fadd_rn(term_o, term_v);
                }

                l_i = l_new;
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    // 3. 最终写回
    if (is_valid_q) {
        // float inv_l = 1.0f / (l_i + 1e-6f);
        // 【要求7】禁止 FP32 div/rcp，使用 helper (转FP16->rcp->转FP32)
        // 加上 epsilon 防止除零 (FP32 加法)
        float l_sum = __fadd_rn(l_i, 1e-6f);
        float inv_l = fp32_rcp_via_fp16(l_sum);

        for (int i = 0; i < D; i+=4) {
             if (i + 3 < D) {
                 float4 out_val;
                 // 【要求1】乘法使用 __fmul_rn
                 out_val.x = __fmul_rn(r_o[i], inv_l);
                 out_val.y = __fmul_rn(r_o[i+1], inv_l);
                 out_val.z = __fmul_rn(r_o[i+2], inv_l);
                 out_val.w = __fmul_rn(r_o[i+3], inv_l);
                 *reinterpret_cast<float4*>(&out_ptr[q_global_idx * D + i]) = out_val;
             } else {
                 for (int k=i; k<D; ++k) {
                     out_ptr[q_global_idx * D + k] = __fmul_rn(r_o[k], inv_l);
                 }
             }
        }
    }
}

void launch_attention_fp32(const float* input, const float* weight, const float* value, float* output, int B, int H, int S, int D, float scale) {
    if (D > MAX_DIM) {
        printf("Dim > %d not supported\n", MAX_DIM);
        return;
    }

    int TR = 64; 
    int TC = 32; 

    dim3 block(TR);
    dim3 grid((S + TR - 1) / TR, B * H);

    size_t smem_size = 2 * TC * D * sizeof(float);

    cudaFuncSetAttribute(attention_fp32_strict_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);

    attention_fp32_strict_kernel<<<grid, block, smem_size>>>(input, weight, value, output, S, D, scale);
}