#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// ==========================================
// 辅助函数：严格遵循指令约束
// ==========================================

// 封装 FP32 加法，强制使用 ADD 避免 FMA 融合
__device__ __forceinline__ float add_rn(float a, float b) {
    return __fadd_rn(a, b);
}

// 封装 FP32 乘法，强制使用 MUL 避免 FMA 融合
__device__ __forceinline__ float mul_rn(float a, float b) {
    return __fmul_rn(a, b);
}

// 封装 FP32 减法，强制使用 ADD + Neg 避免 FMA 融合
__device__ __forceinline__ float sub_rn(float a, float b) {
    return __fadd_rn(a, -b);
}

// Warp内的求和归约 (使用 add_rn)
__device__ __forceinline__ float warpReduceSum(float val) {
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = add_rn(val, __shfl_down_sync(mask, val, offset));
    }
    return val;
}

// Warp内的Pair求和归约 (用于同时归约 sum 和 sum_sq)
__device__ __forceinline__ float2 warpReduceSum2(float2 val) {
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val.x = add_rn(val.x, __shfl_down_sync(mask, val.x, offset));
        val.y = add_rn(val.y, __shfl_down_sync(mask, val.y, offset));
    }
    return val;
}

// ==========================================
// Kernel 实现
// ==========================================

__global__ void GroupNormKernelFP16(
    half* __restrict__ output,
    const half* __restrict__ input,
    const half* __restrict__ gamma,
    const half* __restrict__ beta,
    int N, int C, int HxW, int groups, float eps) {

    // 1. 索引计算
    int n = blockIdx.x;
    int g = blockIdx.y;
    
    int channels_per_group = C / groups;
    int num_elements_in_group = channels_per_group * HxW;
    int c_start = g * channels_per_group; 
    
    // Group 偏移
    int group_offset = n * C * HxW + c_start * HxW;
    const half* src = input + group_offset;
    half* dst = output + group_offset;

    // =========================================================
    // 第一遍：计算统计量 (Sum, SumSq)
    // 策略：使用 float2 累加，禁止 FP32 FMA，使用 128-bit 向量化加载
    // =========================================================
    float2 local_stats = make_float2(0.0f, 0.0f);

    // 重新解释指针为 int4 (128-bit, 8个half) 以利用 GA100 高带宽
    // 这里假设地址通常是对齐的。如果不对齐，可以回退到 half2，但 GA100 上通常 tensor 指针是对齐的。
    // 为了最大化兼容性和性能平衡，我们检查对齐，若不对齐则回退（此处简化为假设 vector load，
    // 但使用 unroll 的 half2 处理以确保安全且高效）。
    
    // 使用 half2 指针进行访问，每次循环处理 4 个 half2 (8 个元素)
    const half2* src_h2 = reinterpret_cast<const half2*>(src);
    int num_h2 = num_elements_in_group / 2;
    
    int idx = threadIdx.x;
    
    // 主循环：每次处理 4 * blockDim 个 half2 (即 8 * blockDim 个元素)
    // 这种展开有助于掩盖延迟
    for (; idx + 3 * blockDim.x < num_h2; idx += 4 * blockDim.x) {
        half2 v0 = src_h2[idx];
        half2 v1 = src_h2[idx + blockDim.x];
        half2 v2 = src_h2[idx + 2 * blockDim.x];
        half2 v3 = src_h2[idx + 3 * blockDim.x];

        float2 f0 = __half22float2(v0);
        float2 f1 = __half22float2(v1);
        float2 f2 = __half22float2(v2);
        float2 f3 = __half22float2(v3);

        // 累加 Sum (禁止 FMA: 使用 add_rn)
        local_stats.x = add_rn(local_stats.x, f0.x); local_stats.x = add_rn(local_stats.x, f0.y);
        local_stats.x = add_rn(local_stats.x, f1.x); local_stats.x = add_rn(local_stats.x, f1.y);
        local_stats.x = add_rn(local_stats.x, f2.x); local_stats.x = add_rn(local_stats.x, f2.y);
        local_stats.x = add_rn(local_stats.x, f3.x); local_stats.x = add_rn(local_stats.x, f3.y);

        // 累加 SumSq (禁止 FMA: 使用 mul_rn 然后 add_rn)
        local_stats.y = add_rn(local_stats.y, mul_rn(f0.x, f0.x)); local_stats.y = add_rn(local_stats.y, mul_rn(f0.y, f0.y));
        local_stats.y = add_rn(local_stats.y, mul_rn(f1.x, f1.x)); local_stats.y = add_rn(local_stats.y, mul_rn(f1.y, f1.y));
        local_stats.y = add_rn(local_stats.y, mul_rn(f2.x, f2.x)); local_stats.y = add_rn(local_stats.y, mul_rn(f2.y, f2.y));
        local_stats.y = add_rn(local_stats.y, mul_rn(f3.x, f3.x)); local_stats.y = add_rn(local_stats.y, mul_rn(f3.y, f3.y));
    }

    // 处理剩余的 half2
    for (; idx < num_h2; idx += blockDim.x) {
        half2 v = src_h2[idx];
        float2 f = __half22float2(v);
        local_stats.x = add_rn(local_stats.x, f.x);
        local_stats.x = add_rn(local_stats.x, f.y);
        local_stats.y = add_rn(local_stats.y, mul_rn(f.x, f.x));
        local_stats.y = add_rn(local_stats.y, mul_rn(f.y, f.y));
    }

    // 处理奇数尾部
    if (num_elements_in_group % 2 != 0) {
        int tail_idx = num_h2 * 2;
        // 简单的 Grid Stride 判断
        // 尾部索引是固定的，如果当前线程负责该数据的加载区域，则加载
        // 由于上面循环步长是 blockDim，这里直接让 Lane 0 处理即可，因为只有一个元素
        if (threadIdx.x == 0) {
            half val = src[tail_idx];
            float val_f = __half2float(val);
            local_stats.x = add_rn(local_stats.x, val_f);
            local_stats.y = add_rn(local_stats.y, mul_rn(val_f, val_f));
        }
    }

    // =========================================================
    // 3. 归约 (Reduction)
    // =========================================================
    local_stats = warpReduceSum2(local_stats);

    __shared__ float2 s_stats;
    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);

    if (tile.thread_rank() == 0) {
        if (threadIdx.x == 0) s_stats = make_float2(0.0f, 0.0f);
    }
    cta.sync();

    if (tile.thread_rank() == 0) {
        atomicAdd(&s_stats.x, local_stats.x);
        atomicAdd(&s_stats.y, local_stats.y);
    }
    cta.sync();

    // =========================================================
    // 4. 计算参数 (Mean, Rstd) - 遵守指令约束
    // =========================================================
    // 约束7: 禁止 __frcp_rn。这里使用标准除法 /，编译器会生成 fdiv_rn (允许)。
    // 只有在显式求倒数时才受限。
    
    float mean = s_stats.x / num_elements_in_group;
    // var = E[x^2] - E[x]^2. 禁止 FMA -> 使用 mul_rn 和 sub_rn
    float mean_sq = mul_rn(mean, mean);
    float avg_sq = s_stats.y / num_elements_in_group;
    float var = sub_rn(avg_sq, mean_sq);
    
    // rsqrtf 是允许的
    float rstd = rsqrtf(max(var, 0.0f) + eps);

    // =========================================================
    // 5. 第二遍：归一化 (利用 FP16 FMA __hfma2)
    // 公式变换： y = (x - mean) * rstd * g + b
    //           y = x * (rstd * g) + (b - mean * rstd * g)
    // 令 A = rstd * g, B = b - mean * A
    // 则 y = x * A + B  =>  __hfma2(x, A, B)
    // =========================================================
    
    half2* dst_h2 = reinterpret_cast<half2*>(dst);
    idx = threadIdx.x;

    // 同样展开循环
    for (; idx < num_h2; idx += blockDim.x) {
        half2 val_h2 = src_h2[idx];
        
        // 计算当前 half2 对应的两个元素的 Channel Index
        int elem_idx_0 = idx * 2;
        int elem_idx_1 = elem_idx_0 + 1;

        // 整数除法计算 Channel (如果 HxW 是 2 的幂，编译器会优化为位移)
        int c_local_0 = elem_idx_0 / HxW;
        int c_local_1 = elem_idx_1 / HxW; 

        // 准备系数 A (Scale) 和 B (Bias)
        float scale_f_0, scale_f_1;
        float bias_f_0, bias_f_1;

        // 处理第一个元素
        {
            int c_global = c_start + c_local_0;
            float g = (gamma != nullptr) ? __half2float(gamma[c_global]) : 1.0f;
            float b = (beta != nullptr) ? __half2float(beta[c_global]) : 0.0f;
            
            // 计算 A = rstd * g (禁止 FP32 FMA)
            scale_f_0 = mul_rn(rstd, g);
            // 计算 B = b - mean * A (禁止 FP32 FMA)
            bias_f_0 = sub_rn(b, mul_rn(mean, scale_f_0));
        }

        // 处理第二个元素 (优化：如果是同一个 Channel，复用系数)
        if (c_local_0 == c_local_1) {
            scale_f_1 = scale_f_0;
            bias_f_1 = bias_f_0;
        } else {
            int c_global = c_start + c_local_1;
            float g = (gamma != nullptr) ? __half2float(gamma[c_global]) : 1.0f;
            float b = (beta != nullptr) ? __half2float(beta[c_global]) : 0.0f;
            scale_f_1 = mul_rn(rstd, g);
            bias_f_1 = sub_rn(b, mul_rn(mean, scale_f_1));
        }

        // 将系数打包为 half2
        half2 scale_h2 = __float22half2_rn(make_float2(scale_f_0, scale_f_1));
        half2 bias_h2 = __float22half2_rn(make_float2(bias_f_0, bias_f_1));

        // 关键优化：使用 FP16 FMA (__hfma2)
        // res = val * scale + bias
        half2 res_h2 = __hfma2(val_h2, scale_h2, bias_h2);

        dst_h2[idx] = res_h2;
    }

    // 处理尾部
    if (num_elements_in_group % 2 != 0) {
        int tail_idx = num_h2 * 2;
        if (threadIdx.x == 0) {
            int c_local = tail_idx / HxW;
            int c_global = c_start + c_local;
            
            float val = __half2float(src[tail_idx]);
            float g = (gamma != nullptr) ? __half2float(gamma[c_global]) : 1.0f;
            float b = (beta != nullptr) ? __half2float(beta[c_global]) : 0.0f;
            
            // FP32 计算系数 (禁止 FMA)
            float scale = mul_rn(rstd, g);
            float bias = sub_rn(b, mul_rn(mean, scale));
            
            // 转回 FP16 使用 FMA
            half h_scale = __float2half(scale);
            half h_bias = __float2half(bias);
            half h_val = src[tail_idx];
            
            // __hfma (scalar)
            dst[tail_idx] = __hfma(h_val, h_scale, h_bias);
        }
    }
}

// Host 端入口函数，保持签名一致
void launch_groupnorm_fp16(void* output, const void* input, const void* weight, const void* bias, int N, int C, int HxW, int groups, float eps) {
    dim3 grid(N, groups);
    dim3 block(256); // 保持 256

    GroupNormKernelFP16<<<grid, block>>>(
        (half*)output, 
        (const half*)input, 
        (const half*)weight, 
        (const half*)bias, 
        N, C, HxW, groups, eps
    );
}
//[N=64, C=512, 128x128]：3.172 ms