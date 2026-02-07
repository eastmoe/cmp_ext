#include <cuda_runtime.h>
#include <cuda_fp16.h> // 必须引入，用于 half2 和 h2rcp 指令
#include <cmath>
#include <cstdint>     // 修复：包含 uintptr_t

// 辅助函数：计算单个 float4 向量的 SiLU
// 优化策略：
// 1. 使用 __expf (FP32)
// 2. 避免 FP32 FMA，使用 __fadd_rn / __fmul_rn
// 3. 倒数计算使用 FP16 向量化 h2rcp
__device__ __forceinline__ float4 silu_vec4(float4 v) {
    float4 res;

    // --- 步骤 1: 计算分母 (1 + exp(-x))，保持 FP32 ---
    // 约束 4: 使用 __expf
    // 约束 1: 使用 __fadd_rn 避免 FMA
    // 注意：一元负号 -v.x 通常编译为简单的符号位翻转，不需要 intrinsic
    float d_x = __fadd_rn(1.0f, __expf(-v.x));
    float d_y = __fadd_rn(1.0f, __expf(-v.y));
    float d_z = __fadd_rn(1.0f, __expf(-v.z));
    float d_w = __fadd_rn(1.0f, __expf(-v.w));

    // --- 步骤 2: 计算倒数 (1 / d)，利用 vectorized FP16 ---
    // 约束 7: 禁止 FP32 __frcp_rn，必须转为 FP16 使用 h2rcp
    
    // 将 4 个 FP32 分母打包为 2 个 FP16x2 (half2)
    __half2 h2_0 = __float22half2_rn(make_float2(d_x, d_y));
    __half2 h2_1 = __float22half2_rn(make_float2(d_z, d_w));

    // 使用硬件级 FP16 向量倒数指令
    h2_0 = h2rcp(h2_0);
    h2_1 = h2rcp(h2_1);

    // 将结果解包回 FP32
    float2 r_0 = __half22float2(h2_0);
    float2 r_1 = __half22float2(h2_1);

    // --- 步骤 3: 最终乘法 (x * rcp)，保持 FP32 ---
    // 约束 1: 使用 __fmul_rn 避免 FMA
    res.x = __fmul_rn(v.x, r_0.x);
    res.y = __fmul_rn(v.y, r_0.y);
    res.z = __fmul_rn(v.z, r_1.x);
    res.w = __fmul_rn(v.w, r_1.y);

    return res;
}

// 辅助函数：计算标量 SiLU
__device__ __forceinline__ float silu_scalar(float x) {
    // 约束 4 & 1: FP32 exp 和 add
    float denom = __fadd_rn(1.0f, __expf(-x));

    // 约束 7: 即使是标量，也按要求转换为 FP16 使用 h2rcp
    // 构造 half2，第二个元素填充 1.0 或重复 denom 均可，这里重复 denom 以利用 SIMD
    __half2 h_denom = __float22half2_rn(make_float2(denom, denom));
    
    // 计算向量倒数
    h_denom = h2rcp(h_denom);
    
    // 取回低位 float
    float rcp_val = __low2float(h_denom);

    // 约束 1: FP32 mul
    return __fmul_rn(x, rcp_val);
}

__global__ void silu_kernel_fp32(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 检查指针是否对齐到 16 字节
    bool is_aligned = (reinterpret_cast<std::uintptr_t>(input) % 16 == 0) && 
                      (reinterpret_cast<std::uintptr_t>(output) % 16 == 0);

    if (is_aligned) {
        // --- 向量化路径 ---
        int n_vec = n >> 2; // n / 4
        
        const float4* in_vec = reinterpret_cast<const float4*>(input);
        float4* out_vec = reinterpret_cast<float4*>(output);

        for (int i = idx; i < n_vec; i += stride) {
            out_vec[i] = silu_vec4(in_vec[i]);
        }

        // 处理尾部
        int tail_start = n_vec << 2; 
        for (int i = tail_start + idx; i < n; i += stride) {
            output[i] = silu_scalar(input[i]);
        }
    } else {
        // --- 标量路径 ---
        for (int i = idx; i < n; i += stride) {
            output[i] = silu_scalar(input[i]);
        }
    }
}

void launch_silu_fp32(const float* input, float* output, int total_elements) {
    int threads = 256;
    // 向上取整计算 blocks
    int n_vec = (total_elements + 3) / 4;
    int blocks = (n_vec + threads - 1) / threads;
    
    silu_kernel_fp32<<<blocks, threads>>>(input, output, total_elements);
}
//[8192x8192]：0.412 ms