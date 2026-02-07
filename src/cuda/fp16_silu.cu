#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 辅助函数：计算 half2 的 silu
// 约束遵循：
// 1. 不使用 h2exp，转换为 float 后使用 __expf
// 2. 不使用 FP32 FMA
// 3. 使用 h2rcp
__device__ __forceinline__ __half2 silu_h2_opt(const __half2 x) {
    // 1. 在 FP16 域计算 -x (A100 FP16吞吐量高)
    __half2 neg_x = __hneg2(x);
    
    // 2. [约束4] 必须转换为 FP32 使用 __expf
    float2 f2_in = __half22float2(neg_x);
    
    // 分别计算两个分量的 exp
    f2_in.x = __expf(f2_in.x);
    f2_in.y = __expf(f2_in.y);
    
    // 转换回 half2
    __half2 e_neg_x = __float22half2_rn(f2_in);
    
    // 3. 计算 1 + exp(-x)
    // 使用 FP16 加法，避免 FP32 FMA 风险且利用双倍吞吐
    const __half2 one = __float2half2_rn(1.0f);
    __half2 den = __hadd2(one, e_neg_x);
    
    // 4. [约束7] 计算 1 / (1 + exp(-x)) -> sigmoid(x)
    // 必须使用向量化的 h2rcp
    __half2 sig = h2rcp(den);
    
    // 5. 结果 = x * sigmoid(x)
    // 使用 FP16 乘法，不涉及 FP32 FMA
    return __hmul2(x, sig);
}

// 辅助函数：计算标量 half 的 silu (用于处理边界)
__device__ __forceinline__ __half silu_h1_opt(const __half x) {
    // FP16 负号
    __half neg_x = __hneg(x);
    
    // [约束4] 转 FP32 计算 exp
    float f_in = __half2float(neg_x);
    float f_exp = __expf(f_in);
    __half e_neg_x = __float2half_rn(f_exp);
    
    // FP16 加法
    const __half one = __float2half_rn(1.0f);
    __half den = __hadd(one, e_neg_x);
    
    // [约束7] 使用 hrcp
    __half sig = hrcp(den);
    
    return __hmul(x, sig);
}

__global__ void silu_kernel_fp16_optimized(const __half* input, __half* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // ----------------------------------------------------------------
    // 1. 向量化处理 (Vectorized Path)
    // 使用 float4 (16字节 = 8个half) 进行内存访问
    // ----------------------------------------------------------------
    
    // 保证只处理 8 的整数倍部分
    int vec_loop_limit = n & ~7; // 等价于 n - (n % 8)
    
    const float4* inp_f4 = reinterpret_cast<const float4*>(input);
    float4* out_f4 = reinterpret_cast<float4*>(output);
    
    // 计算 float4 类型的总个数
    int num_vec_elements = vec_loop_limit >> 3; // 除以 8

    // Grid-Stride Loop
    for (int i = idx; i < num_vec_elements; i += stride) {
        float4 in_val = inp_f4[i];
        float4 out_val;
        
        // 将 float4 视为 4 个 half2 进行处理
        // 使用 reinterpret_cast 将 float 地址强转为 half2 指针是 CUDA 中常见的寄存器别名操作
        
        // 展开计算以隐藏流水线延迟
        __half2 res_0 = silu_h2_opt(*reinterpret_cast<__half2*>(&in_val.x));
        __half2 res_1 = silu_h2_opt(*reinterpret_cast<__half2*>(&in_val.y));
        __half2 res_2 = silu_h2_opt(*reinterpret_cast<__half2*>(&in_val.z));
        __half2 res_3 = silu_h2_opt(*reinterpret_cast<__half2*>(&in_val.w));
        
        // 存回寄存器
        *reinterpret_cast<__half2*>(&out_val.x) = res_0;
        *reinterpret_cast<__half2*>(&out_val.y) = res_1;
        *reinterpret_cast<__half2*>(&out_val.z) = res_2;
        *reinterpret_cast<__half2*>(&out_val.w) = res_3;
        
        // 写回显存 (128-bit store)
        out_f4[i] = out_val;
    }

    // ----------------------------------------------------------------
    // 2. 标量处理剩余部分 (Remainder Path)
    // ----------------------------------------------------------------
    int remainder_start = num_vec_elements << 3; // * 8
    for (int i = remainder_start + idx; i < n; i += stride) {
        output[i] = silu_h1_opt(input[i]);
    }
}

void launch_silu_fp16(const void* input, void* output, int total_elements) {
    // GA100 每个 SM 调度能力强，256 或 512 线程块是典型配置
    int threads = 256;
    
    // 每个线程处理 8 个元素 (float4 * 2个half/float = 8 half)
    // 计算需要的 float4 块数量
    int vec_elements = (total_elements + 7) / 8;
    int blocks = (vec_elements + threads - 1) / threads;
    
    // 限制最大 Grid
    if (blocks > 65535) blocks = 65535; 

    silu_kernel_fp16_optimized<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(input),
        reinterpret_cast<__half*>(output),
        total_elements
    );
}
//8192x8192]：0.209ms