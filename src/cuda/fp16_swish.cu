#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 辅助函数：计算 half2 向量的 Swish
// y = x / (1 + exp(-beta * x))
// 严格遵守禁用 FP16 exp 和禁用 FP32 rcp 的规定
__device__ __forceinline__ half2 swish_h2_opt(half2 x, half2 beta_2) {
    // 1.0 in half2 (常量)
    // __float2half2_rn 将一个 float 广播到两个 half
    const half2 one = __float2half2_rn(1.0f);
    
    // 1. 计算 -beta * x (全程 FP16 硬件指令)
    half2 bx = __hmul2(beta_2, x);
    half2 neg_bx = __hneg2(bx);
    
    // 2. 计算 Exp
    // 约束4: 不使用 FP16 hexp/h2exp，必须转换为 FP32 使用 __expf
    float2 bx_f2;
    // 将 half2 解包为两个 float
    bx_f2.x = __low2float(neg_bx);
    bx_f2.y = __high2float(neg_bx);
    
    // 使用 FP32 计算 exp
    // 约束1: 这里不涉及 FMA，只单纯调用数学函数
    bx_f2.x = __expf(bx_f2.x);
    bx_f2.y = __expf(bx_f2.y);
    
    // 将结果打包回 half2
    // 修复点：__float22half2_rn 接受一个 float2 参数，而不是两个 float
    half2 exp_val = __float22half2_rn(bx_f2);
    
    // 3. 计算分母: 1 + exp(...)
    // 使用 FP16 加法 (Ampere 上吞吐量极高)
    half2 denom = __hadd2(one, exp_val);
    
    // 4. 计算倒数和最终结果
    // 约束7: 不使用 FP32 __frcp_rn，使用向量化 FP16 h2rcp
    // 逻辑：y = x * (1 / denom)
    half2 rcp_denom = h2rcp(denom);
    
    // FP16 乘法
    return __hmul2(x, rcp_denom);
}

__global__ void swish_kernel_fp16_optimized(const half* __restrict__ input, 
                                            const half* __restrict__ beta, 
                                            half* __restrict__ output, 
                                            int n) {
    // 读取一次 beta 到寄存器 (只读缓存 __ldg)
    half b_scalar = __ldg(beta);
    // 广播到 half2 向量
    half2 b_vec = __half2half2(b_scalar);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // ---------------------------------------------------
    // 向量化部分：每个线程一次处理 8 个 half (4个 half2)
    // 利用 128-bit Load/Store 优化显存带宽
    // ---------------------------------------------------
    int vec_n = n / 8;
    
    // 强制转换为 float4 指针以便编译器生成 128-bit LDG.E.128 指令
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    float4* output_vec = reinterpret_cast<float4*>(output);

    for (int i = idx; i < vec_n; i += stride) {
        float4 load_val = input_vec[i];
        float4 store_val;

        // 将 float4 (16 bytes) 寄存器空间重新解释为 4 个 half2
        // 这种转换在寄存器层面是零开销的
        half2* h2_in = reinterpret_cast<half2*>(&load_val);
        half2* h2_out = reinterpret_cast<half2*>(&store_val);

        // 手动展开计算 4 个向量，隐藏指令延迟
        h2_out[0] = swish_h2_opt(h2_in[0], b_vec);
        h2_out[1] = swish_h2_opt(h2_in[1], b_vec);
        h2_out[2] = swish_h2_opt(h2_in[2], b_vec);
        h2_out[3] = swish_h2_opt(h2_in[3], b_vec);

        // 写入结果
        output_vec[i] = store_val;
    }

    // ---------------------------------------------------
    // 尾部处理 (Tail Handling)
    // ---------------------------------------------------
    // 处理剩余无法被 8 整除的元素
    int remainder_start = vec_n * 8;
    for (int i = remainder_start + idx; i < n; i += stride) {
        half x = input[i];
        
        // 标量计算
        half bx = __hmul(b_scalar, x);
        half neg_bx = __hneg(bx);
        
        // 约束4: 即使是尾部标量，也不能用 hexp，必须转 FP32
        float bx_f = __half2float(neg_bx);
        bx_f = __expf(bx_f);
        half exp_val = __float2half(bx_f);
        
        half one = __float2half(1.0f);
        half denom = __hadd(one, exp_val);
        
        // 约束7: 不使用 FP32 倒数，也不使用 __hdiv (通常很慢)
        // 使用 hrcp (h2rcp 的标量版) + hmul
        half rcp_denom = hrcp(denom);
        
        output[i] = __hmul(x, rcp_denom); 
    }
}

// Host 调用端保持逻辑一致
void launch_swish_fp16(const void* input, const void* beta, void* output, int total_elements) {
    int threads = 256;
    int items_per_thread = 8; // 对应 Kernel 中的 float4 * 2 (half 为 2字节, float4 为 16字节 -> 8个half)
    
    int total_threads = (total_elements + items_per_thread - 1) / items_per_thread;
    int blocks = (total_threads + threads - 1) / threads;
    
    // 限制最大 Grid，适配 GA100
    if (blocks > 32768) blocks = 32768; 

    swish_kernel_fp16_optimized<<<blocks, threads>>>(
        (const half*)input, 
        (const half*)beta, 
        (half*)output, 
        total_elements
    );
}
//[8192x8192] (beta=10.0)：0.213ms 