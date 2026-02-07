#include <cuda_runtime.h>
#include <cuda_bf16.h>

// 辅助函数：执行标量 Swish 计算，满足所有指令约束
// 1. 不使用 FP32 FMA (拆分为 mul_rn + add_rn)
// 2. 使用内联数学函数 (__expf, __fdividef)
__device__ __forceinline__ float compute_swish_scalar_opt(float x, float neg_beta) {
    // 逻辑: x / (1 + exp(-beta * x))
    // 变换: x / (1 + exp(neg_beta * x))
    
    // 约束1 & 2: 显式乘法 (Round-to-nearest)
    float exp_arg = __fmul_rn(neg_beta, x);

    // 约束2: 使用内联 exp
    float e_val = __expf(exp_arg);

    // 约束1: 显式加法 (Round-to-nearest)
    float denom = __fadd_rn(1.0f, e_val);

    // 约束2 & 4: 使用内联除法实现 rcp/div
    float res = __fdividef(x, denom);

    return res;
}

__global__ void swish_kernel_bf16(const __nv_bfloat16* __restrict__ input, 
                                  const __nv_bfloat16* __restrict__ beta, 
                                  __nv_bfloat16* __restrict__ output, 
                                  int n) {
    // 加载 Beta 并转为 float
    float b = __bfloat162float(*beta);
    
    // 预计算 -beta，使用 __fmul_rn 满足约束，避免循环内重复计算
    float neg_b = __fmul_rn(b, -1.0f);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // -----------------------------------------------------------
    // 向量化路径: 使用 float4 (128-bit) 一次处理 8 个 bfloat16
    // GA100 上 128-bit 加载能最大化 L2/DRAM 带宽利用率
    // -----------------------------------------------------------
    int vec_loop_limit = n / 8;
    
    // 强制转换为 float4 指针进行向量加载
    const float4* in_vec_ptr = reinterpret_cast<const float4*>(input);
    float4* out_vec_ptr = reinterpret_cast<float4*>(output);

    for (int i = idx; i < vec_loop_limit; i += stride) {
        float4 v_in = in_vec_ptr[i];
        float4 v_out;

        // 将 float4 重新解释为 8 个 bf16
        __nv_bfloat16* raw_in = reinterpret_cast<__nv_bfloat16*>(&v_in);
        __nv_bfloat16* raw_out = reinterpret_cast<__nv_bfloat16*>(&v_out);

        // 展开循环进行计算
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float x = __bfloat162float(raw_in[k]);
            float val = compute_swish_scalar_opt(x, neg_b);
            raw_out[k] = __float2bfloat16(val);
        }

        // 向量化存储
        out_vec_ptr[i] = v_out;
    }

    // -----------------------------------------------------------
    // 标量尾部处理: 处理剩余不足 8 个的元素
    // -----------------------------------------------------------
    int tail_start = vec_loop_limit * 8;
    for (int i = tail_start + idx; i < n; i += stride) {
        float x = __bfloat162float(input[i]);
        float val = compute_swish_scalar_opt(x, neg_b);
        output[i] = __float2bfloat16(val);
    }
}

void launch_swish_bf16(const void* input, const void* beta, void* output, int total_elements) {
    int threads = 256;
    
    // GA100 (A100) 有 108 个 SM。
    // 为了极致优化，我们不需要为每个元素都开启一个线程，而是利用网格跨步循环。
    // 设定 Grid Size 足够填满 GPU 即可 (例如 4 * SM_COUNT)，避免过多 Block 的调度开销。
    // 这里每个线程向量化处理 8 个元素。
    
    int num_sms = 108; 
    int optimal_blocks = num_sms * 4; 
    
    // 计算需要的理论 Blocks 数量 (除以8是因为向量化)
    int needed_blocks = (total_elements + (threads * 8) - 1) / (threads * 8);
    
    // 限制最大 Blocks 数量，利用 Loop 处理大数据
    int blocks = (needed_blocks < optimal_blocks) ? needed_blocks : optimal_blocks;
    if (blocks < 1) blocks = 1;

    swish_kernel_bf16<<<blocks, threads>>>(
        (const __nv_bfloat16*)input, 
        (const __nv_bfloat16*)beta, 
        (__nv_bfloat16*)output, 
        total_elements
    );
}
//Op: Swish [8192x8192] (beta=10.0)
//  Torch         :    2.912 ms, Avg Power:  95.00 W
//  Custom        :    0.230 ms, Avg Power: 100.21 W
//  Speedup       :     12.6 x
