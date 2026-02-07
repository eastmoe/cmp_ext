#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 辅助宏：强制128位（16字节）向量化读写
// 对应GA100显存总线和L1 Cache的最佳访问粒度
#define LDG_FLOAT4(ptr) (*(reinterpret_cast<const float4*>(ptr)))
#define STG_FLOAT4(ptr, val) (*(reinterpret_cast<float4*>(ptr)) = (val))

// 保持Kernel原有名称不变
__global__ void softsign_kernel_fp16(const __half* __restrict__ input, __half* __restrict__ output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // ----------------------------------------------------------------
    // 向量化路径：利用 GA100 的 FP16x2 SIMD 吞吐
    // 每次迭代处理 8 个 __half 元素 (128 bits = 1 float4)
    // ----------------------------------------------------------------
    int n_vec = n >> 3; // n / 8
    
    // 准备常量：1.0 (half2)
    const __half2 h2_one = __float2half2_rn(1.0f);

    for (int i = idx; i < n_vec; i += stride) {
        int offset = i << 3; // i * 8

        // 1. 向量化加载：一次指令加载 8 个 FP16
        float4 load_reg = LDG_FLOAT4(input + offset);
        
        // 将寄存器数据重解释为 half2 数组
        __half2* v = reinterpret_cast<__half2*>(&load_reg);

        // 2. 寄存器级循环展开 (Unrolling)
        // 这里的计算完全是在寄存器中进行，无内存依赖
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            __half2 x = v[k];

            // abs_x = |x| (使用硬件级 half2 abs)
            __half2 abs_x = __habs2(x);

            // denom = 1 + |x| (使用硬件级 half2 add)
            __half2 denom = __hadd2(h2_one, abs_x);

            // 核心优化：Softsign = x / (1+|x|)
            // 依据要求7：禁止 FP32 __frcp_rn，禁止普通除法。
            // 使用 GA100 原生指令 rcp.approx.ftz.f16x2 (h2rcp) 计算倒数
            __half2 rcp_denom = h2rcp(denom);

            // res = x * rcp_denom (使用硬件级 half2 mul)
            // 替代了除法运算
            v[k] = __hmul2(x, rcp_denom);
        }

        // 3. 向量化写回
        STG_FLOAT4(output + offset, load_reg);
    }

    // ----------------------------------------------------------------
    // 标量路径：处理剩余不足 8 个的元素 (Tail effect)
    // ----------------------------------------------------------------
    int vec_end = n_vec << 3;
    const __half h_one = __float2half(1.0f);

    for (int i = vec_end + idx; i < n; i += stride) {
        __half x = input[i];
        
        // 标量版计算
        __half abs_x = __habs(x);
        __half denom = __hadd(h_one, abs_x);
        
        // 依据要求：使用 FP16 倒数指令 hrcp (rcp.approx.f16)
        __half rcp_denom = hrcp(denom);
        
        output[i] = __hmul(x, rcp_denom);
    }
}

// 保持入口函数签名完全一致 (C++ Mangled Symbol: _Z20launch_softsign_fp16PKvPvi)
void launch_softsign_fp16(const void* input, void* output, int total_elements) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    // 调用 Kernel
    softsign_kernel_fp16<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(input),
        reinterpret_cast<__half*>(output),
        total_elements
    );
}
//[8192x8192]： 0.374 ms