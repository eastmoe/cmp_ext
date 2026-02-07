#include <cuda_runtime.h>
#include <cuda_bf16.h>

// 辅助内联函数：严格遵循数学运算约束
__device__ __forceinline__ float silu_math_op(float x) {
    // 约束1 & 2: 不使用FMA，不使用标准库，使用内联指令
    // SiLU(x) = x / (1 + exp(-x))
    
    // 1. 计算 -x: 使用 __fmul_rn(x, -1.0f)
    float neg_x = __fmul_rn(x, -1.0f);
    
    // 2. 计算 exp(-x): 使用 __expf
    float exp_val = __expf(neg_x);
    
    // 3. 计算 1 + exp(-x): 使用 __fadd_rn
    float den = __fadd_rn(1.0f, exp_val);
    
    // 4. 计算除法: 使用 __fdividef (rcp约束也由此覆盖)
    float res = __fdividef(x, den);
    
    return res;
}

__global__ void silu_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 向量化优化：尝试以 128-bit (int4) 为单位加载，即一次处理 8 个 bf16
    // 假设深度学习张量通常是内存对齐的
    int vec_n = n / 8;
    const int4* input_i4 = reinterpret_cast<const int4*>(input);
    int4* output_i4 = reinterpret_cast<int4*>(output);

    // Grid-Stride Loop 处理向量化部分
    for (int i = idx; i < vec_n; i += stride) {
        int4 load_val = input_i4[i];
        int4 store_val;

        // 将 int4 视作 4 个 __nv_bfloat162 进行处理
        __nv_bfloat162* in_b2 = reinterpret_cast<__nv_bfloat162*>(&load_val);
        __nv_bfloat162* out_b2 = reinterpret_cast<__nv_bfloat162*>(&store_val);

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            __nv_bfloat162 val = in_b2[k];
            // 转换为 float2 进行计算
            float2 f2 = __bfloat1622float2(val);

            // 分别对两个通道应用优化后的数学逻辑
            f2.x = silu_math_op(f2.x);
            f2.y = silu_math_op(f2.y);

            // 转回 bf162
            out_b2[k] = __float22bfloat162_rn(f2);
        }

        output_i4[i] = store_val;
    }

    // 处理剩余的尾部元素 (非8倍数部分)
    int tail_start = vec_n * 8;
    for (int i = tail_start + idx; i < n; i += stride) {
        float x = __bfloat162float(input[i]);
        float val = silu_math_op(x);
        output[i] = __float2bfloat16(val);
    }
}

void launch_silu_bf16(const void* input, void* output, int total_elements) {
    int threads = 256;
    // 由于使用了向量化(x8)，这里计算Block数量时除以8，避免启动过多空闲线程
    // 加上 grid stride loop 逻辑，即使 total_elements 很大也能正确处理
    int vec_elements = 8;
    int items_per_thread = vec_elements; 
    int blocks = (total_elements + (threads * items_per_thread) - 1) / (threads * items_per_thread);
    
    // 限制最大 grid 大小以适应 GPU 调度，防止 grid 过大增加开销（可选优化，GA100 SM数很多）
    if (blocks > 65535) blocks = 65535; 
    if (blocks == 0) blocks = 1;

    silu_kernel_bf16<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(input),
        reinterpret_cast<__nv_bfloat16*>(output),
        total_elements
    );
}
//[8192x8192]  0.208 ms, Avg Power:  75.09 W  16.4 x
