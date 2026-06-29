#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void swish_kernel_fp16_optimized(const half* __restrict__ input, 
                                            const half* __restrict__ beta, 
                                            half* __restrict__ output, 
                                            int n) {
    // 读取一次 beta 到寄存器 (只读缓存 __ldg)
    half b_scalar = __ldg(beta);
    float b_float = __half2float(b_scalar);

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
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 xf = __half22float2(h2_in[k]);
            float2 res;

            float t_x = __fmul_rn(__fmul_rn(b_float, xf.x), -1.0f);
            float t_y = __fmul_rn(__fmul_rn(b_float, xf.y), -1.0f);
            float d_x = __fadd_rn(1.0f, __expf(t_x));
            float d_y = __fadd_rn(1.0f, __expf(t_y));

            res.x = __fmul_rn(xf.x, rsqrtf(__fmul_rn(d_x, d_x)));
            res.y = __fmul_rn(xf.y, rsqrtf(__fmul_rn(d_y, d_y)));
            h2_out[k] = __float22half2_rn(res);
        }

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
        
        float xf = __half2float(x);
        float t = __fmul_rn(__fmul_rn(b_float, xf), -1.0f);
        float denom = __fadd_rn(1.0f, __expf(t));
        output[i] = __float2half(__fmul_rn(xf, rsqrtf(__fmul_rn(denom, denom))));
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
