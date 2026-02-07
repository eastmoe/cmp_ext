#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>  // 修复 error: identifier "uintptr_t" is undefined

// 辅助设备函数：对 half2 进行 SoftShrink 逻辑计算
// 逻辑：Result = sign(x) * max(|x| - lambda, 0)
// 优化：
// 1. 使用 SIMD 指令处理 half2
// 2. 移除 if-else 分支，使用纯数学和位运算
// 3. 避免 float 转换
__device__ __forceinline__ __half2 softshrink_h2_op(__half2 x, __half2 lam, __half2 zero) {
    // 1. 计算绝对值 |x|
    __half2 abs_x = __habs2(x);

    // 2. 减去 lambda
    __half2 diff = __hsub2(abs_x, lam);

    // 3. 取 max(diff, 0) (即 ReLU 逻辑)
    __half2 rect = __hmax2(diff, zero);

    // 4. 恢复符号位
    // 使用 union 进行类型转换以进行位操作，避免违反严格别名规则
    union { __half2 h; unsigned int u; } u_x, u_rect;
    u_x.h = x;
    u_rect.h = rect;

    // 0x80008000 掩码提取两个 half 的符号位
    // u_rect (结果) 必然为正，直接与输入 x 的符号位进行 OR 操作
    u_rect.u = (u_x.u & 0x80008000u) | u_rect.u;

    return u_rect.h;
}

__global__ void softshrink_kernel_fp16(const __half* input, __half* output, int n, float lambd) {
    // 准备常量 (广播到 half2)
    const __half h_lambd_scalar = __float2half(lambd);
    const __half2 h2_lambd = __half2half2(h_lambd_scalar);
    const __half2 h2_zero = __float2half2_rn(0.0f);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 检查指针是否 16 字节对齐，以启用 float4 (128-bit) 向量化访问
    // GA100 上对齐访问对性能至关重要
    bool is_aligned = (reinterpret_cast<uintptr_t>(input) % 16 == 0) && 
                      (reinterpret_cast<uintptr_t>(output) % 16 == 0);

    // 向量化路径：每线程处理 8 个元素 (float4)
    if (is_aligned) {
        int n_vec = n / 8; // float4 的数量
        
        // 重新解释指针为 float4*
        const float4* in_f4 = reinterpret_cast<const float4*>(input);
        float4* out_f4 = reinterpret_cast<float4*>(output);

        // 使用网格跨步循环处理向量
        for (int i = idx; i < n_vec; i += stride) {
            float4 v_in = in_f4[i];
            float4 v_out;

            // 将 float4 中的 4 个 float 分量视为 4 个 half2 进行处理
            // 手动展开以提高指令吞吐 (ILP)
            
            // 处理分量 .x (包含两个 half)
            {
                union { float f; __half2 h; } conv;
                conv.f = v_in.x;
                conv.h = softshrink_h2_op(conv.h, h2_lambd, h2_zero);
                v_out.x = conv.f;
            }
            // 处理分量 .y
            {
                union { float f; __half2 h; } conv;
                conv.f = v_in.y;
                conv.h = softshrink_h2_op(conv.h, h2_lambd, h2_zero);
                v_out.y = conv.f;
            }
            // 处理分量 .z
            {
                union { float f; __half2 h; } conv;
                conv.f = v_in.z;
                conv.h = softshrink_h2_op(conv.h, h2_lambd, h2_zero);
                v_out.z = conv.f;
            }
            // 处理分量 .w
            {
                union { float f; __half2 h; } conv;
                conv.f = v_in.w;
                conv.h = softshrink_h2_op(conv.h, h2_lambd, h2_zero);
                v_out.w = conv.f;
            }

            out_f4[i] = v_out;
        }

        // 处理尾部剩余元素 (无法组成 float4 的部分)
        // 只需要处理 [n_vec * 8, n) 范围
        int tail_start = n_vec * 8;
        
        // 标量尾部处理所需常量
        __half h_neg_lambd = __float2half(-lambd);
        __half h_zero_scalar = __float2half(0.0f);
        
        for (int i = tail_start + idx; i < n; i += stride) {
            __half x = input[i];
            // 标量部分保留原逻辑或简单逻辑
            if (__hgt(x, h_lambd_scalar)) {
                output[i] = __hsub(x, h_lambd_scalar);
            } else if (__hlt(x, h_neg_lambd)) {
                output[i] = __hadd(x, h_lambd_scalar);
            } else {
                output[i] = h_zero_scalar;
            }
        }
    } 
    else {
        // 非对齐回退路径：使用标准的标量处理
        // 即使未对齐，在 GA100 上 L1 Cache 也能较好处理
        __half h_neg_lambd = __float2half(-lambd);
        __half h_zero_scalar = __float2half(0.0f);

        for (int i = idx; i < n; i += stride) {
            __half x = input[i];
            if (__hgt(x, h_lambd_scalar)) {
                output[i] = __hsub(x, h_lambd_scalar);
            } else if (__hlt(x, h_neg_lambd)) {
                output[i] = __hadd(x, h_lambd_scalar);
            } else {
                output[i] = h_zero_scalar;
            }
        }
    }
}

void launch_softshrink_fp16(const void* input, void* output, int total_elements, float lambd) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    // 修复 error: a __global__ function call must be configured
    softshrink_kernel_fp16<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(input),
        reinterpret_cast<__half*>(output),
        total_elements, 
        lambd
    );
}