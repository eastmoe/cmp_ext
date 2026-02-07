#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

// 辅助函数：使用位操作计算float绝对值
__device__ __forceinline__ float bitwise_abs(float x) {
    // 0x7fffffff 掩码用于清除符号位 (IEEE 754 float32)
    return __int_as_float(__float_as_int(x) & 0x7fffffff);
}

__global__ void softsign_kernel_bf16(const __nv_bfloat16* __restrict__ input, __nv_bfloat16* __restrict__ output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // --- 向量化处理部分 (128-bit / 8 elements per iteration) ---
    // GA100 L2优化：使用 float4 (16 bytes) 向量化加载，最大化内存带宽利用率
    int vec_loop_limit = n >> 3; // n / 8
    
    // 强制转换为float4进行128位加载/存储
    const float4* in_vec = reinterpret_cast<const float4*>(input);
    float4* out_vec = reinterpret_cast<float4*>(output);

    for (int i = idx; i < vec_loop_limit; i += stride) {
        float4 loaded_data = in_vec[i];
        float4 result_data;
        
        // 将float4 (128bit) 视为 4个 __nv_bfloat162 (32bit)
        // 使用指针强转避免非对齐访问风险（在寄存器层面处理）
        __nv_bfloat162* v_in = reinterpret_cast<__nv_bfloat162*>(&loaded_data);
        __nv_bfloat162* v_out = reinterpret_cast<__nv_bfloat162*>(&result_data);

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            __nv_bfloat162 val_bf2 = v_in[k];
            
            // 转换为float2进行计算 (FP32 Compute)
            float2 val_f2 = __bfloat1622float2(val_bf2);

            // 逻辑: x / (1.0f + abs(x))
            // 约束: 无FP32 FMA，无标准库，使用位操作求绝对值
            
            // --- 处理 float2.x ---
            float x1 = val_f2.x;
            float abs_x1 = bitwise_abs(x1);             // Bitwise Abs
            float denom1 = __fadd_rn(1.0f, abs_x1);     // Explicit Add (No FMA)
            val_f2.x = __fdividef(x1, denom1);          // Intrinsic divide
            
            // --- 处理 float2.y ---
            float x2 = val_f2.y;
            float abs_x2 = bitwise_abs(x2);             // Bitwise Abs
            float denom2 = __fadd_rn(1.0f, abs_x2);     // Explicit Add (No FMA)
            val_f2.y = __fdividef(x2, denom2);          // Intrinsic divide

            // 转回 bfloat162 并存入寄存器数组
            v_out[k] = __float22bfloat162_rn(val_f2);
        }
        
        // 128位写回显存
        out_vec[i] = result_data;
    }

    // --- 标量尾部处理 (Scalar Tail) ---
    // 处理剩余不足8个的元素
    int tail_start = vec_loop_limit << 3; // vec_loop_limit * 8
    for (int i = tail_start + idx; i < n; i += stride) {
        // Load and convert
        float x = __bfloat162float(input[i]);
        
        // Compute (Constraints applied)
        float abs_x = bitwise_abs(x);           // Bitwise Abs
        float denom = __fadd_rn(1.0f, abs_x);   // Explicit Add
        float res = __fdividef(x, denom);       // Intrinsic divide
        
        // Store
        output[i] = __float2bfloat16(res);
    }
}

void launch_softsign_bf16(const void* input, void* output, int total_elements) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    softsign_kernel_bf16<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(input),
        reinterpret_cast<__nv_bfloat16*>(output),
        total_elements
    );
}
//Op: Softsign [8192x8192]
//  Torch         :    2.270 ms, Avg Power:  86.24 W
//  Custom        :    0.374 ms, Avg Power: 108.41 W
//  Speedup       :      6.1 x
