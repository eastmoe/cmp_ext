#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 预计算的常数
// A = 2 * sqrt(2/pi) = 1.5957691216
// B = A * 0.044715 = 0.071354814

__global__ void gelu_fp16_optimized_kernel(const half* __restrict__ input, half* __restrict__ output, int n) {
    // 定义 FP16 vector 常量
    const half2 kA = __float2half2_rn(1.595769122f);
    const half2 kB = __float2half2_rn(0.071354815f);
    const half2 kOne = __float2half2_rn(1.0f);

    // 定义 Scalar 常量 (用于处理尾部)
    const half kA_s = __float2half(1.595769122f);
    const half kB_s = __float2half(0.071354815f);
    const half kOne_s = __float2half(1.0f);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 向量化循环：每个线程处理 8 个元素 (float4 = 4 * half2)
    // 使用 stride * 8 作为步长
    for (int i = idx * 8; i < n; i += stride * 8) {
        
        // 1. 向量化路径：如果剩余元素足够8个
        if (i + 8 <= n) {
            // Load 128-bit (8 halves)
            float4 v_load = *reinterpret_cast<const float4*>(&input[i]);
            
            // 将 float4 重新解释为 4 个 half2
            half2* h2_data = reinterpret_cast<half2*>(&v_load);
            half2 res[4];

            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                half2 x = h2_data[k];

                // 计算多项式 z = 2 * sqrt(2/pi) * (x + 0.044715 * x^3)
                // 变形为: z = x * (kA + kB * x^2)
                
                half2 x2 = __hmul2(x, x);
                // 使用 FP16 FMA: kB * x2 + kA
                half2 poly = __hfma2(kB, x2, kA); 
                half2 z = __hmul2(x, poly);

                // 我们需要计算 Sigmoid(z) = 1 / (1 + exp(-z))
                // 取负号: -z
                half2 neg_z = __hneg2(z);

                // Constraint 4: 必须转换为 FP32 使用 __expf
                float2 f_z = __half22float2(neg_z);
                
                // 对两个分量分别计算 expf
                f_z.x = __expf(f_z.x);
                f_z.y = __expf(f_z.y);

                // 转换回 FP16
                half2 exp_val = __float22half2_rn(f_z);

                // den = 1.0 + exp_val
                half2 den = __hadd2(kOne, exp_val);

                // Constraint 7: 必须使用 FP16 h2rcp，不能用 FP32 倒数
                half2 inv_den = h2rcp(den);

                // result = x * inv_den
                res[k] = __hmul2(x, inv_den);
            }

            // Store 128-bit
            *reinterpret_cast<float4*>(&output[i]) = *reinterpret_cast<float4*>(res);

        } else {
            // 2. 标量路径：处理剩余不足8个的元素
            // 为了避免复杂的剩余逻辑，这里直接在剩余范围内循环
            for (int j = i; j < n; ++j) {
                half x = input[j];
                
                // Scalar math mirroring the vector logic
                half x2 = __hmul(x, x);
                half poly = __hfma(kB_s, x2, kA_s);
                half z = __hmul(x, poly);
                
                half neg_z = __hneg(z);
                
                // Constriant 4: Convert to FP32 for exp
                float f_z = __half2float(neg_z);
                float f_exp = __expf(f_z);
                half h_exp = __float2half(f_exp);
                
                half den = __hadd(kOne_s, h_exp);
                
                // Constraint 7: Use FP16 rcp
                half inv_den = hrcp(den);
                
                output[j] = __hmul(x, inv_den);
            }
        }
    }
}

void launch_gelu_fp16(const void* input, void* output, int total_elements) {
    // 针对 GA100 优化 Block 大小
    int threads = 256;
    // 每个线程处理 8 个元素
    int elements_per_block = threads * 8;
    int blocks = (total_elements + elements_per_block - 1) / elements_per_block;
    
    // 限制 Grid 大小以避免不必要的启动开销，但 GA100 很大，通常这就够了
    gelu_fp16_optimized_kernel<<<blocks, threads>>>(
        reinterpret_cast<const half*>(input),
        reinterpret_cast<half*>(output),
        total_elements
    );
}