#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 预计算的常数
// A = 2 * sqrt(2/pi) = 1.5957691216
// B = A * 0.044715 = 0.071354814

__global__ void gelu_fp16_optimized_kernel(const half* __restrict__ input, half* __restrict__ output, int n) {
    const float kA = 1.595769122f;
    const float kB = 0.071354815f;

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

                float2 xf = __half22float2(x);
                float2 out;

                float x2_l = __fmul_rn(xf.x, xf.x);
                float poly_l = __fadd_rn(kA, __fmul_rn(kB, x2_l));
                float z_l = __fmul_rn(xf.x, poly_l);
                float den_l = __fadd_rn(1.0f, __expf(__fmul_rn(z_l, -1.0f)));
                out.x = __fmul_rn(xf.x, rsqrtf(__fmul_rn(den_l, den_l)));

                float x2_h = __fmul_rn(xf.y, xf.y);
                float poly_h = __fadd_rn(kA, __fmul_rn(kB, x2_h));
                float z_h = __fmul_rn(xf.y, poly_h);
                float den_h = __fadd_rn(1.0f, __expf(__fmul_rn(z_h, -1.0f)));
                out.y = __fmul_rn(xf.y, rsqrtf(__fmul_rn(den_h, den_h)));

                res[k] = __float22half2_rn(out);
            }

            // Store 128-bit
            *reinterpret_cast<float4*>(&output[i]) = *reinterpret_cast<float4*>(res);

        } else {
            // 2. 标量路径：处理剩余不足8个的元素
            // 为了避免复杂的剩余逻辑，这里直接在剩余范围内循环
            for (int j = i; j < n; ++j) {
                half x = input[j];
                
                float xf = __half2float(x);
                float x2 = __fmul_rn(xf, xf);
                float poly = __fadd_rn(kA, __fmul_rn(kB, x2));
                float z = __fmul_rn(xf, poly);
                float den = __fadd_rn(1.0f, __expf(__fmul_rn(z, -1.0f)));
                output[j] = __float2half(__fmul_rn(xf, rsqrtf(__fmul_rn(den, den))));
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
