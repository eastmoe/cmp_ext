#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void softplus_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int total_elements, float beta, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        // 为了计算简单和精度，先转为 float 计算
        float x = __bfloat162float(input[idx]);
        float bx = x * beta;
        
        float val;
        if (bx > threshold) {
            val = x;
        } else {
            val = logf(1.0f + expf(bx)) / beta;
        }
        output[idx] = __float2bfloat16(val);
    }
}

void launch_softplus_bf16(const void* input, void* output, int total_elements, float beta, float threshold) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    softplus_kernel_bf16<<<blocks, threads>>>(
        (const __nv_bfloat16*)input, 
        (__nv_bfloat16*)output, 
        total_elements, 
        beta, 
        threshold
    );
}