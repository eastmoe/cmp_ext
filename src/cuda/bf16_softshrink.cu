#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void softshrink_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int n, float lambd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        // 转为 float 计算以保证最简单的兼容性
        float x = __bfloat162float(input[i]);
        float res;

        if (x > lambd) {
            res = x - lambd;
        } else if (x < -lambd) {
            res = x + lambd;
        } else {
            res = 0.0f;
        }

        output[i] = __float2bfloat16(res);
    }
}

void launch_softshrink_bf16(const void* input, void* output, int total_elements, float lambd) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    // ----------- 修复点 -----------
    // 增加了 <<<blocks, threads>>> 
    softshrink_kernel_bf16<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(input),
        reinterpret_cast<__nv_bfloat16*>(output),
        total_elements, 
        lambd
    );
    
    // 建议：检查是否有启动错误（可选，但推荐）
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // 在实际PyTorch扩展中通常使用 TORCH_CHECK 或 printf 调试
        // printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}