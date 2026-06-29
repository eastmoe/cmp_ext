#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float reciprocal_positive_rsqrt(float x) {
    return rsqrtf(__fmul_rn(x, x));
}

// GA100 Optimized Softsign Kernel
__global__ void softsign_kernel_fp32(const float* __restrict__ input, float* __restrict__ output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 1. Vectorized Loop (Process 4 floats per iteration)
    // ----------------------------------------------------
    int vec_limit = n / 4;
    for (int i = idx; i < vec_limit; i += stride) {
        // Load input as float4 (128-bit load for memory coalescing)
        float4 in_data = reinterpret_cast<const float4*>(input)[i];
        float4 out_data;

        // Step A: Calculate Denominator (1.0 + |x|)
        // Constraint 1: Do NOT use FMA. Must split into explicit ADD.
        // Use __fadd_rn explicitly.
        float d1 = __fadd_rn(1.0f, fabsf(in_data.x));
        float d2 = __fadd_rn(1.0f, fabsf(in_data.y));
        float d3 = __fadd_rn(1.0f, fabsf(in_data.z));
        float d4 = __fadd_rn(1.0f, fabsf(in_data.w));

        // Step B: Final Multiplication (x * reciprocal)
        // Constraint 1: Do NOT use FMA. Must split into explicit MUL.
        // Use __fmul_rn explicitly.
        out_data.x = __fmul_rn(in_data.x, reciprocal_positive_rsqrt(d1));
        out_data.y = __fmul_rn(in_data.y, reciprocal_positive_rsqrt(d2));
        out_data.z = __fmul_rn(in_data.z, reciprocal_positive_rsqrt(d3));
        out_data.w = __fmul_rn(in_data.w, reciprocal_positive_rsqrt(d4));

        // Store output
        reinterpret_cast<float4*>(output)[i] = out_data;
    }

    // 2. Scalar Tail Loop (Handle remaining elements)
    // ----------------------------------------------------
    int tail_start = vec_limit * 4;
    for (int i = tail_start + idx; i < n; i += stride) {
        float x = input[i];

        // Constraint 1: Explicit ADD
        float d = __fadd_rn(1.0f, fabsf(x));

        // Constraint 1: Explicit MUL
        output[i] = __fmul_rn(x, reciprocal_positive_rsqrt(d));
    }
}

void launch_softsign_fp32(const float* input, float* output, int total_elements) {
    int threads = 256;
    // Calculate blocks based on scalar elements, but kernel handles vectorization internally.
    int blocks = (total_elements + threads - 1) / threads;
    
    // Using a large grid can help occupancy, but limiting it prevents tail effect overhead 
    // for small N. For GA100, occupancy is key.
    if (blocks > 65535) blocks = 65535; // Cap grid size to reasonable limit for grid-stride loop

    softsign_kernel_fp32<<<blocks, threads>>>(input, output, total_elements);
}
// [8192x8192]：0.413 ms
