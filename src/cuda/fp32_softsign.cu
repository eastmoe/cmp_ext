#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

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

        // Step B: Calculate Reciprocal (1.0 / denom)
        // Constraint 7: Do NOT use FP32 reciprocal (__frcp_rn). 
        // Must convert to vectorized FP16 and use h2rcp.
        
        // Convert float pairs to half2
        half2 h_d12 = __float22half2_rn(make_float2(d1, d2));
        half2 h_d34 = __float22half2_rn(make_float2(d3, d4));

        // Use FP16 vectorized reciprocal
        half2 h_rcp12 = h2rcp(h_d12);
        half2 h_rcp34 = h2rcp(h_d34);

        // Convert back to float2
        float2 rcp12 = __half22float2(h_rcp12);
        float2 rcp34 = __half22float2(h_rcp34);

        // Step C: Final Multiplication (x * rcp)
        // Constraint 1: Do NOT use FMA. Must split into explicit MUL.
        // Use __fmul_rn explicitly.
        out_data.x = __fmul_rn(in_data.x, rcp12.x);
        out_data.y = __fmul_rn(in_data.y, rcp12.y);
        out_data.z = __fmul_rn(in_data.z, rcp34.x);
        out_data.w = __fmul_rn(in_data.w, rcp34.y);

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

        // Constraint 7: Even for scalar, must use vectorized FP16 h2rcp.
        // Duplicate data to form a half2 vector
        half2 h_d = __float22half2_rn(make_float2(d, d));
        h_d = h2rcp(h_d); // Vectorized op
        float2 rcp_pair = __half22float2(h_d);
        
        // Use the first component
        float rcp = rcp_pair.x;

        // Constraint 1: Explicit MUL
        output[i] = __fmul_rn(x, rcp);
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