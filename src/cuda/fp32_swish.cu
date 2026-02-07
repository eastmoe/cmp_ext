#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// GA100 Optimization for Swish
// Constraints Applied:
// 1. No FP32 FMA -> explicit __fmul_rn, __fadd_rn
// 2. FP16 Vectorized Reciprocal -> float2half2 -> __h2rcp -> half22float
// 3. No FP16 exp/trig -> use FP32 __expf
// 4. Vectorized IO -> float4

__global__ void __launch_bounds__(256) swish_kernel_opt_ga100(
    const float* __restrict__ input, 
    const float* __restrict__ beta, 
    float* __restrict__ output, 
    int n
) {
    // 加载 Beta 并预计算负 Beta，避免循环内取反
    // 使用 __fmul_rn(-1.0f) 确保不生成 FMA 或其他融合指令
    const float b = __ldg(beta);
    const float neg_b = __fmul_rn(b, -1.0f);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // -----------------------------------------------------------
    // 1. Vectorized Path (Float4)
    // -----------------------------------------------------------
    int n_vec = n / 4;
    const float4* in_vec = reinterpret_cast<const float4*>(input);
    float4* out_vec = reinterpret_cast<float4*>(output);

    for (int i = idx; i < n_vec; i += stride) {
        float4 v = in_vec[i];
        float4 r;

        // Step 1: Calculate Exponent Arguments in FP32
        // Formula: tmp = -beta * x
        // Constraint 1: Must split FMA. Using __fmul_rn.
        float t_x = __fmul_rn(v.x, neg_b);
        float t_y = __fmul_rn(v.y, neg_b);
        float t_z = __fmul_rn(v.z, neg_b);
        float t_w = __fmul_rn(v.w, neg_b);

        // Step 2: Compute Exp in FP32 (Constraint 4)
        // 使用 __expf (Hardware Special Function Unit)
        float e_x = __expf(t_x);
        float e_y = __expf(t_y);
        float e_z = __expf(t_z);
        float e_w = __expf(t_w);

        // Step 3: Compute Denominator (1 + exp)
        // Constraint 1: Must split FMA/Add. Using __fadd_rn.
        float d_x = __fadd_rn(1.0f, e_x);
        float d_y = __fadd_rn(1.0f, e_y);
        float d_z = __fadd_rn(1.0f, e_z);
        float d_w = __fadd_rn(1.0f, e_w);

        // Step 4: Compute Reciprocal using Vectorized FP16 (Constraint 7)
        // Forbidden: __frcp_rn, __fdividef
        // Required: Convert to FP16 -> h2rcp -> Convert back
        
        // Pack floats into half2 vectors
        // make_float2 is a helper to construct the vector for conversion
        __half2 h2_xy = __float22half2_rn(make_float2(d_x, d_y));
        __half2 h2_zw = __float22half2_rn(make_float2(d_z, d_w));

        // Execute Hardware Vectorized Reciprocal
        h2_xy = h2rcp(h2_xy);
        h2_zw = h2rcp(h2_zw);

        // Unpack back to FP32
        float2 rcp_xy = __half22float2(h2_xy);
        float2 rcp_zw = __half22float2(h2_zw);

        // Step 5: Final Multiply (x * rcp)
        // Constraint 1: Must split FMA. Using __fmul_rn.
        r.x = __fmul_rn(v.x, rcp_xy.x);
        r.y = __fmul_rn(v.y, rcp_xy.y);
        r.z = __fmul_rn(v.z, rcp_zw.x);
        r.w = __fmul_rn(v.w, rcp_zw.y);

        // Vector Store
        out_vec[i] = r;
    }

    // -----------------------------------------------------------
    // 2. Tail Path (Scalar)
    // -----------------------------------------------------------
    int vec_end_idx = n_vec * 4;
    for (int i = vec_end_idx + idx; i < n; i += stride) {
        float x = input[i];
        
        // FP32 Mul
        float t = __fmul_rn(x, neg_b);
        
        // FP32 Exp
        float e = __expf(t);
        
        // FP32 Add
        float d = __fadd_rn(1.0f, e);

        // Constraint 7: Must use vectorized h2rcp even for scalar logic
        // Pack scalar 'd' into low part of half2, high part dummy (1.0)
        __half2 h2_scalar = __float22half2_rn(make_float2(d, 1.0f));
        
        // Vectorized Reciprocal
        h2_scalar = h2rcp(h2_scalar);
        
        // Convert back
        float2 rcp_res = __half22float2(h2_scalar);
        
        // FP32 Mul (x * rcp) using the low part (x)
        output[i] = __fmul_rn(x, rcp_res.x);
    }
}

void launch_swish_fp32(const float* input, const float* beta, float* output, int total_elements) {
    const int threads = 256;
    
    // A100 SM calculation
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    
    int blocks = (total_elements + threads - 1) / threads;
    // Saturate the GPU but avoid excessive grid lists
    if (blocks > num_sms * 16) blocks = num_sms * 16; 

    swish_kernel_opt_ga100<<<blocks, threads>>>(input, beta, output, total_elements);
}