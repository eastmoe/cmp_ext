#include <cuda_runtime.h>
#include <cuda_bf16.h>

// GELU Tanh Approximation Constants
// sqrt(2/pi)
#define GELU_COEF_A 0.79788456f 
// 0.044715
#define GELU_COEF_B 0.044715f
#define HALF        0.5f
#define ONE         1.0f

// Helper function to compute GELU for a single float value
// Constraints: No FMA, No standard math lib, use intrinsics, Tanh approx.
__device__ __forceinline__ float compute_gelu_math(float x) {
    // Math: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    
    // Explicitly split FMA: x^2 = x * x
    float x2 = __fmul_rn(x, x);
    
    // x^3 = x^2 * x
    float x3 = __fmul_rn(x2, x);
    
    // inner = 0.044715 * x^3
    float inner = __fmul_rn(GELU_COEF_B, x3);
    
    // poly = x + inner
    float poly = __fadd_rn(x, inner);
    
    // arg = sqrt(2/pi) * poly
    float arg = __fmul_rn(GELU_COEF_A, poly);
    
    // tanh approximation
    float th = __tanhf(arg);
    
    // 1 + tanh
    float factor = __fadd_rn(ONE, th);
    
    // 0.5 * x
    float half_x = __fmul_rn(HALF, x);
    
    // Final result: (0.5 * x) * (1 + tanh(...))
    return __fmul_rn(half_x, factor);
}

__global__ void gelu_bf16_kernel(const __nv_bfloat16* __restrict__ input, __nv_bfloat16* __restrict__ output, int n) {
    // Reinterpret pointers as vectorized types (__nv_bfloat162) to load 32-bits (2 elements) at once
    const __nv_bfloat162* input_v = reinterpret_cast<const __nv_bfloat162*>(input);
    __nv_bfloat162* output_v = reinterpret_cast<__nv_bfloat162*>(output);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Vectorized loop: Process 2 elements per thread
    // n / 2 represents the number of full pairs
    int vec_limit = n / 2;
    
    for (int i = idx; i < vec_limit; i += stride) {
        // 1. Vector Load
        __nv_bfloat162 in_val = input_v[i];
        
        // 2. Unpack to floats
        float x1 = __low2float(in_val);
        float x2 = __high2float(in_val);
        
        // 3. Compute GELU (FP32 math)
        float res1 = compute_gelu_math(x1);
        float res2 = compute_gelu_math(x2);
        
        // 4. Pack back to bfloat162 and Store
        output_v[i] = __floats2bfloat162_rn(res1, res2);
    }

    // Handle the remaining element if n is odd
    if (n % 2 != 0) {
        // The thread that maps to the index 'vec_limit' handles the tail
        // Because the stride loop might have incremented 'i' past vec_limit, 
        // we check if the initial mapping of this thread corresponds to the tail index.
        if (idx == vec_limit) {
            int tail_idx = n - 1;
            float x = __bfloat162float(input[tail_idx]);
            float res = compute_gelu_math(x);
            output[tail_idx] = __float2bfloat16(res);
        }
    }
}

void launch_gelu_bf16(const void* input, void* output, int total_elements) {
    int threads = 256;
    // Calculate blocks based on vectorized elements (2 elements per thread)
    // We effectively process (total_elements + 1) / 2 vectors.
    int total_vectors = (total_elements + 1) / 2;
    int blocks = (total_vectors + threads - 1) / threads;

    gelu_bf16_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(input),
        reinterpret_cast<__nv_bfloat16*>(output),
        total_elements
    );
}
//[8192x8192] 0.280 ms, Avg Power:  86.56 W  8.5 x
