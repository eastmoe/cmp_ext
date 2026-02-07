#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 辅助宏：将 float 常量转为 half/half2
#define H(x) (__float2half(x))
#define H2(x) (__float2half2_rn(x))

// Abramowitz Coefficients
#define ERF_P  0.3275911f
#define ERF_A1 0.254829592f
#define ERF_A2 -0.284496736f
#define ERF_A3 1.421413741f
#define ERF_A4 -1.453152027f
#define ERF_A5 1.061405429f

// 约束：Exp 必须在 FP32 下计算
__device__ __forceinline__ half2 custom_exp_h2_via_fp32(half2 x) {
    float2 f = __half22float2(x);
    f.x = __expf(f.x);
    f.y = __expf(f.y);
    return __float22half2_rn(f);
}

__device__ __forceinline__ half custom_exp_h_via_fp32(half x) {
    float f = __half2float(x);
    f = __expf(f);
    return __float2half(f);
}

__global__ void erf_kernel_fp16(const half* input, half* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 2;

    // 向量化路径
    if (vec_idx + 1 < n) {
        // Load
        half2 val = ((const half2*)input)[idx];
        
        // 1. Abs
        // h2abs 没有直接的 intrinsic，通常用 __habs2 或者位操作，这里我们用 __habs2
        half2 x = __habs2(val);
        
        // 保存符号
        // 笨办法：val >= 0 ? 1 : -1。
        // 为了保持 pure arithmetic，我们先算完结果再乘符号，或者最后处理符号
        // 这里为了简单，后续通过比较处理符号。
        
        // 2. Denom = 1 + p*x
        // 使用 FMA: 1 + p*x -> __hfma2(p, x, 1.0)
        half2 denom = __hfma2(H2(ERF_P), x, H2(1.0f));

        // 3. Reciprocal t = 1/denom
        // 约束: 使用 h2rcp
        half2 t = h2rcp(denom);

        // 4. Polynomial (Horner's Method) with FMA
        // poly = ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t
        half2 poly = __hfma2(H2(ERF_A5), t, H2(ERF_A4));
        poly = __hfma2(poly, t, H2(ERF_A3));
        poly = __hfma2(poly, t, H2(ERF_A2));
        poly = __hfma2(poly, t, H2(ERF_A1));
        poly = __hmul2(poly, t); // 最后一步是乘法

        // 5. Exp part: exp(-val^2)
        // 无论 val 正负，val^2 都是正的。
        // -val^2 = 0 - val*val
        half2 neg_x_sq = __hneg2(__hmul2(val, val)); // 也可以用 __hfma2(val, -val, 0)
        
        // 约束: 必须转 FP32 算 exp
        half2 e_val = custom_exp_h2_via_fp32(neg_x_sq);

        // 6. Result = 1 - poly * exp
        // Res = 1 + (-poly * exp) -> __hfma2(-poly, exp, 1.0)
        // 或者 __hfma2(poly, -exp, 1.0)
        half2 neg_e = __hneg2(e_val);
        half2 res = __hfma2(poly, neg_e, H2(1.0f));

        // 7. 恢复符号 (copysign)
        // 既然不能用 standard copysign，我们手动乘符号
        // 原始 val
        half2 final_res;
        
        // 分别处理 high/low (CUDA half2 没有直接 vector copysign)
        half v_low = __low2half(val);
        half v_high = __high2half(val);
        half r_low = __low2half(res);
        half r_high = __high2half(res);

        // 简单的符号位处理：如果 v < 0, r = -r
        // 也可以: sign = (v >= 0) ? 1.0 : -1.0; r = r * sign;
        // 这里用乘法保持流水线
        half s_low = __hge(v_low, H(0.0f)) ? H(1.0f) : H(-1.0f);
        half s_high = __hge(v_high, H(0.0f)) ? H(1.0f) : H(-1.0f);
        
        final_res = __halves2half2(__hmul(r_low, s_low), __hmul(r_high, s_high));

        ((half2*)output)[idx] = final_res;
    } 
    // 标量尾部处理
    else if (vec_idx < n) {
        half val = input[vec_idx];
        half x = __habs(val);
        
        // 标量 FMA
        half denom = __hfma(H(ERF_P), x, H(1.0f));
        half t = hrcp(denom);

        half poly = __hfma(H(ERF_A5), t, H(ERF_A4));
        poly = __hfma(poly, t, H(ERF_A3));
        poly = __hfma(poly, t, H(ERF_A2));
        poly = __hfma(poly, t, H(ERF_A1));
        poly = __hmul(poly, t);

        half neg_x_sq = __hneg(__hmul(val, val));
        half e_val = custom_exp_h_via_fp32(neg_x_sq);

        half res = __hfma(poly, __hneg(e_val), H(1.0f));
        
        half sign = __hge(val, H(0.0f)) ? H(1.0f) : H(-1.0f);
        output[vec_idx] = __hmul(res, sign);
    }
}

void launch_erf_fp16(const void* input, void* output, int total_elements) {
    int threads = 256;
    int vec_elements = (total_elements + 1) / 2;
    int blocks = (vec_elements + threads - 1) / threads;
    erf_kernel_fp16<<<blocks, threads>>>((const half*)input, (half*)output, total_elements);
}