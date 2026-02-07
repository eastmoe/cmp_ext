#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// 辅助函数：处理单个 half2 数据的 Softplus 逻辑
// 严格遵循所有指令约束
__device__ __forceinline__ __half2 compute_softplus_h2(__half2 v_h2, float beta, float threshold, float2 inv_beta_f2) {
    // 1. 转为 float2 进行高精度乘法和 Exp 计算 (约束4: 禁止FP16 hexp)
    float2 v = __half22float2(v_h2);
    float2 bx;

    // 约束1: 禁止 FP32 FMA，使用 __fmul_rn
    bx.x = __fmul_rn(v.x, beta);
    bx.y = __fmul_rn(v.y, beta);

    // -------------------------------------------------------------
    // 计算 Log(1 + exp(bx)) 分支
    // -------------------------------------------------------------
    
    // 约束4: 使用 __expf (FP32)
    float2 e;
    e.x = __expf(bx.x);
    e.y = __expf(bx.y);

    // 约束1: 禁止 FMA，使用 __fadd_rn
    float2 sum;
    sum.x = __fadd_rn(1.0f, e.x);
    sum.y = __fadd_rn(1.0f, e.y);

    // 约束8: 禁止 FP32 Log，必须转换为 FP16 使用 h2log
    __half2 sum_h2 = __float22half2_rn(sum);
    
    // 向量化 FP16 Log
    __half2 log_h2 = h2log(sum_h2);

    // 转回 float2 继续后续计算 (为了保持与 inv_beta 的精度匹配，以及处理 threshold 分支)
    float2 log_val = __half22float2(log_h2);

    float2 res_calc;
    // 约束1: 禁止 FMA，使用 __fmul_rn
    // inv_beta_f2 已经在 Kernel 入口处通过 h2rcp 计算并转为 float
    res_calc.x = __fmul_rn(log_val.x, inv_beta_f2.x);
    res_calc.y = __fmul_rn(log_val.y, inv_beta_f2.y);

    // -------------------------------------------------------------
    // Threshold 选择逻辑
    // -------------------------------------------------------------
    float2 result;
    result.x = (bx.x > threshold) ? v.x : res_calc.x;
    result.y = (bx.y > threshold) ? v.y : res_calc.y;

    return __float22half2_rn(result);
}

// 针对 GA100 优化的 Kernel
__global__ void softplus_kernel_opt_ga100(const __half* __restrict__ input, 
                                          __half* __restrict__ output, 
                                          int total_elements, 
                                          float beta, 
                                          float threshold) {
    // -------------------------------------------------------------
    // 约束7: 禁止 FP32 __frcp_rn。计算 1/beta 必须使用 h2rcp
    // -------------------------------------------------------------
    // 将标量 beta 转换为 half2
    __half2 h2_beta = __float2half2_rn(beta);
    
    // 使用 vector FP16 reciprocal (h2rcp)
    __half2 h2_inv_beta = h2rcp(h2_beta);
    
    // 转回 float2 用于后续的高精度乘法 (根据 Softplus 公式 1/beta * log(...))
    float2 inv_beta_f2 = __half22float2(h2_inv_beta);

    // -------------------------------------------------------------
    // 向量化处理 Loop (128-bit load/store)
    // -------------------------------------------------------------
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // 每个线程一次处理 8 个 half (即 1 个 float4)
    // 使用 reinterpret_cast 进行向量化加载
    const float4* in_vec_ptr = reinterpret_cast<const float4*>(input);
    float4* out_vec_ptr = reinterpret_cast<float4*>(output);

    int vec_loop_limit = total_elements / 8;

    for (int i = tid; i < vec_loop_limit; i += stride) {
        // Load 128-bit (8 x half)
        float4 in_data = in_vec_ptr[i];
        float4 out_data;

        // 将 float4 重新解释为 4 个 half2
        __half2* in_h2 = reinterpret_cast<__half2*>(&in_data);
        __half2* out_h2 = reinterpret_cast<__half2*>(&out_data);

        // 展开循环处理 4 个 half2 (Instruction Level Parallelism)
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            out_h2[k] = compute_softplus_h2(in_h2[k], beta, threshold, inv_beta_f2);
        }

        // Store 128-bit
        out_vec_ptr[i] = out_data;
    }

    // -------------------------------------------------------------
    // 处理剩余元素 (Tail Handling)
    // -------------------------------------------------------------
    int remaining_start = vec_loop_limit * 8;
    for (int i = remaining_start + tid; i < total_elements; i += stride) {
        __half val_h = input[i];
        float val = __half2float(val_h);
        
        // bx = val * beta (No FMA)
        float bx = __fmul_rn(val, beta);
        float res_val;

        if (bx > threshold) {
            res_val = val;
        } else {
            // 约束4: No hexp -> __expf
            float e = __expf(bx);
            
            // 约束1: No FMA -> __fadd_rn
            float log_in = __fadd_rn(1.0f, e);
            
            // 约束8: No FP32 Log -> 转换为 half 使用 hlog
            __half log_in_h = __float2half(log_in);
            __half log_res_h = hlog(log_in_h); // scalar half log
            float lg = __half2float(log_res_h);

            // 约束1: No FMA -> __fmul_rn
            // 使用之前 h2rcp 算出来的 inv_beta
            res_val = __fmul_rn(lg, inv_beta_f2.x);
        }
        output[i] = __float2half(res_val);
    }
}

void launch_softplus_fp16(const void* input, void* output, int total_elements, float beta, float threshold) {
    // 针对 GA100 调整 Block 大小
    int threads = 256;
    // 每个线程处理 8 个元素 (Vectorized float4)
    int elems_per_thread = 8;
    
    int total_threads_needed = (total_elements + elems_per_thread - 1) / elems_per_thread;
    int blocks = (total_threads_needed + threads - 1) / threads;
    
    // 限制最大 Grid 大小
    if (blocks > 65535) blocks = 65535; 

    softplus_kernel_opt_ga100<<<blocks, threads>>>(
        (const __half*)input, 
        (__half*)output, 
        total_elements, 
        beta, 
        threshold
    );
}
//[8192x8192] (beta=1.0, threshold=20.0):31.2 x