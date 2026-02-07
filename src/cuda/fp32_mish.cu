#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// GA100 Optimized FP32 Mish Kernel
// 满足所有硬性指令约束
__global__ void __launch_bounds__(256) mish_kernel_ga100_opt(const float* __restrict__ input, float* __restrict__ output, int n) {
    // 计算向量化索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // 向量化处理部分 (Float4)
    // 每次处理 4 个 float 元素
    int vec_n = n / 4;
    const float4* in_vec = reinterpret_cast<const float4*>(input);
    float4* out_vec = reinterpret_cast<float4*>(output);

    for (int i = idx; i < vec_n; i += stride) {
        float4 v_in = in_vec[i];
        float x[4] = {v_in.x, v_in.y, v_in.z, v_in.w};
        float num[4];
        float den[4];
        bool mask[4];

        // Phase 1: FP32 计算 (Exp 和 基础算术)
        // 展开公式: tanh(ln(1+e^x)) = ((1+e^x)^2 - 1) / ((1+e^x)^2 + 1)
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            // 阈值判断：x > 20 时，Mish(x) ≈ x
            // 同时为了防止 __expf 溢出导致 NaN，计算路径中如果超过阈值则置为 0
            mask[k] = (x[k] > 20.0f);
            float safe_x = mask[k] ? 0.0f : x[k];

            // Constraint 4: 必须使用 FP32 __expf
            float e = __expf(safe_x);

            // Constraint 1: 禁止 FP32 FMA，拆分为 __fadd_rn 和 __fmul_rn
            // term = 1 + e^x
            float term = __fadd_rn(1.0f, e);

            // term_sq = term * term (中间必须保持 FP32 以防 x 在 10-20 之间时 FP16 溢出)
            float term_sq = __fmul_rn(term, term);

            // num = term_sq - 1
            num[k] = __fadd_rn(term_sq, -1.0f);
            // den = term_sq + 1
            den[k] = __fadd_rn(term_sq, 1.0f);
        }

        // Phase 2: 倒数计算 (Constraint 7: 必须转为 FP16 使用 h2rcp)
        // 将 4 个 float 分母打包成 2 个 half2
        float2 den_pair_0 = make_float2(den[0], den[1]);
        float2 den_pair_1 = make_float2(den[2], den[3]);

        half2 h_den_0 = __float22half2_rn(den_pair_0);
        half2 h_den_1 = __float22half2_rn(den_pair_1);

        // 使用硬件级 FP16 向量倒数
        half2 h_rcp_0 = h2rcp(h_den_0);
        half2 h_rcp_1 = h2rcp(h_den_1);

        // 转回 FP32
        float2 rcp_pair_0 = __half22float2(h_rcp_0);
        float2 rcp_pair_1 = __half22float2(h_rcp_1);
        
        float rcp[4] = {rcp_pair_0.x, rcp_pair_0.y, rcp_pair_1.x, rcp_pair_1.y};
        float res_arr[4];

        // Phase 3: 最终组合
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            // Constraint 1: 使用显式 FP32 乘法
            // tanh_val = num * rcp
            float tanh_val = __fmul_rn(num[k], rcp[k]);
            
            // res = x * tanh_val
            float calc_res = __fmul_rn(x[k], tanh_val);

            // 如果超过阈值，直接返回 x，否则返回计算值
            res_arr[k] = mask[k] ? x[k] : calc_res;
        }

        // 写入结果
        out_vec[i] = make_float4(res_arr[0], res_arr[1], res_arr[2], res_arr[3]);
    }

    // 尾部标量处理 (处理不能被 4 整除的剩余元素)
    int tail_start = vec_n * 4;
    for (int i = tail_start + idx; i < n; i += stride) {
        float x = input[i];
        
        // 同样的逻辑，但针对标量
        bool mask = (x > 20.0f);
        float safe_x = mask ? 0.0f : x;
        
        float e = __expf(safe_x);
        float term = __fadd_rn(1.0f, e);
        float term_sq = __fmul_rn(term, term);
        float num = __fadd_rn(term_sq, -1.0f);
        float den = __fadd_rn(term_sq, 1.0f);

        // 即使是标量也必须遵守 Constraint 7 使用 h2rcp
        // 构造一个 duplicate 的 half2 来计算
        half2 h_den = __float22half2_rn(make_float2(den, den));
        half2 h_rcp = h2rcp(h_den);
        float rcp = __low2float(h_rcp); // 取低位结果

        float tanh_val = __fmul_rn(num, rcp);
        float calc_res = __fmul_rn(x, tanh_val);

        output[i] = mask ? x : calc_res;
    }
}

void launch_mish_fp32(const float* input, float* output, int total_elements) {
    int threads = 256;
    // 由于内核主要以 float4 (4元素) 为单位处理，Block 数量按 total/4 计算
    // 但为了保证尾部处理和足够的网格覆盖，我们按 total_elements/4 向上取整
    int num_vectors = (total_elements + 3) / 4;
    int blocks = (num_vectors + threads - 1) / threads;
    
    mish_kernel_ga100_opt<<<blocks, threads>>>(input, output, total_elements);
}