#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// 标量 Mish 计算辅助函数
// 实现公式: x * tanh(ln(1 + exp(x)))
// 严格遵守禁用 FMA 和使用内联函数的限制
__device__ __forceinline__ float mish_scalar_op(float x) {
    float sp;
    // 优化：当 x >= 20 时，softplus(x) ≈ x，tanh(x) ≈ 1，结果 ≈ x
    // 这避免了溢出并减少了计算量
    if (x >= 20.0f) {
        sp = x;
    } else {
        // Softplus: ln(1 + exp(x))
        // 使用 __expf 和 __logf 替代标准库函数
        // 使用 __fadd_rn 替代标准加法
        float exp_val = __expf(x);
        float sum_val = __fadd_rn(1.0f, exp_val);
        sp = __logf(sum_val);
    }
    
    // tanh(sp) 使用 __tanhf
    float tanh_val = __tanhf(sp);
    
    // 结果: x * tanh_val，使用 __fmul_rn 替代标准乘法
    return __fmul_rn(x, tanh_val);
}

// 处理打包的 BF16 对（存储在 float 中）的辅助函数
__device__ __forceinline__ float process_bf16_pair(float raw) {
    // 将 float 的二进制位解释为两个 bf16 (__nv_bfloat162)
    __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&raw);
    
    // 转换为两个 float 进行计算
    float2 f = __bfloat1622float2(v);
    
    // 分别计算 Mish
    f.x = mish_scalar_op(f.x);
    f.y = mish_scalar_op(f.y);
    
    // 转换回 bf16 向量
    __nv_bfloat162 res = __float22bfloat162_rn(f);
    
    // 重新解释为 float 返回
    return *reinterpret_cast<float*>(&res);
}

__global__ void mish_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 检查地址是否满足 16 字节对齐（float4 要求）
    // GA100 上通常张量是内存对齐的，这允许我们使用 128 位加载
    bool aligned = (((uintptr_t)input | (uintptr_t)output) & 15) == 0;

    if (aligned) {
        // 向量化路径：每个线程处理 8 个元素 (1个 float4)
        int n_vec = n / 8;
        const float4* in_vec = (const float4*)input;
        float4* out_vec = (float4*)output;

        for (int i = idx; i < n_vec; i += stride) {
            float4 v = in_vec[i];
            
            // float4 包含 4 个 float，每个 float 包含 2 个 bf16
            // 共处理 8 个 bf16
            v.x = process_bf16_pair(v.x);
            v.y = process_bf16_pair(v.y);
            v.z = process_bf16_pair(v.z);
            v.w = process_bf16_pair(v.w);
            
            out_vec[i] = v;
        }

        // 处理尾部剩余元素 (< 8 个)
        int tail_start = n_vec * 8;
        int tail_rem = n - tail_start;
        
        // 使用前几个线程处理尾部，避免启动新循环
        if (idx < tail_rem) {
            int i = tail_start + idx;
            float x = __bfloat162float(input[i]);
            float res = mish_scalar_op(x);
            output[i] = __float2bfloat16(res);
        }
    } else {
        // 未对齐回退路径：标量处理
        // 由于 Launch 配置是按向量化缩减了 Block 数量，这里的 stride 会自适应
        for (int i = idx; i < n; i += stride) {
            float x = __bfloat162float(input[i]);
            float res = mish_scalar_op(x);
            output[i] = __float2bfloat16(res);
        }
    }
}

void launch_mish_bf16(const void* input, void* output, int total_elements) {
    int threads = 256;
    
    // 优化 Launch 配置：
    // 每个线程处理 8 个元素 (128-bit load)，因此大幅减少需要的 Block 数量
    int elems_per_thread = 8;
    int total_threads = (total_elements + elems_per_thread - 1) / elems_per_thread;
    int blocks = (total_threads + threads - 1) / threads;
    
    mish_kernel_bf16<<<blocks, threads>>>(
        (const __nv_bfloat16*)input, 
        (__nv_bfloat16*)output, 
        total_elements
    );
}
//[8192x8192]：0.210 ms, Avg Power:  71.58 W 34.1 x
