#include <cuda_runtime.h>
#include <cuda_bf16.h>

// 保持函数名称、入口一致

// 1. 复用 FP32 规约，修改为 __fadd_rn 以避免隐式 FMA
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val = __fadd_rn(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; 
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__global__ void layernorm_bf16_kernel(
    __nv_bfloat16* output, 
    const __nv_bfloat16* input, 
    const __nv_bfloat16* gamma, 
    const __nv_bfloat16* beta, 
    int rows, 
    int cols, 
    float eps) {
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (bid >= rows) return;

    // 使用向量化指针
    const __nv_bfloat16* row_input = input + bid * cols;
    __nv_bfloat16* row_output = output + bid * cols;
    
    // 向量化处理所需的变量
    int vec_cols = cols / 2;
    const __nv_bfloat162* row_input_v = reinterpret_cast<const __nv_bfloat162*>(row_input);
    __nv_bfloat162* row_output_v = reinterpret_cast<__nv_bfloat162*>(row_output);
    const __nv_bfloat162* gamma_v = reinterpret_cast<const __nv_bfloat162*>(gamma);
    const __nv_bfloat162* beta_v = reinterpret_cast<const __nv_bfloat162*>(beta);

    float sum = 0.0f;
    float sum_sq = 0.0f;

    // 1. 统计阶段：向量化读取与计算
    int i = tid;
    for (; i < vec_cols; i += blockDim.x) {
        __nv_bfloat162 val_v = row_input_v[i];
        float2 val_f2 = __bfloat1622float2(val_v);

        // 显式拆分 FMA 为 add 和 mul (Constraint 1)
        // val.x
        sum = __fadd_rn(sum, val_f2.x);
        sum_sq = __fadd_rn(sum_sq, __fmul_rn(val_f2.x, val_f2.x));
        
        // val.y
        sum = __fadd_rn(sum, val_f2.y);
        sum_sq = __fadd_rn(sum_sq, __fmul_rn(val_f2.y, val_f2.y));
    }

    // 处理剩余的尾部元素 (Scalar)
    if (i * 2 < cols) {
        int idx = i * 2;
        if (idx < cols) { // 实际上此时 idx 肯定是 cols-1
             float val = __bfloat162float(row_input[idx]);
             sum = __fadd_rn(sum, val);
             sum_sq = __fadd_rn(sum_sq, __fmul_rn(val, val));
        }
    }

    sum = blockReduceSum(sum);
    sum_sq = blockReduceSum(sum_sq);

    __shared__ float s_mean;
    __shared__ float s_inv_std;

    if (tid == 0) {
        float fcols = (float)cols;
        // 使用 __fdividef 替代除法 (Constraint 2 & 4)
        float mean = __fdividef(sum, fcols);
        
        // var = sum_sq/cols - mean^2
        // 显式拆分：sub(div(sum_sq, cols), mul(mean, mean))
        float mean_sq = __fmul_rn(mean, mean);
        float avg_sq = __fdividef(sum_sq, fcols);
        float var = __fsub_rn(avg_sq, mean_sq); // 使用 __fsub_rn 避免 FMA
        
        if (var < 0.0f) var = 0.0f;
        
        s_mean = mean;
        // 使用 rsqrtf 替代 sqrt (Constraint 5, 实际上这里直接求 inv_std 最优)
        s_inv_std = rsqrtf(__fadd_rn(var, eps));
    }
    __syncthreads();

    float mean = s_mean;
    float inv_std = s_inv_std;

    // 2. 归一化并写回：向量化
    i = tid;
    for (; i < vec_cols; i += blockDim.x) {
        __nv_bfloat162 val_v = row_input_v[i];
        float2 val_f2 = __bfloat1622float2(val_v);
        
        // 计算 val.x
        // (val - mean) * inv_std
        float res_x = __fsub_rn(val_f2.x, mean);
        res_x = __fmul_rn(res_x, inv_std);

        if (gamma != nullptr) {
            float g = __bfloat162float(gamma[i * 2]); // Gamma 即使向量化读取，为了安全需确认对齐，这里复用标量逻辑或假定对齐
            // 优化：尝试向量化读取 Gamma/Beta
            // 注意：gamma/beta 指针可能为空，这里为了极限性能假设若存在则对齐
            // 但为了安全起见，我们解包 vector 读取
            float2 g_v = __bfloat1622float2(gamma_v[i]);
            res_x = __fmul_rn(res_x, g_v.x);
        }
        if (beta != nullptr) {
            float2 b_v = __bfloat1622float2(beta_v[i]);
            res_x = __fadd_rn(res_x, b_v.x);
        }

        // 计算 val.y
        float res_y = __fsub_rn(val_f2.y, mean);
        res_y = __fmul_rn(res_y, inv_std);

        if (gamma != nullptr) {
            float2 g_v = __bfloat1622float2(gamma_v[i]);
            res_y = __fmul_rn(res_y, g_v.y);
        }
        if (beta != nullptr) {
            float2 b_v = __bfloat1622float2(beta_v[i]);
            res_y = __fadd_rn(res_y, b_v.y);
        }

        // 转回 BF16x2 并存储
        row_output_v[i] = __float22bfloat162_rn(make_float2(res_x, res_y));
    }

    // 处理尾部
    if (i * 2 < cols) {
        int idx = i * 2;
        if (idx < cols) {
            float val = __bfloat162float(row_input[idx]);
            val = __fsub_rn(val, mean);
            val = __fmul_rn(val, inv_std);

            if (gamma != nullptr) val = __fmul_rn(val, __bfloat162float(gamma[idx]));
            if (beta != nullptr) val = __fadd_rn(val, __bfloat162float(beta[idx]));

            row_output[idx] = __float2bfloat16(val);
        }
    }
}

void launch_layernorm_bf16(void* output, const void* input, const void* gamma, const void* beta, int rows, int cols, float eps) {
    int block_size = 256;
    while (block_size > cols && block_size > 32) block_size /= 2;

    dim3 grid(rows);
    dim3 block(block_size);

    // 这里的 buffer 转换保持不变，内核内部处理向量化
    layernorm_bf16_kernel<<<grid, block>>>(
        (__nv_bfloat16*)output, 
        (const __nv_bfloat16*)input, 
        (const __nv_bfloat16*)gamma, 
        (const __nv_bfloat16*)beta, 
        rows, cols, eps
    );
}
//Op: LayerNorm [In=64x128x1024, Norm=(1024,)]
//  Torch         :    0.353 ms, Avg Power: 182.68 W
//  Custom        :    0.066 ms, Avg Power: 247.07 W
//  Speedup       :      5.4 x
