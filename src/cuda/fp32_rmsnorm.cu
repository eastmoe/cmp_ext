#include <cuda.h>
#include <cuda_runtime.h>
// Warp Reduce 求和 - 严格使用 __fadd_rn 避免 FMA
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float shuffled = __shfl_down_sync(0xffffffff, val, offset);
        val = __fadd_rn(val, shuffled);
    }
    return val;
}

// Block Reduce 求和 - 严格使用 __fadd_rn
__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; 
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val); 

    if (lane == 0) shared[wid] = val; 
    __syncthreads();

    // 假设 BlockDim 不超过 1024
    float warp_val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    
    if (wid == 0) warp_val = warpReduceSum(warp_val); 
    
    return warp_val;
}

__global__ void rmsnorm_fp32_kernel(float* output, const float* input, const float* weight, int cols, float eps) {
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // 定位当前行的指针
    const float* row_in_ptr = input + row_idx * cols;
    float* row_out_ptr = output + row_idx * cols;

    // -----------------------------------------------------------------
    // 1. 计算平方和 (Sum of Squares)
    // -----------------------------------------------------------------
    float sum_sq = 0.0f;

    // 使用 float4 向量化加载以优化 GA100 内存带宽
    int vec_loop_limit = cols / 4;
    const float4* row_in_vec = (const float4*)row_in_ptr;

    // 主向量循环
    #pragma unroll
    for (int i = tid; i < vec_loop_limit; i += blockDim.x) {
        float4 v = row_in_vec[i];
        
        // Constraint 1: 禁止 FP32 FMA，强制拆分为 fmul + fadd
        float sq_x = __fmul_rn(v.x, v.x);
        float sq_y = __fmul_rn(v.y, v.y);
        float sq_z = __fmul_rn(v.z, v.z);
        float sq_w = __fmul_rn(v.w, v.w);

        sum_sq = __fadd_rn(sum_sq, sq_x);
        sum_sq = __fadd_rn(sum_sq, sq_y);
        sum_sq = __fadd_rn(sum_sq, sq_z);
        sum_sq = __fadd_rn(sum_sq, sq_w);
    }

    // 处理剩余元素 (cols % 4 != 0)
    int tail_start = vec_loop_limit * 4 + tid;
    if (tail_start < cols) {
        // 由于剩余部分小于 BlockDim，这里不需要循环，只需要一次判断
        // 但为了通用性，如果 blockDim 极小，保持 stride 逻辑
        for (int i = tail_start; i < cols; i += blockDim.x) {
            float v = row_in_ptr[i];
            float sq = __fmul_rn(v, v);
            sum_sq = __fadd_rn(sum_sq, sq);
        }
    }

    // -----------------------------------------------------------------
    // 2. 规约求全行平方和
    // -----------------------------------------------------------------
    sum_sq = blockReduceSum(sum_sq);
    
    // -----------------------------------------------------------------
    // 3. 计算 RMS 的倒数
    // -----------------------------------------------------------------
    __shared__ float inv_rms;
    if (tid == 0) {
        float mean = __fdividef(sum_sq, (float)cols);
        
        // Constraint 1: 使用 fadd
        float var_eps = __fadd_rn(mean, eps);
        
        // rsqrtf 使用 SFU，不违反 Constraint 7 (rcp)
        inv_rms = rsqrtf(var_eps);
    }
    __syncthreads();

    // 读取共享内存中的广播值
    float rms_scale = inv_rms;

    // -----------------------------------------------------------------
    // 4. 计算输出并写入 (Output)
    // -----------------------------------------------------------------
    float4* row_out_vec = (float4*)row_out_ptr;
    const float4* weight_vec = (const float4*)weight;

    // 主向量循环
    #pragma unroll
    for (int i = tid; i < vec_loop_limit; i += blockDim.x) {
        float4 v_in = row_in_vec[i];
        float4 v_w = weight_vec[i];
        float4 v_out;

        // 计算公式: out = in * inv_rms * weight
        // Constraint 1: 严格禁止 FMA，使用链式 fmul
        
        // x
        float tmp_x = __fmul_rn(v_in.x, rms_scale);
        v_out.x = __fmul_rn(tmp_x, v_w.x);

        // y
        float tmp_y = __fmul_rn(v_in.y, rms_scale);
        v_out.y = __fmul_rn(tmp_y, v_w.y);

        // z
        float tmp_z = __fmul_rn(v_in.z, rms_scale);
        v_out.z = __fmul_rn(tmp_z, v_w.z);

        // w
        float tmp_w = __fmul_rn(v_in.w, rms_scale);
        v_out.w = __fmul_rn(tmp_w, v_w.w);

        row_out_vec[i] = v_out;
    }

    // 处理剩余元素
    if (tail_start < cols) {
        for (int i = tail_start; i < cols; i += blockDim.x) {
            float val_in = row_in_ptr[i];
            float val_w = weight[i];
            
            float tmp = __fmul_rn(val_in, rms_scale);
            row_out_ptr[i] = __fmul_rn(tmp, val_w);
        }
    }
}

void launch_rmsnorm_fp32(float* output, const float* input, const float* weight, int rows, int cols, float eps) {
    int threads = 256;
    if (cols > 256) threads = 1024;
    // 每个 Block 处理一行
    rmsnorm_fp32_kernel<<<rows, threads>>>(output, input, weight, cols, eps);
}
