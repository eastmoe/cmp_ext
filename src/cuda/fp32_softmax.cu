#include <cuda_runtime.h>
#include <cuda_fp16.h>  // 必须包含，用于 half2 和 __h2rcp
#include <algorithm>
#include <cmath>

// ==========================================
// 辅助函数：严格遵守算术指令限制
// ==========================================

// 约束 1: 禁用 FP32 FMA，强制使用 __fadd_rn
__device__ __forceinline__ float add_strict(float a, float b) {
    return __fadd_rn(a, b);
}

// 约束 1: 禁用 FP32 FMA，强制使用 __fsub_rn
__device__ __forceinline__ float sub_strict(float a, float b) {
    return __fsub_rn(a, b);
}

// 约束 1: 禁用 FP32 FMA，强制使用 __fmul_rn
__device__ __forceinline__ float mul_strict(float a, float b) {
    return __fmul_rn(a, b);
}

// 约束 7: 必须转换为 FP16 使用 h2rcp 算完再转换回去
// 解决方案: 使用向量化指令 __floats2half2_rn 和 h2rcp
__device__ __forceinline__ float rcp_via_h2rcp(float x) {
    // 1. 将 FP32 标量 x 复制两份，转换为 half2 (Packed FP16)
    // 对应指令: cvt.rn.f16x2.f32
    __half2 h2_val = __floats2half2_rn(x, x);
    
    // 2. 使用硬件 FP16 向量倒数指令
    // 对应指令: rcp.approx.f16x2
    __half2 h2_res = h2rcp(h2_val);
    
    // 3. 取低位 FP16 转回 FP32
    // 对应指令: cvt.f32.f16
    return __low2float(h2_res);
}

// ==========================================
// Warp Level Primitives
// ==========================================

// Warp Reduce Max
__device__ __forceinline__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp Reduce Sum (Strict FP32 Add)
__device__ __forceinline__ float warpReduceSumStrict(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float shfl_val = __shfl_down_sync(0xffffffff, val, offset);
        val = add_strict(val, shfl_val);
    }
    return val;
}

// ==========================================
// Softmax Kernel (Optimized for GA100)
// ==========================================

__global__ void softmax_kernel_fp32_opt(const float* __restrict__ input, float* __restrict__ output, int rows, int cols) {
    // 动态分配 Shared Memory
    extern __shared__ float s_mem[]; 

    int row_idx = blockIdx.x;
    if (row_idx >= rows) return;

    // 计算当前行的偏移量
    size_t offset = (size_t)row_idx * cols;
    const float* row_input = input + offset;
    float* row_output = output + offset;

    // 线程索引
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;

    // GA100 优化：使用 float4 向量化访问以饱和内存带宽
    int vec_cols = cols / 4;
    
    // ---------------------------------------------------
    // Step 1: Find Max (Pass 1)
    // ---------------------------------------------------
    float local_max = -1e37f;

    // Vectorized Loop
    for (int i = tid; i < vec_cols; i += blockDim.x) {
        // reinterpret_cast 强制生成 LDG.128 指令
        float4 v = reinterpret_cast<const float4*>(row_input)[i];
        local_max = max(local_max, v.x);
        local_max = max(local_max, v.y);
        local_max = max(local_max, v.z);
        local_max = max(local_max, v.w);
    }
    // Scalar Loop
    for (int i = vec_cols * 4 + tid; i < cols; i += blockDim.x) {
        local_max = max(local_max, row_input[i]);
    }

    // Warp Reduce
    local_max = warpReduceMax(local_max);
    
    // Block Reduce (通过 Shared Memory)
    if (lane == 0) s_mem[wid] = local_max;
    __syncthreads();

    float global_max = -1e37f;
    if (wid == 0) {
        float val = (tid * 32 < blockDim.x) ? s_mem[lane] : -1e37f;
        global_max = warpReduceMax(val);
        s_mem[0] = global_max; // 广播
    }
    __syncthreads();
    global_max = s_mem[0];

    // ---------------------------------------------------
    // Step 2: Compute Exp & Sum (Pass 2)
    // ---------------------------------------------------
    float local_sum = 0.0f;

    // Vectorized Loop
    for (int i = tid; i < vec_cols; i += blockDim.x) {
        float4 v = reinterpret_cast<const float4*>(row_input)[i];
        // Strict Math: add(sum, exp(sub(x, max)))
        local_sum = add_strict(local_sum, expf(sub_strict(v.x, global_max)));
        local_sum = add_strict(local_sum, expf(sub_strict(v.y, global_max)));
        local_sum = add_strict(local_sum, expf(sub_strict(v.z, global_max)));
        local_sum = add_strict(local_sum, expf(sub_strict(v.w, global_max)));
    }
    // Scalar Loop
    for (int i = vec_cols * 4 + tid; i < cols; i += blockDim.x) {
        float val = row_input[i];
        local_sum = add_strict(local_sum, expf(sub_strict(val, global_max)));
    }

    // Warp Reduce
    local_sum = warpReduceSumStrict(local_sum);

    if (lane == 0) s_mem[wid] = local_sum;
    __syncthreads();

    // Block Reduce
    float global_sum = 0.0f;
    if (wid == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        float val = (tid < num_warps) ? s_mem[tid] : 0.0f;
        global_sum = warpReduceSumStrict(val);
        s_mem[0] = global_sum;
    }
    __syncthreads();
    global_sum = s_mem[0];

    // ---------------------------------------------------
    // Step 3: Compute Reciprocal (Constraint 7 Fix)
    // ---------------------------------------------------
    // 使用 FP16 向量化指令计算倒数
    float inv_global_sum = rcp_via_h2rcp(global_sum);

    // ---------------------------------------------------
    // Step 4: Normalize & Write (Pass 3)
    // ---------------------------------------------------
    
    // Vectorized Loop
    for (int i = tid; i < vec_cols; i += blockDim.x) {
        float4 v_in = reinterpret_cast<const float4*>(row_input)[i];
        float4 v_out;

        // 展开计算，禁止 FMA
        float e_x = expf(sub_strict(v_in.x, global_max));
        v_out.x   = mul_strict(e_x, inv_global_sum);

        float e_y = expf(sub_strict(v_in.y, global_max));
        v_out.y   = mul_strict(e_y, inv_global_sum);

        float e_z = expf(sub_strict(v_in.z, global_max));
        v_out.z   = mul_strict(e_z, inv_global_sum);

        float e_w = expf(sub_strict(v_in.w, global_max));
        v_out.w   = mul_strict(e_w, inv_global_sum);

        reinterpret_cast<float4*>(row_output)[i] = v_out;
    }

    // Scalar Loop
    for (int i = vec_cols * 4 + tid; i < cols; i += blockDim.x) {
        float val = row_input[i];
        float e = expf(sub_strict(val, global_max));
        row_output[i] = mul_strict(e, inv_global_sum);
    }
}

void launch_softmax_fp32(const float* input, float* output, int rows, int cols) {
    int block_size = 256;
    if (cols < 256) block_size = 128;
    if (cols > 256 && cols <= 512) block_size = 256;
    
    size_t shared_mem_size = 32 * sizeof(float);

    softmax_kernel_fp32_opt<<<rows, block_size, shared_mem_size>>>(input, output, rows, cols);
}
//[8192x8192]：2.783 ms 87.72 W