#include <cuda_runtime.h>
#include <stdio.h>

// 简单的 Warp 规约求和
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// 简单的 Block 规约求和
__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);     // 每个 Warp 内部求和

    if (lane == 0) shared[wid] = val; // 将 Warp 结果写入 Shared Memory

    __syncthreads(); // 等待所有 Warp 写完

    // 最后由第一个 Warp 处理 Shared Memory 中的部分和
    // 假设 BlockDim.x <= 1024，即最多 32 个 Warp
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    
    if (wid == 0) val = warpReduceSum(val); // 再次 Warp 规约

    return val;
}

__global__ void layernorm_fp32_kernel(
    float* output, 
    const float* input, 
    const float* gamma, 
    const float* beta, 
    int rows, 
    int cols, 
    float eps) {
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (bid >= rows) return;

    // 指向当前行
    const float* row_input = input + bid * cols;
    float* row_output = output + bid * cols;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    // 1. 计算均值 (Mean) 和 平方和
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = row_input[i];
        sum += val;
        sum_sq += val * val;
    }

    sum = blockReduceSum(sum);
    sum_sq = blockReduceSum(sum_sq);

    __shared__ float s_mean;
    __shared__ float s_var;

    if (tid == 0) {
        s_mean = sum / cols;
        // Var = E[X^2] - (E[X])^2
        s_var = (sum_sq / cols) - (s_mean * s_mean);
        // 防止精度问题导致的负方差
        if (s_var < 0.0f) s_var = 0.0f;
    }
    __syncthreads();

    float mean = s_mean;
    float inv_std = rsqrtf(s_var + eps);

    // 2. 归一化并应用 Gamma/Beta
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = (row_input[i] - mean) * inv_std;
        
        if (gamma != nullptr) val *= gamma[i];
        if (beta != nullptr) val += beta[i];
        
        row_output[i] = val;
    }
}

void launch_layernorm_fp32(float* output, const float* input, const float* gamma, const float* beta, int rows, int cols, float eps) {
    int block_size = 256;
    // 如果 cols 很小，可以减小 block_size，但为了简单这里固定
    while (block_size > cols && block_size > 32) block_size /= 2;

    dim3 grid(rows);
    dim3 block(block_size);

    layernorm_fp32_kernel<<<grid, block>>>(output, input, gamma, beta, rows, cols, eps);
}