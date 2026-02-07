#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 复用 FP32 的规约逻辑，因为我们在 FP16 Kernel 中内部也使用 float 累加
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; 
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__global__ void layernorm_fp16_kernel(
    half* output, 
    const half* input, 
    const half* gamma, 
    const half* beta, 
    int rows, 
    int cols, 
    float eps) {
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (bid >= rows) return;

    const half* row_input = input + bid * cols;
    half* row_output = output + bid * cols;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    // 1. 读取并转换为 float 进行统计
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = __half2float(row_input[i]);
        sum += val;
        sum_sq += val * val;
    }

    sum = blockReduceSum(sum);
    sum_sq = blockReduceSum(sum_sq);

    __shared__ float s_mean;
    __shared__ float s_inv_std;

    if (tid == 0) {
        float mean = sum / cols;
        float var = (sum_sq / cols) - (mean * mean);
        if (var < 0.0f) var = 0.0f;
        s_mean = mean;
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean = s_mean;
    float inv_std = s_inv_std;

    // 2. 归一化并转回 half
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = __half2float(row_input[i]);
        val = (val - mean) * inv_std;

        if (gamma != nullptr) val *= __half2float(gamma[i]);
        if (beta != nullptr) val += __half2float(beta[i]);

        row_output[i] = __float2half(val);
    }
}

void launch_layernorm_fp16(void* output, const void* input, const void* gamma, const void* beta, int rows, int cols, float eps) {
    int block_size = 256;
    while (block_size > cols && block_size > 32) block_size /= 2;

    dim3 grid(rows);
    dim3 block(block_size);

    layernorm_fp16_kernel<<<grid, block>>>(
        (half*)output, 
        (const half*)input, 
        (const half*)gamma, 
        (const half*)beta, 
        rows, cols, eps
    );
}