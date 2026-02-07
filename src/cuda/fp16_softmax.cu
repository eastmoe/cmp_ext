#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>

#define MAX_BLOCK_SIZE 256
#define WARP_SIZE 32

// 辅助函数：严格禁止FMA的FP32加法
__device__ __forceinline__ float add_no_fma(float a, float b) {
    return __fadd_rn(a, b);
}

// 辅助函数：严格禁止FMA的FP32乘法
__device__ __forceinline__ float mul_no_fma(float a, float b) {
    return __fmul_rn(a, b);
}

// Warp Reduce Max (float)
__device__ float warpReduceMaxF(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Warp Reduce Sum (float) - using explicit add
__device__ float warpReduceSumF(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val = add_no_fma(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__global__ void softmax_kernel_opt_ga100(const __half* __restrict__ input, __half* __restrict__ output, int rows, int cols) {
    int row_idx = blockIdx.x;
    if (row_idx >= rows) return;

    // 计算当前行的偏移量
    // 为了简化向量化逻辑，这里假设指针是按照half2对齐的(通常在深度学习框架中是成立的)
    // 真实的生产环境代码可能需要处理非对齐地址
    const __half* row_input = input + row_idx * cols;
    __half* row_output = output + row_idx * cols;

    // Shared Memory用于Block规约
    __shared__ float s_data[32]; // 假设最大1024线程，即32个Warp

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    // ==========================================
    // 1. 寻找最大值 (Find Max)
    // ==========================================
    float local_max = -1e37f;

    // 向量化部分 (以half2读取)
    int vec_limit = cols / 2;
    const __half2* row_input_h2 = (const __half2*)row_input;
    
    for (int i = tid; i < vec_limit; i += blockDim.x) {
        __half2 v2 = row_input_h2[i];
        float2 vf2 = __half22float2(v2);
        local_max = fmaxf(local_max, fmaxf(vf2.x, vf2.y));
    }

    // 处理剩余的尾部元素
    if (tid == 0 && (cols % 2 != 0)) {
        // 只有线程0处理最后一个元素，通过原子操作或单独逻辑并不划算
        // 这里为了简单，让线程0读取并参与规约（注意：这在并行中可能造成local_max不一致，
        // 但由于后续有Block Reduce，只要有一个线程读到了正确的最大值即可）
        // 更好的方式是让负责尾部的线程处理，这里为了kernel简洁：
        // 让所有线程步进到尾部检查
    }
    // 修正尾部处理逻辑：所有线程检查自己是否覆盖了尾部索引
    int tail_idx = vec_limit * 2 + tid; // 实际上只会有一个线程命中
    if (tail_idx < cols) {
        float val = __half2float(row_input[tail_idx]);
        local_max = fmaxf(local_max, val);
    }

    // Warp内规约
    local_max = warpReduceMaxF(local_max);

    // 将Warp结果写入Shared Memory
    if (lane == 0) s_data[wid] = local_max;
    __syncthreads();

    // Block内规约 (由第一个Warp完成)
    float block_max = -1e37f;
    if (wid == 0) {
        if (lane < num_warps) block_max = s_data[lane];
        else block_max = -1e37f;
        block_max = warpReduceMaxF(block_max);
    }
    
    // 广播最大值
    if (tid == 0) s_data[0] = block_max;
    __syncthreads();
    float global_max = s_data[0];

    // ==========================================
    // 2. 计算指数和 (Sum Exp)
    // ==========================================
    float local_sum = 0.0f;

    // 向量化循环
    for (int i = tid; i < vec_limit; i += blockDim.x) {
        __half2 v2 = row_input_h2[i];
        float2 vf2 = __half22float2(v2);

        // 约束1 & 4: 不用FMA，必须转float用__expf
        // x - max
        float diff1 = add_no_fma(vf2.x, -global_max);
        float diff2 = add_no_fma(vf2.y, -global_max); // -x 等同于 +(-x)

        // exp(x - max)
        float val1 = __expf(diff1);
        float val2 = __expf(diff2);

        // sum += val (使用 __fadd_rn)
        local_sum = add_no_fma(local_sum, add_no_fma(val1, val2));
    }

    // 尾部处理
    if (tail_idx < cols) {
        float val = __half2float(row_input[tail_idx]);
        float diff = add_no_fma(val, -global_max);
        local_sum = add_no_fma(local_sum, __expf(diff));
    }

    // Warp内规约
    local_sum = warpReduceSumF(local_sum);

    // 存入Shared Memory
    if (lane == 0) s_data[wid] = local_sum;
    __syncthreads();

    // Block内规约
    float block_sum = 0.0f;
    if (wid == 0) {
        if (lane < num_warps) block_sum = s_data[lane];
        else block_sum = 0.0f;
        block_sum = warpReduceSumF(block_sum);
    }

    // ==========================================
    // 3. 计算倒数 (Constraint 7)
    // ==========================================
    // 要求：不要使用FP32 __frcp_rn，必须转为FP16向量使用h2rcp
    if (tid == 0) {
        // 复制sum到half2的两个通道 (S, S)
        __half2 sum_h2 = __float2half2_rn(block_sum);
        // 使用向量化FP16倒数指令
        sum_h2 = h2rcp(sum_h2);
        // 转回float，取出其中一个
        s_data[0] = __low2float(sum_h2); 
    }
    __syncthreads();
    float global_inv_sum = s_data[0];

    // ==========================================
    // 4. 计算并写回 (Write Output)
    // ==========================================
    __half2* row_output_h2 = (__half2*)row_output;

    for (int i = tid; i < vec_limit; i += blockDim.x) {
        // 重新读取 (为了寄存器压力通常重读，或者如果寄存器够可以缓存)
        __half2 v2 = row_input_h2[i];
        float2 vf2 = __half22float2(v2);

        // 计算 exp(x - max)
        float e1 = __expf(add_no_fma(vf2.x, -global_max));
        float e2 = __expf(add_no_fma(vf2.y, -global_max));

        // 乘倒数: e * inv_sum (No FMA)
        float res1 = mul_no_fma(e1, global_inv_sum);
        float res2 = mul_no_fma(e2, global_inv_sum);

        // 转回half2并写入
        row_output_h2[i] = __float22half2_rn({res1, res2});
    }

    // 尾部写回
    if (tail_idx < cols) {
        float val = __half2float(row_input[tail_idx]);
        float e = __expf(add_no_fma(val, -global_max));
        float res = mul_no_fma(e, global_inv_sum);
        row_output[tail_idx] = __float2half(res);
    }
}

void launch_softmax_fp16(const void* input, void* output, int rows, int cols) {
    // 针对每个Row启动一个Block
    int block_size = 256;
    // 根据cols大小调整block_size通常更好，但此处固定256以匹配原逻辑
    if (cols < 256) block_size = 128;
    if (cols < 64) block_size = 32; // Warp size

    softmax_kernel_opt_ga100<<<rows, block_size>>>(
        (const __half*)input, 
        (__half*)output, 
        rows, cols
    );
}
//[8192x8192]：0.467 ms