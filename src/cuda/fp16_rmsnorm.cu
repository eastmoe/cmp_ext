#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 优化后的 Warp Reduce：严格禁止 FP32 FMA，使用 __fadd_rn
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = __fadd_rn(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// 优化后的 Block Reduce：严格禁止 FP32 FMA
__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__global__ void rmsnorm_fp16_kernel(half* output, const half* input, const half* weight, int cols, float eps) {
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    const half* row_in = input + row_idx * cols;
    half* row_out = output + row_idx * cols;

    float sum_sq = 0.0f;

    // GA100极限优化：使用 128-bit (float4) 加载，一次处理 8 个 half
    // 强制使用 float4 指针访问全局内存以生成 LDG.128 指令
    int vec_limit = (cols / 8) * 8;

    // --- 阶段 1: 计算平方和 ---
    for (int i = tid * 8; i < vec_limit; i += blockDim.x * 8) {
        // 加载 128-bit 数据 (16字节)
        float4 in_data = *((const float4*)(row_in + i));
        
        // 重新解释为 4 个 half2
        const half2* h2_ptr = (const half2*)&in_data;

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            half2 val_h2 = h2_ptr[k];
            
            // 修复错误：手动拆分 half2 为两个 float，避开 __half2float2 的宏定义问题
            // 这些是底层指令，始终可用
            float f_low = __low2float(val_h2);
            float f_high = __high2float(val_h2);

            // 限制 1：不要用任何 FP32 FMA，拆分为 __fmul_rn 和 __fadd_rn
            float sq_x = __fmul_rn(f_low, f_low);
            float sq_y = __fmul_rn(f_high, f_high);
            
            // 累加
            sum_sq = __fadd_rn(sum_sq, __fadd_rn(sq_x, sq_y));
        }
    }

    // 处理剩余的尾部元素 (非8倍数部分)
    for (int i = vec_limit + tid; i < cols; i += blockDim.x) {
        float x = __half2float(row_in[i]);
        // 限制 1：拆分 FMA
        sum_sq = __fadd_rn(sum_sq, __fmul_rn(x, x));
    }

    // 块内规约
    sum_sq = blockReduceSum(sum_sq);
    
    __shared__ float inv_rms;
    if (tid == 0) {
        // 限制 7：不要使用 __frcp_rn。
        // 使用标准的 IEEE 除法 __fdiv_rn 计算均值
        float mean = __fdiv_rn(sum_sq, __int2float_rn(cols));
        // 使用 rsqrtf 计算平方根倒数 (FP32)
        inv_rms = rsqrtf(__fadd_rn(mean, eps));
    }
    __syncthreads();

    // 准备缩放因子，广播到 half2 以利用向量计算
    // __float2half2_rn 会将 float 转换为两个相同的 half 放入 half2
    half2 h2_inv_rms = __float2half2_rn(inv_rms);

    // --- 阶段 2: 归一化并输出 (使用 FP16 Vector Math) ---
    for (int i = tid * 8; i < vec_limit; i += blockDim.x * 8) {
        // 向量化加载 input 和 weight (128-bit load)
        float4 in_data = *((const float4*)(row_in + i));
        float4 wt_data = *((const float4*)(weight + i));
        float4 out_data;

        const half2* h2_in = (const half2*)&in_data;
        const half2* h2_wt = (const half2*)&wt_data;
        half2* h2_out = (half2*)&out_data;

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            // 限制 1：FP16 FMA/Mul 尽管用
            // 计算: x * w * inv_rms
            // 使用纯 FP16 向量指令 __hmul2，GA100 上吞吐量极大
            half2 x = h2_in[k];
            half2 w = h2_wt[k];
            
            half2 prod = __hmul2(x, w);
            h2_out[k] = __hmul2(prod, h2_inv_rms);
        }

        // 向量化存储 (128-bit store)
        *((float4*)(row_out + i)) = out_data;
    }

    // 处理尾部
    for (int i = vec_limit + tid; i < cols; i += blockDim.x) {
        half x = row_in[i];
        half w = weight[i];
        
        // Scalar FP16 math
        half prod = __hmul(x, w);
        // __low2half 从 half2 中提取标量 half 用于计算 (虽然 inv_rms 是标量，但此处在 half 域计算)
        // 或者直接转换: __float2half(inv_rms)
        row_out[i] = __hmul(prod, __float2half(inv_rms));
    }
}

void launch_rmsnorm_fp16(void* output, const void* input, const void* weight, int rows, int cols, float eps) {
    int threads = 256;
    if (cols > 256) threads = 1024;
    rmsnorm_fp16_kernel<<<rows, threads>>>(
        (half*)output, (const half*)input, (const half*)weight, cols, eps
    );
}