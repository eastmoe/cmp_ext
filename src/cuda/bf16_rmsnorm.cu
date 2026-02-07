#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// 辅助：bf16 转 float (保持内联)
__device__ inline float bf162float(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

// 辅助：float 转 bf16 (保持内联)
__device__ inline __nv_bfloat16 float2bf16(float val) {
    return __float2bfloat16(val);
}

// 辅助：向量类型转换 union，用于 128-bit 加载时的类型双关
union VectorBf16 {
    int4 vec;
    __nv_bfloat16 vals[8];
};

// Warp Reduce Sum: 使用 __fadd_rn 替代默认加法
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = __fadd_rn(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Block Reduce Sum: 使用 __fadd_rn 替代默认加法
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

__global__ void rmsnorm_bf16_kernel(__nv_bfloat16* output, const __nv_bfloat16* input, const __nv_bfloat16* weight, int cols, float eps) {
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // 计算当前行的输入输出指针
    const __nv_bfloat16* row_in = input + row_idx * cols;
    __nv_bfloat16* row_out = output + row_idx * cols;

    float sum_sq = 0.0f;
    int idx = tid * 8; // 每个线程处理 8 个元素 (128 bit)
    int stride = blockDim.x * 8;

    // 阶段 1: 计算平方和
    // 向量化部分
    int i = idx;
    for (; i + 7 < cols; i += stride) {
        // 使用 int4 加载 128位 (8个 bf16)
        // 注意：此处假设 row_in 起始地址通常是 16 字节对齐的
        VectorBf16 v_in;
        v_in.vec = *reinterpret_cast<const int4*>(&row_in[i]);

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float x = bf162float(v_in.vals[k]);
            // 严禁 FMA: sum += x * x -> sum = add(sum, mul(x, x))
            float prod = __fmul_rn(x, x);
            sum_sq = __fadd_rn(sum_sq, prod);
        }
    }
    // 标量尾部处理
    for (; i < cols; i++) {
        float x = bf162float(row_in[i]);
        float prod = __fmul_rn(x, x);
        sum_sq = __fadd_rn(sum_sq, prod);
    }

    // 规约求和
    sum_sq = blockReduceSum(sum_sq);
    
    __shared__ float inv_rms;
    if (tid == 0) {
        // inv_rms = 1 / sqrt(mean + eps)
        // 约束：除法必须用 __fdividef，sqrt 必须用 rsqrt+mul (这里直接求逆平方根，即 rsqrt)
        // 逻辑：Mean = sum / cols
        float mean = __fdividef(sum_sq, (float)cols);
        float tmp = __fadd_rn(mean, eps);
        // 使用 intrinsics rsqrt
        inv_rms = __frsqrt_rn(tmp);
    }
    __syncthreads();

    float local_inv_rms = inv_rms; // 读取到寄存器

    // 阶段 2: 归一化并缩放
    // 向量化部分
    i = idx;
    for (; i + 7 < cols; i += stride) {
        VectorBf16 v_in, v_w, v_out;
        
        // 向量加载
        v_in.vec = *reinterpret_cast<const int4*>(&row_in[i]);
        v_w.vec = *reinterpret_cast<const int4*>(&weight[i]);

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float x = bf162float(v_in.vals[k]);
            float w = bf162float(v_w.vals[k]);
            
            // output = (x * inv_rms) * w
            // 严禁 FMA，显式拆分
            float norm = __fmul_rn(x, local_inv_rms);
            float final_val = __fmul_rn(norm, w);
            
            v_out.vals[k] = float2bf16(final_val);
        }
        
        // 向量存储
        *reinterpret_cast<int4*>(&row_out[i]) = v_out.vec;
    }
    // 标量尾部处理
    for (; i < cols; i++) {
        float x = bf162float(row_in[i]);
        float w = bf162float(weight[i]);
        
        float norm = __fmul_rn(x, local_inv_rms);
        float final_val = __fmul_rn(norm, w);
        
        row_out[i] = float2bf16(final_val);
    }
}

void launch_rmsnorm_bf16(void* output, const void* input, const void* weight, int rows, int cols, float eps) {
    // 保持原来的线程块策略，但建议根据寄存器压力调整
    int threads = 256;
    if (cols > 256) threads = 1024; // 对于 GA100，大 Block 有助于隐藏延迟
    
    // 计算需要的 shared memory 大小（如果有动态 shared memory，此处没有）
    rmsnorm_bf16_kernel<<<rows, threads>>>(
        (__nv_bfloat16*)output, (const __nv_bfloat16*)input, (const __nv_bfloat16*)weight, cols, eps
    );
}
//Op: RMSNorm   [In=64x128x1024, Norm=(1024,)]
//  Torch         :    0.163 ms, Avg Power:  97.43 W
//  Custom        :    0.108 ms, Avg Power:  97.43 W
//  Speedup       :      1.5 x
