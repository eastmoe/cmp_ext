#include <cuda_runtime.h>
#include <cuda_bf16.h>

// 辅助函数：Warp内的规约求和
__device__ __forceinline__ void warpReduceSum(float& sum, float& sum_sq) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
        sum_sq = __fadd_rn(sum_sq, __shfl_down_sync(0xffffffff, sum_sq, offset));
    }
}

__global__ void GroupNormKernelBF16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
    int N, int C, int HxW, int groups, float eps) {

    // 计算当前Block处理的Group信息
    int n = blockIdx.x;
    int g = blockIdx.y;
    
    // 使用浮点除法替代整数除法逻辑（如果需要），但此处为索引计算，保持整数运算
    // C和groups通常可以整除，此处保持逻辑一致性
    int channels_per_group = C / groups;
    int num_elements_in_group = channels_per_group * HxW;
    int c_start = g * channels_per_group;

    // 共享内存：存储Mean和Var
    __shared__ float s_mem[2];
    if (threadIdx.x == 0) {
        s_mem[0] = 0.0f; 
        s_mem[1] = 0.0f;
    }
    __syncthreads();

    // -------------------------------------------------------
    // Pass 1: 计算 Sum 和 Sum_Sq
    // -------------------------------------------------------
    
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    // 计算当前batch和group的偏移
    size_t batch_offset = (size_t)n * C * HxW;
    size_t group_offset = (size_t)c_start * HxW;
    const __nv_bfloat16* input_ptr = input + batch_offset + group_offset;

    // 向量化配置：int4 = 16 bytes = 8 x BF16
    int lane_idx = threadIdx.x;
    int stride = blockDim.x;
    int vec_loop_end = num_elements_in_group & ~7; // 向下对齐到8

    // 向量化循环
    for (int i = lane_idx * 8; i < vec_loop_end; i += stride * 8) {
        // 使用int4加载128位数据 (8个BF16)
        // 注意：此处假设数据地址对齐到16字节。如果不对齐需回退到 __nv_bfloat162 或标量
        int4 v_raw = *reinterpret_cast<const int4*>(reinterpret_cast<const char*>(input_ptr) + i * sizeof(__nv_bfloat16));
        __nv_bfloat16* v_bf16 = reinterpret_cast<__nv_bfloat16*>(&v_raw);

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = __bfloat162float(v_bf16[k]);
            // 显式拆分FMA: sum += val
            local_sum = __fadd_rn(local_sum, val);
            // 显式拆分FMA: sum_sq += val * val
            local_sum_sq = __fadd_rn(local_sum_sq, __fmul_rn(val, val));
        }
    }

    // 处理剩余元素
    for (int i = vec_loop_end + lane_idx; i < num_elements_in_group; i += stride) {
        float val = __bfloat162float(input_ptr[i]);
        local_sum = __fadd_rn(local_sum, val);
        local_sum_sq = __fadd_rn(local_sum_sq, __fmul_rn(val, val));
    }

    // Warp级归约
    warpReduceSum(local_sum, local_sum_sq);

    // Block级归约 (仅由Warp的第一个线程写入共享内存)
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&s_mem[0], local_sum);
        atomicAdd(&s_mem[1], local_sum_sq);
    }
    __syncthreads();

    // -------------------------------------------------------
    // 计算统计量
    // -------------------------------------------------------
    
    // Rule 4: rcp必须使用除法 (__fdividef)
    float inv_N = __fdividef(1.0f, (float)num_elements_in_group);
    
    // Mean = Sum * (1/N)
    float mean = __fmul_rn(s_mem[0], inv_N);
    
    // Var = (SumSq * 1/N) - (Mean * Mean)
    float avg_sum_sq = __fmul_rn(s_mem[1], inv_N);
    float mean_sq = __fmul_rn(mean, mean);
    float var = __fadd_rn(avg_sum_sq, -mean_sq); // var = avg_sum_sq - mean_sq
    
    // Rule 5: sqrt必须使用rsqrt+mul替代 (此处直接计算rstd = 1/sqrt)
    // rstd = rsqrtf(max(var, 0) + eps)
    // 避免使用 max 标准库，使用 fmaxf 内联
    float val_for_rsqrt = __fadd_rn(fmaxf(var, 0.0f), eps);
    float rstd = rsqrtf(val_for_rsqrt);

    // -------------------------------------------------------
    // Pass 2: 归一化并写入
    // -------------------------------------------------------

    __nv_bfloat16* output_ptr = output + batch_offset + group_offset;

    // 向量化循环
    for (int i = lane_idx * 8; i < vec_loop_end; i += stride * 8) {
        int4 v_raw_in = *reinterpret_cast<const int4*>(reinterpret_cast<const char*>(input_ptr) + i * sizeof(__nv_bfloat16));
        __nv_bfloat16* v_bf16_in = reinterpret_cast<__nv_bfloat16*>(&v_raw_in);
        
        __nv_bfloat16 v_bf16_out[8];

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            int curr_idx = i + k;
            
            // 计算Gamma/Beta索引
            int c_local = curr_idx / HxW; 
            int c_global = c_start + c_local;

            float val = __bfloat162float(v_bf16_in[k]);
            
            // norm = (val - mean) * rstd
            float diff = __fadd_rn(val, -mean);
            float norm_val = __fmul_rn(diff, rstd);

            float g_val = 1.0f;
            float b_val = 0.0f;
            
            // 简单的加载，编译器会自动优化为 __ldg
            if (gamma) g_val = __bfloat162float(gamma[c_global]);
            if (beta)  b_val = __bfloat162float(beta[c_global]);

            // res = norm * gamma + beta (禁止FMA)
            float scaled = __fmul_rn(norm_val, g_val);
            float res = __fadd_rn(scaled, b_val);
            
            v_bf16_out[k] = __float2bfloat16(res);
        }

        // 向量化写入 (int4)
        *reinterpret_cast<int4*>(reinterpret_cast<char*>(output_ptr) + i * sizeof(__nv_bfloat16)) = *reinterpret_cast<int4*>(v_bf16_out);
    }

    // 处理剩余元素
    for (int i = vec_loop_end + lane_idx; i < num_elements_in_group; i += stride) {
        int c_local = i / HxW;
        int c_global = c_start + c_local;

        float val = __bfloat162float(input_ptr[i]);
        float diff = __fadd_rn(val, -mean);
        float norm_val = __fmul_rn(diff, rstd);

        float g_val = (gamma) ? __bfloat162float(gamma[c_global]) : 1.0f;
        float b_val = (beta) ? __bfloat162float(beta[c_global]) : 0.0f;

        float res = __fadd_rn(__fmul_rn(norm_val, g_val), b_val);
        output_ptr[i] = __float2bfloat16(res);
    }
}

void launch_groupnorm_bf16(void* output, const void* input, const void* weight, const void* bias, int N, int C, int HxW, int groups, float eps) {
    dim3 grid(N, groups);
    dim3 block(256); // 保持原有Block大小，可根据HxW调整
    
    // 动态共享内存：本Kernel使用静态大小(2个float)，但如果需要扩展可调整
    size_t smem_size = 0;

    GroupNormKernelBF16<<<grid, block, smem_size>>>(
        (__nv_bfloat16*)output, (const __nv_bfloat16*)input, (const __nv_bfloat16*)weight, (const __nv_bfloat16*)bias, 
        N, C, HxW, groups, eps
    );
}
//Op: GroupNorm [N=64, C=512, 128x128]
//  Torch         :   20.653 ms, Avg Power: 103.26 W
//  Custom        :    2.696 ms, Avg Power:  98.10 W
//  Speedup       :      7.7 x
