#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// 辅助函数：严格遵循禁用FP32 FMA的要求，显式拆分为mul + add
__device__ __forceinline__ float strict_fma_rn(float acc, float a, float b) {
    return __fadd_rn(acc, __fmul_rn(a, b));
}

// 辅助函数：遵循使用div替代rcp的要求
__device__ __forceinline__ float strict_div(float a, float b) {
    return __fdividef(a, b);
}

__global__ void conv2d_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    int B, int C_in, int H_in, int W_in,
    int C_out, int K_H, int K_W,
    int H_out, int W_out,
    int s_h, int s_w, int p_h, int p_w, int d_h, int d_w
) {
    // 向量化优化：每个线程处理2个相邻的输出像素
    // 原始的一维索引现在代表一个“像素对”
    int idx_pair = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 计算第一个像素的真实索引
    int idx0 = idx_pair * 2;
    int idx1 = idx0 + 1;

    int total_pixels = H_out * W_out;
    int total_threads_needed = B * C_out * total_pixels;

    if (idx0 >= total_threads_needed) return;

    // 预计算常量，减少除法开销
    int c_out_stride = total_pixels;
    int b_stride = total_pixels * C_out;

    // 解析 idx0 的坐标
    // 优化：idx1 紧邻 idx0，大概率共享 b, c_out, h_out
    int b_idx     = idx0 / b_stride;
    int rem_b     = idx0 % b_stride;
    int c_out_idx = rem_b / c_out_stride;
    int rem_c     = rem_b % c_out_stride;
    int h_out_idx = rem_c / W_out;
    int w_out_idx0 = rem_c % W_out;

    // 检查 idx1 是否有效以及是否在同一个计算平面(Batch/Channel)内
    // 如果跨越了Channel或Batch边界，权重无法共享，逻辑会变复杂。
    // 鉴于卷积特征图通常较大，边界情况极少，这里采用分支处理：
    // 大部分情况 idx1 就在 idx0 旁边 (w_out_idx + 1)
    bool has_second = (idx1 < total_threads_needed);
    
    // 简单的判断：如果 w_out_idx0 + 1 < W_out，说明 idx1 和 idx0 在同一行，完全共享权重和输入行基址
    bool continuous_row = (w_out_idx0 + 1 < W_out);

    // 初始化累加器 (FP32)
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // 基础偏移量预计算
    int input_batch_offset = b_idx * C_in * H_in * W_in;
    int weight_cout_offset = c_out_idx * C_in * K_H * K_W;

    // 为了性能，针对最常见的连续情况(continuous_row)进行优化路径
    if (continuous_row && has_second) {
        int w_out_idx1 = w_out_idx0 + 1;
        
        // 循环展开
        #pragma unroll 4
        for (int c = 0; c < C_in; ++c) {
            int in_c_offset = input_batch_offset + c * H_in * W_in;
            int w_c_offset  = weight_cout_offset + c * K_H * K_W;

            for (int i = 0; i < K_H; ++i) {
                // 行坐标相同
                int in_row = h_out_idx * s_h - p_h + i * d_h;
                
                // 行检查可在此处做一次
                if (in_row >= 0 && in_row < H_in) {
                    int in_row_offset = in_c_offset + in_row * W_in;
                    int w_row_offset = w_c_offset + i * K_W;

                    for (int j = 0; j < K_W; ++j) {
                        // 权重加载 (共享!) - GA100 L2优化关键
                        float w_val = __bfloat162float(weight[w_row_offset + j]);

                        // 输入列坐标
                        int in_col0 = w_out_idx0 * s_w - p_w + j * d_w;
                        int in_col1 = w_out_idx1 * s_w - p_w + j * d_w; 
                        // 注意：如果s_w=1, d_w=1，in_col1 就是 in_col0 + 1，理论上可向量化加载输入，
                        // 但由于padding和stride的不确定性，保持标量加载更安全。

                        // 像素0计算
                        if (in_col0 >= 0 && in_col0 < W_in) {
                            float val0 = __bfloat162float(input[in_row_offset + in_col0]);
                            acc0 = strict_fma_rn(acc0, val0, w_val);
                        }

                        // 像素1计算
                        if (in_col1 >= 0 && in_col1 < W_in) {
                            float val1 = __bfloat162float(input[in_row_offset + in_col1]);
                            acc1 = strict_fma_rn(acc1, val1, w_val);
                        }
                    }
                }
            }
        }
    } else {
        // 边界情况或单个尾部像素：退回到单像素处理逻辑，或者分别处理
        // 处理 idx0
        for (int c = 0; c < C_in; ++c) {
            for (int i = 0; i < K_H; ++i) {
                int in_row = h_out_idx * s_h - p_h + i * d_h;
                if (in_row >= 0 && in_row < H_in) {
                    for (int j = 0; j < K_W; ++j) {
                         int in_col = w_out_idx0 * s_w - p_w + j * d_w;
                         if (in_col >= 0 && in_col < W_in) {
                             int input_offset = ((b_idx * C_in + c) * H_in + in_row) * W_in + in_col;
                             int weight_offset = ((c_out_idx * C_in + c) * K_H + i) * K_W + j;
                             float w_val = __bfloat162float(weight[weight_offset]);
                             float in_val = __bfloat162float(input[input_offset]);
                             acc0 = strict_fma_rn(acc0, in_val, w_val);
                         }
                    }
                }
            }
        }

        // 处理 idx1 (如果存在)
        if (has_second) {
             // 重新计算 idx1 的坐标 (因为可能换行或换通道了)
            int b_idx1     = idx1 / b_stride;
            int rem_b1     = idx1 % b_stride;
            int c_out_idx1 = rem_b1 / c_out_stride;
            int rem_c1     = rem_b1 % c_out_stride;
            int h_out_idx1 = rem_c1 / W_out;
            int w_out_idx1 = rem_c1 % W_out;

            for (int c = 0; c < C_in; ++c) {
                for (int i = 0; i < K_H; ++i) {
                    int in_row = h_out_idx1 * s_h - p_h + i * d_h;
                    if (in_row >= 0 && in_row < H_in) {
                        for (int j = 0; j < K_W; ++j) {
                             int in_col = w_out_idx1 * s_w - p_w + j * d_w;
                             if (in_col >= 0 && in_col < W_in) {
                                 int input_offset = ((b_idx1 * C_in + c) * H_in + in_row) * W_in + in_col;
                                 int weight_offset = ((c_out_idx1 * C_in + c) * K_H + i) * K_W + j;
                                 float w_val = __bfloat162float(weight[weight_offset]);
                                 float in_val = __bfloat162float(input[input_offset]);
                                 acc1 = strict_fma_rn(acc1, in_val, w_val);
                             }
                        }
                    }
                }
            }
        }
    }

    // 处理 Bias
    if (bias != nullptr) {
        float b_val = __bfloat162float(bias[c_out_idx]);
        acc0 = __fadd_rn(acc0, b_val);
        
        // 注意：idx1 的 bias 可能不同，如果它跨越了 channel 边界
        if (has_second) {
             int c_out_idx1 = (idx1 / total_pixels) % C_out;
             float b_val1 = __bfloat162float(bias[c_out_idx1]);
             acc1 = __fadd_rn(acc1, b_val1);
        }
    }

    // 结果回写
    output[idx0] = __float2bfloat16(acc0);
    if (has_second) {
        output[idx1] = __float2bfloat16(acc1);
    }
}

void launch_conv2d_bf16(
    const void* input, const void* weight, const void* bias, void* output,
    int B, int C_in, int H_in, int W_in, 
    int C_out, int K_H, int K_W,
    int H_out, int W_out,
    int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w
) {
    int total_elements = B * C_out * H_out * W_out;
    
    // 优化：每个线程处理2个元素，减少Grid大小
    int threads = 256;
    int elements_per_thread = 2;
    // 向上取整计算需要的线程块
    int total_threads = (total_elements + elements_per_thread - 1) / elements_per_thread;
    int blocks = (total_threads + threads - 1) / threads;

    conv2d_bf16_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(input),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<const __nv_bfloat16*>(bias),
        reinterpret_cast<__nv_bfloat16*>(output),
        B, C_in, H_in, W_in, C_out, K_H, K_W, H_out, W_out,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w
    );
}
//[N=64, C=128, 128x128]：124.060 ms, Avg Power: 168.22 W 0.2x

