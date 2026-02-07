#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

// 辅助函数：将1D索引解算为4D坐标 (B, C, H, W)
// 使用 restrict 关键字和 inline 提高编译器优化能力
__device__ __forceinline__ void get_coords(
    int idx, 
    int W_out, int H_out, int C_out, 
    int* b, int* c, int* h, int* w) 
{
    *w = idx % W_out;
    int tmp = idx / W_out;
    *h = tmp % H_out;
    tmp = tmp / H_out;
    *c = tmp % C_out;
    *b = tmp / C_out;
}

__global__ void conv_transpose2d_fp16_ga100_opt_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int B, int C_in, int H_in, int W_in, 
    int C_out, int K_H, int K_W, 
    int H_out, int W_out,
    int stride_h, int stride_w, 
    int pad_h, int pad_w, 
    int dil_h, int dil_w) 
{
    // 每个线程处理 2 个 Output Channels
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 我们将 C_out 视为对处理，所以总任务数基于 C_out_pairs
    int C_out_pairs = (C_out + 1) / 2;
    int total_threads = B * C_out_pairs * H_out * W_out;

    if (idx >= total_threads) return;

    // 坐标解算
    int w_out, h_out, c_out_pair, b;
    int tmp = idx;
    w_out = tmp % W_out; tmp /= W_out;
    h_out = tmp % H_out; tmp /= H_out;
    c_out_pair = tmp % C_out_pairs; 
    b = tmp / C_out_pairs;

    int c_out_0 = c_out_pair * 2;
    int c_out_1 = c_out_0 + 1;
    bool has_second = (c_out_1 < C_out);

    // 初始化累加器 (全链路 FP16，使用 half 类型)
    // 使用 __float2half(0.0f) 确保初始化为 FP16 的零
    half sum0 = __float2half(0.0f);
    half sum1 = __float2half(0.0f);

    // 加载 Bias (直接使用 half，不转 float)
    if (bias != nullptr) {
        sum0 = bias[c_out_0];
        if (has_second) {
            sum1 = bias[c_out_1];
        }
    }

    // Input Base: 指向当前 Batch 的起始位置
    const half* input_b_ptr = input + (size_t)b * C_in * H_in * W_in;
    
    // Weight Stride
    // Weight 布局: [C_in, C_out, K_H, K_W]
    int weight_stride_cin = C_out * K_H * K_W;
    int weight_stride_cout = K_H * K_W;

    // 循环 C_in
    for (int c_in = 0; c_in < C_in; ++c_in) {
        
        // 当前 channel 的 input 指针
        const half* input_c_ptr = input_b_ptr + c_in * (H_in * W_in);
        
        // 当前 c_in 对应的 weight 起始位置
        const half* weight_cin_ptr = weight + c_in * weight_stride_cin;

        // 循环 Kernel (K_H, K_W)
        for (int k_h = 0; k_h < K_H; ++k_h) {
            // 预计算 h_in 的缩放值
            int h_in_scaled = h_out + pad_h - k_h * dil_h;
            
            // 快速检查 h 维度是否有效
            if (h_in_scaled >= 0 && (h_in_scaled % stride_h == 0)) {
                int h_in = h_in_scaled / stride_h;
                if (h_in < H_in) {
                    
                    // 优化：计算 input 行指针
                    const half* input_row_ptr = input_c_ptr + h_in * W_in;
                    
                    for (int k_w = 0; k_w < K_W; ++k_w) {
                        int w_in_scaled = w_out + pad_w - k_w * dil_w;
                        
                        // 快速检查 w 维度
                        if (w_in_scaled >= 0 && (w_in_scaled % stride_w == 0)) {
                            int w_in = w_in_scaled / stride_w;
                            if (w_in < W_in) {
                                // -------------------------------------------------
                                // 核心计算区域 (全 FP16)
                                // -------------------------------------------------
                                
                                // 1. 读取 Input
                                // 使用 __ldg 读取 half 数据
                                half val_in = __ldg(&input_row_ptr[w_in]);

                                // 2. 计算权重偏移
                                int k_offset = k_h * K_W + k_w;
                                
                                // 读取 Weight 0
                                half w0 = __ldg(&weight_cin_ptr[c_out_0 * weight_stride_cout + k_offset]);

                                // 计算 0: 使用 FP16 FMA (__hfma)
                                // sum0 = val_in * w0 + sum0
                                // 满足约束1：使用硬件级 FP16 FMA
                                sum0 = __hfma(val_in, w0, sum0);

                                // 如果有第二个通道，读取 Weight 1 并计算
                                if (has_second) {
                                    half w1 = __ldg(&weight_cin_ptr[c_out_1 * weight_stride_cout + k_offset]);
                                    
                                    // 计算 1: 使用 FP16 FMA
                                    sum1 = __hfma(val_in, w1, sum1);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 写回结果 (直接写入 half)
    size_t out_offset_base = ((size_t)b * C_out * H_out + h_out) * W_out + w_out;
    size_t plane_stride = H_out * W_out;

    output[out_offset_base + c_out_0 * plane_stride] = sum0;
    
    if (has_second) {
        output[out_offset_base + c_out_1 * plane_stride] = sum1;
    }
}

void launch_conv_transpose2d_fp16(const void* input, const void* weight, const void* bias, void* output,
    int B, int C_in, int H_in, int W_in, int C_out, int K_H, int K_W, int H_out, int W_out,
    int stride_h, int stride_w, int pad_h, int pad_w, int out_pad_h, int out_pad_w, int dil_h, int dil_w) {

    // 计算总线程数：按照 (C_out + 1) / 2 进行成对处理
    int C_out_pairs = (C_out + 1) / 2;
    int total_elements = B * C_out_pairs * H_out * W_out;
    
    int blockSize = 256;
    int gridSize = (total_elements + blockSize - 1) / blockSize;

    conv_transpose2d_fp16_ga100_opt_kernel<<<gridSize, blockSize>>>(
        (const half*)input, (const half*)weight, (const half*)bias, (half*)output,
        B, C_in, H_in, W_in, C_out, K_H, K_W, H_out, W_out,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w
    );
}