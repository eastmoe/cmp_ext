#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>


#define DIV_CEIL(a, b) (((a) + (b) - 1) / (b))

// -------------------------------------------------------------------------
// 优化版 Kernel: Ampere 1x2 Spatial + 8 Channel Tiling
//[N=32, C=128, 128x128]：20ms 151W
// 1. Thread Coarsening: 每个线程计算 2 个水平相邻像素 (W) x 8 个输出通道 (C_out)。
//    这使得权重加载的开销分摊到了 2 个像素上，提升了计算密度。
// 2. Pointer Hoisting: 预先计算 8 个通道的权重指针，避免内层循环的乘法。
// -------------------------------------------------------------------------
__global__ void __launch_bounds__(256) conv2d_fp32_kernel_ampere_opt_v2(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in, int H_in, int W_in,
    int C_out, int K_H, int K_W,
    int H_out, int W_out,
    int s_h, int s_w, int p_h, int p_w, int d_h, int d_w
) {
    // ----------------------------------------------------------------
    // 1. 坐标计算 (Thread Mapping)
    // ----------------------------------------------------------------
    int tid_x = threadIdx.x; // 0..15
    int tid_y = threadIdx.y; // 0..15

    // 注意：Grid X 现在是基于 "Tile 宽度 32" 计算的，而不是 16
    // 每个 Block 处理 16行 x 32列 的输出区域
    int blocks_in_row = DIV_CEIL(W_out, 32); 
    
    int blk_w = blockIdx.x % blocks_in_row;
    int blk_h = blockIdx.x / blocks_in_row;

    // 计算当前线程负责的 2 个输出像素的 W 坐标
    // 线程 0 处理 w=0, w=1; 线程 1 处理 w=2, w=3 ... (利用局部性)
    int w_out_0 = blk_w * 32 + tid_x * 2;
    int w_out_1 = w_out_0 + 1;
    
    int h_out = blk_h * 16 + tid_y;
    int c_out_base = blockIdx.y * 8;
    int b_idx = blockIdx.z;

    // 只要 h_out 越界，整个线程都无事可做
    if (h_out >= H_out) return;

    // 检查 w_out 是否在范围内
    bool valid_w0 = (w_out_0 < W_out);
    bool valid_w1 = (w_out_1 < W_out);

    // 如果两个像素都越界，退出
    if (!valid_w0 && !valid_w1) return;

    // ----------------------------------------------------------------
    // 2. 准备累加器和指针
    // ----------------------------------------------------------------
    // 2 个像素，每个像素 8 个通道 -> 16 个寄存器
    float sum0[8] = {0.0f};
    float sum1[8] = {0.0f};

    // 输入基地址
    int h_in_base = h_out * s_h - p_h;
    int w_in_base_0 = w_out_0 * s_w - p_w;
    int w_in_base_1 = w_out_1 * s_w - p_w; // 通常等于 w_in_base_0 + s_w

    long long input_batch_offset = (long long)b_idx * C_in * H_in * W_in;
    const float* input_base_ptr = input + input_batch_offset;

    // 权重指针提升 (Pointer Hoisting)
    // 预先计算 8 个输出通道对应的权重起始位置
    // 避免在内层循环做乘法
    const float* w_ptrs[8];
    int weight_stride_oc = C_in * K_H * K_W;
    
    // 我们使用掩码来处理 C_out 边界，避免分支跳转
    // 虽然读取可能会越界，但只要权重内存分配有少量padding或不跨页，通常安全。
    // 为了绝对严谨，这里计算有效性。
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        int c = c_out_base + k;
        // 如果通道越界，指向 weight[0] 避免非法访存，最后写回时会过滤
        w_ptrs[k] = (c < C_out) ? (weight + c * weight_stride_oc) : weight; 
    }

    // ----------------------------------------------------------------
    // 3. 核心计算循环
    // ----------------------------------------------------------------
    for (int c = 0; c < C_in; ++c) {
        // 当前 Input Channel 的起始
        const float* current_in_channel = input_base_ptr + c * H_in * W_in;
        
        // 内层循环：卷积核
        for (int i = 0; i < K_H; ++i) {
            int in_row = h_in_base + i * d_h;
            bool row_valid = (in_row >= 0 && in_row < H_in);
            
            // 只有行有效时才计算 row_offset，否则设为 0 (配合列检查)
            int row_offset = row_valid ? (in_row * W_in) : 0;

            for (int j = 0; j < K_W; ++j) {
                // 加载 2 个 Input 像素
                float in_val0 = 0.0f;
                float in_val1 = 0.0f;

                if (row_valid) {
                    int in_col0 = w_in_base_0 + j * d_w;
                    int in_col1 = w_in_base_1 + j * d_w;

                    // 利用逻辑与的短路特性，或者编译器谓词优化
                    if (valid_w0 && in_col0 >= 0 && in_col0 < W_in) {
                        in_val0 = current_in_channel[row_offset + in_col0];
                    }
                    // 对 Pixel 1 做同样的检查
                    if (valid_w1 && in_col1 >= 0 && in_col1 < W_in) {
                        in_val1 = current_in_channel[row_offset + in_col1];
                    }
                }

                // 计算 & 累加
                // GA100 L1 Cache 会广播 w_val 给 Warp 中的所有线程
                // 关键点：这里 w_val 读取一次，被 2 次 FMA 使用 (Reuse)
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    // 直接使用指针解引用，然后自增
                    // 这比 `weight[base + i*KW + j]` 快得多
                    float w_val = *w_ptrs[k];
                    w_ptrs[k]++; // 指针推进到下一个 (i, j) 或下一个 c_in

                    // FMA 指令
                    sum0[k] += in_val0 * w_val;
                    sum1[k] += in_val1 * w_val;
                }
            }
        }
    }

    // ----------------------------------------------------------------
    // 4. 写回结果
    // ----------------------------------------------------------------
    long long out_batch_offset = (long long)b_idx * C_out * H_out * W_out;
    long long total_pixels = H_out * W_out;

    // Lambda 宏或者是内联写法处理写回
    auto write_back = [&](int w_curr, float* acc, bool w_valid) {
        if (!w_valid) return;
        long long out_spatial = (long long)h_out * W_out + w_curr;
        
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            int current_c_out = c_out_base + k;
            if (current_c_out < C_out) {
                float val = acc[k];
                if (bias) val += bias[current_c_out];
                
                long long out_addr = out_batch_offset + (long long)current_c_out * total_pixels + out_spatial;
                output[out_addr] = val;
            }
        }
    };

    write_back(w_out_0, sum0, valid_w0);
    write_back(w_out_1, sum1, valid_w1);
}

// -------------------------------------------------------------------------
// 入口函数 (保持名称和签名不变)
// -------------------------------------------------------------------------
void launch_conv2d_fp32(
    const float* input, const float* weight, const float* bias, float* output,
    int B, int C_in, int H_in, int W_in, 
    int C_out, int K_H, int K_W,
    int H_out, int W_out,
    int s_h, int s_w, int p_h, int p_w, int d_h, int d_w
) {
    // Block 尺寸保持 16x16 (256 线程) 以保持高 Occupancy
    dim3 threads_per_block(16, 16);
    
    // Grid 计算变化：
    // 每个 Block 的宽度覆盖现在是 32 (16 threads * 2 pixels/thread)
    // 高度覆盖仍然是 16
    int blocks_w = DIV_CEIL(W_out, 32); // 注意这里变成了 32
    int blocks_h = DIV_CEIL(H_out, 16);
    int grid_x = blocks_w * blocks_h;
    
    // Grid Y: Output Channels (每 block 处理 8 个)
    int grid_y = DIV_CEIL(C_out, 8);
    
    // Grid Z: Batch
    int grid_z = B;

    dim3 blocks(grid_x, grid_y, grid_z);

    // 启动优化后的 Kernel
    conv2d_fp32_kernel_ampere_opt_v2<<<blocks, threads_per_block>>>(
        input, weight, bias, output,
        B, C_in, H_in, W_in, C_out, K_H, K_W, H_out, W_out,
        s_h, s_w, p_h, p_w, d_h, d_w
    );
}