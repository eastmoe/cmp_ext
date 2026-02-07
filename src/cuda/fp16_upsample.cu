#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <vector_types.h>
#include <cstdint>

using vec_t = int4; 
#define ALIGN_SIZE 8 

__global__ void upsample_nearest_fp16_kernel_optimized_v3(
    const half* __restrict__ input,
    half* __restrict__ output,
    int B, int C, int H_in, int W_in, int H_out, int W_out) 
{
    int total_rows = B * C * H_out;
    int row_idx = blockIdx.x * blockDim.y + threadIdx.y; // 如果你改变了 block 维度，这里要注意，但在 grid-stride loop 中通常只用 blockIdx.x

    const float height_scale = (float)H_in / (float)H_out;
    const float width_scale  = (float)W_in / (float)W_out;

    // 检查对齐：基地址16字节对齐 且 宽度是8的倍数
    bool is_aligned = ((uintptr_t)output % 16 == 0) && (W_out % 8 == 0);

    // Grid Stride Loop (基于 Block)
    for (int i = blockIdx.x; i < total_rows; i += gridDim.x) {
        
        // --- 1. 修复坐标解析逻辑 (Critical Fix) ---
        int temp = i;
        int h_out = temp % H_out;
        temp /= H_out;
        int c = temp % C;
        temp /= C;  // <--- 原代码缺少这行！导致 b 计算错误
        int b = temp;

        // 计算输入行的垂直坐标
        int h_in = min((int)(h_out * height_scale), H_in - 1);
        
        // 计算偏移量
        long long input_row_offset = ((long long)b * C + c) * H_in * W_in + (long long)h_in * W_in;
        long long output_row_offset = (long long)i * W_out;

        const half* in_ptr = input + input_row_offset;
        half* out_ptr = output + output_row_offset;

        if (is_aligned) {
            // [Fast Path] 向量化路径
            int stride = blockDim.x * ALIGN_SIZE;
            int w_out_base = threadIdx.x * ALIGN_SIZE;
            
            for (int w_out = w_out_base; w_out < W_out; w_out += stride) {
                half values[8];
                
                // 收集数据
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    int w_in = min((int)((w_out + k) * width_scale), W_in - 1);
                    values[k] = in_ptr[w_in]; 
                }

                // --- 2. 优化 Packing 逻辑 (更安全) ---
                // 将 8 个 half 组合成 4 个 half2，再转为 int4，避免指针强转的风险
                half2 h0 = __halves2half2(values[0], values[1]);
                half2 h1 = __halves2half2(values[2], values[3]);
                half2 h2 = __halves2half2(values[4], values[5]);
                half2 h3 = __halves2half2(values[6], values[7]);

                // 将 half2 的位重新解释为 int (sizeof(half2) == sizeof(int) == 4 bytes)
                // 使用 reinterpret_cast 是标准做法，或者使用 union
                int i0 = *(reinterpret_cast<int*>(&h0));
                int i1 = *(reinterpret_cast<int*>(&h1));
                int i2 = *(reinterpret_cast<int*>(&h2));
                int i3 = *(reinterpret_cast<int*>(&h3));

                vec_t packed = make_int4(i0, i1, i2, i3);

                // 128-bit 写入
                *(vec_t*)(out_ptr + w_out) = packed;
            }
        } else {
            // [Safe Path] 标量路径
            for (int w_out = threadIdx.x; w_out < W_out; w_out += blockDim.x) {
                int w_in = min((int)(w_out * width_scale), W_in - 1);
                out_ptr[w_out] = in_ptr[w_in];
            }
        }
    }
}

void launch_upsample_nearest_fp16(const void* input, void* output, int B, int C, int H_in, int W_in, int H_out, int W_out) {
    int total_rows = B * C * H_out;
    int threads = 256;
    int blocks = std::min((total_rows + threads - 1) / threads, 65535); 
    // 注意：这里的 blocks 计算其实不太重要，因为 kernel 内部是 grid-stride loop
    // 但为了利用率，通常取 total_rows 或 device limit 的较小值
    if (total_rows > 65535) blocks = 65535;
    if (blocks == 0) blocks = 1;

    upsample_nearest_fp16_kernel_optimized_v3<<<blocks, threads>>>(
        (const half*)input, 
        (half*)output, 
        B, C, H_in, W_in, H_out, W_out
    );
}