#include <cuda_runtime.h>
#include <algorithm>

__global__ void upsample_nearest_fp32_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int C, int H_in, int W_in, int H_out, int W_out) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * C * H_out * W_out;

    if (index < total_elements) {
        // 计算输出坐标 (n, c, h_out, w_out)
        int w_out_idx = index % W_out;
        int h_out_idx = (index / W_out) % H_out;
        int c_idx     = (index / (W_out * H_out)) % C;
        int b_idx     = index / (W_out * H_out * C);

        // 映射到输入坐标 (最近邻)
        // src = floor(dst * scale)
        const float height_scale = (float)H_in / (float)H_out;
        const float width_scale = (float)W_in / (float)W_out;

        // 使用 min 确保不越界 (虽然理论上 floor 不会越界，但在边界值时浮点精度需小心)
        int h_in_idx = min((int)(h_out_idx * height_scale), H_in - 1);
        int w_in_idx = min((int)(w_out_idx * width_scale), W_in - 1);

        // 计算输入线性索引
        int input_offset = ((b_idx * C + c_idx) * H_in + h_in_idx) * W_in + w_in_idx;

        output[index] = input[input_offset];
    }
}

void launch_upsample_nearest_fp32(const float* input, float* output, int B, int C, int H_in, int W_in, int H_out, int W_out) {
    int total_elements = B * C * H_out * W_out;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    upsample_nearest_fp32_kernel<<<blocks, threads>>>(input, output, B, C, H_in, W_in, H_out, W_out);
}