#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>

__global__ void upsample_nearest_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B, int C, int H_in, int W_in, int H_out, int W_out) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * C * H_out * W_out;

    if (index < total_elements) {
        int w_out_idx = index % W_out;
        int h_out_idx = (index / W_out) % H_out;
        int c_idx     = (index / (W_out * H_out)) % C;
        int b_idx     = index / (W_out * H_out * C);

        const float height_scale = (float)H_in / (float)H_out;
        const float width_scale = (float)W_in / (float)W_out;

        int h_in_idx = min((int)(h_out_idx * height_scale), H_in - 1);
        int w_in_idx = min((int)(w_out_idx * width_scale), W_in - 1);

        int input_offset = ((b_idx * C + c_idx) * H_in + h_in_idx) * W_in + w_in_idx;

        output[index] = input[input_offset];
    }
}

void launch_upsample_nearest_bf16(const void* input, void* output, int B, int C, int H_in, int W_in, int H_out, int W_out) {
    int total_elements = B * C * H_out * W_out;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    upsample_nearest_bf16_kernel<<<blocks, threads>>>(
        (const __nv_bfloat16*)input, 
        (__nv_bfloat16*)output, 
        B, C, H_in, W_in, H_out, W_out
    );
}