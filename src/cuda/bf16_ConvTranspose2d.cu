#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>

__global__ void conv_transpose2d_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    int B, int C_in, int H_in, int W_in, 
    int C_out, int K_H, int K_W, 
    int H_out, int W_out,
    int stride_h, int stride_w, 
    int pad_h, int pad_w, 
    int dil_h, int dil_w) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * C_out * H_out * W_out;

    if (idx < total_elements) {
        int w_out = idx % W_out;
        int tmp = idx / W_out;
        int h_out = tmp % H_out;
        tmp = tmp / H_out;
        int c_out = tmp % C_out;
        int b = tmp / C_out;

        // Accumulate in float
        float sum = (bias != nullptr) ? __bfloat162float(bias[c_out]) : 0.0f;

        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int k_h = 0; k_h < K_H; ++k_h) {
                for (int k_w = 0; k_w < K_W; ++k_w) {
                    
                    int h_in_scaled = h_out + pad_h - k_h * dil_h;
                    int w_in_scaled = w_out + pad_w - k_w * dil_w;

                    if (h_in_scaled >= 0 && w_in_scaled >= 0 &&
                        h_in_scaled % stride_h == 0 && w_in_scaled % stride_w == 0) {
                        
                        int h_in = h_in_scaled / stride_h;
                        int w_in = w_in_scaled / stride_w;

                        if (h_in < H_in && w_in < W_in) {
                            int in_idx = ((b * C_in + c_in) * H_in + h_in) * W_in + w_in;
                            int w_idx = ((c_in * C_out + c_out) * K_H + k_h) * K_W + k_w;
                            
                            float val_in = __bfloat162float(input[in_idx]);
                            float val_w = __bfloat162float(weight[w_idx]);
                            sum += val_in * val_w;
                        }
                    }
                }
            }
        }
        output[idx] = __float2bfloat16(sum);
    }
}

void launch_conv_transpose2d_bf16(const void* input, const void* weight, const void* bias, void* output,
    int B, int C_in, int H_in, int W_in, int C_out, int K_H, int K_W, int H_out, int W_out,
    int stride_h, int stride_w, int pad_h, int pad_w, int out_pad_h, int out_pad_w, int dil_h, int dil_w) {

    int total_elements = B * C_out * H_out * W_out;
    int blockSize = 256;
    int gridSize = (total_elements + blockSize - 1) / blockSize;

    conv_transpose2d_bf16_kernel<<<gridSize, blockSize>>>(
        (const __nv_bfloat16*)input, (const __nv_bfloat16*)weight, (const __nv_bfloat16*)bias, (__nv_bfloat16*)output,
        B, C_in, H_in, W_in, C_out, K_H, K_W, H_out, W_out,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w
    );
}