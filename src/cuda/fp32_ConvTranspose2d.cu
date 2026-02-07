#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// -------------------------------------------------------------------------
// Kernel: Optimized Output-Centric ConvTranspose2D for GA100
// -------------------------------------------------------------------------
// Optimizations:
// 1. Loop Reordering: Moved C_in loop to innermost to remove coordinate math from hot path.
// 2. No-FMA enforcement: Uses __fmul_rn and __fadd_rn strictly.
// 3. Grid Strategy: 
//      Z: Batch (B)
//      Y: Output Channel (C_out)
//      X: Output Spatial (H_out * W_out) flattened
// -------------------------------------------------------------------------
__global__ void __launch_bounds__(256) conv_transpose2d_fp32_opt_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in, int H_in, int W_in, 
    int C_out, int K_H, int K_W, 
    int H_out, int W_out,
    int stride_h, int stride_w, 
    int pad_h, int pad_w, 
    int dil_h, int dil_w) {

    // 1. Calculate Output Coordinates
    // Grid: X=Spatial, Y=C_out, Z=Batch
    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int c_out = blockIdx.y;
    int b = blockIdx.z;

    int total_spatial = H_out * W_out;

    // Boundary check
    if (spatial_idx >= total_spatial || c_out >= C_out || b >= B) return;

    // Decode Spatial (h_out, w_out)
    // Using integer division, compliant with "No FP32 Reciprocal" rule.
    int h_out = spatial_idx / W_out;
    int w_out = spatial_idx % W_out;

    // 2. Initialize Accumulator
    // Load bias if present, otherwise 0.0
    float acc = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Pre-calculate pointers and strides to reduce integer math in inner loop
    // Input Input Layout: [B, C_in, H_in, W_in]
    // Weight Layout: [C_in, C_out, K_H, K_W] (Standard PyTorch ConvTranspose)
    
    // Base pointers specific to Batch and Output Channel
    // Input base moves with Batch. We will add C_in offset inside loop.
    const float* input_batch_base = input + (size_t)b * C_in * H_in * W_in;
    
    // Weight base moves with C_out. We will add C_in offset inside loop.
    // w_idx = ((c_in * C_out + c_out) * K_H + k_h) * K_W + k_w;
    // factor out c_in: c_in * (C_out * K_H * K_W) + (c_out * K_H * K_W + k_h * K_W + k_w)
    const float* weight_out_base = weight + (size_t)c_out * K_H * K_W;

    // Strides for the inner loop (Iterating over C_in)
    int input_cin_stride = H_in * W_in;
    int weight_cin_stride = C_out * K_H * K_W;

    // 3. Iterate over Kernel Spatial Dimensions (K_H, K_W)
    // Moving this loop OUTSIDE the C_in loop is the key optimization.
    // It allows us to compute (h_in, w_in) once per kernel position, avoiding
    // expensive mod/div/branching inside the reduction loop.
    for (int k_h = 0; k_h < K_H; ++k_h) {
        // Inverse coordinate mapping:
        // h_out = h_in * stride + k_h * dil - pad
        // => h_in * stride = h_out + pad - k_h * dil
        int h_in_scaled = h_out + pad_h - k_h * dil_h;
        
        // Check if vertical dimension maps to valid input
        if (h_in_scaled >= 0 && (h_in_scaled % stride_h == 0)) {
            int h_in = h_in_scaled / stride_h;
            
            if (h_in < H_in) {
                for (int k_w = 0; k_w < K_W; ++k_w) {
                    int w_in_scaled = w_out + pad_w - k_w * dil_w;

                    // Check if horizontal dimension maps to valid input
                    if (w_in_scaled >= 0 && (w_in_scaled % stride_w == 0)) {
                        int w_in = w_in_scaled / stride_w;

                        if (w_in < W_in) {
                            // Valid input pixel found at (h_in, w_in)
                            
                            // Pointer to the specific spatial location in Input and Weight
                            // input[b, 0, h_in, w_in]
                            const float* in_ptr = input_batch_base + (h_in * W_in + w_in);
                            
                            // weight[0, c_out, k_h, k_w]
                            const float* w_ptr = weight_out_base + (k_h * K_W + k_w);

                            // 4. Inner Loop: Reduction over Input Channels (C_in)
                            // This loop is now branch-free and memory/compute intensive.
                            // Pointers just increment by their respective strides.
                            for (int c_in = 0; c_in < C_in; ++c_in) {
                                float val_in = *in_ptr;
                                float val_w = *w_ptr;

                                // Constraint 1: STRICT NO FMA
                                // Force separate multiply and add
                                float prod = __fmul_rn(val_in, val_w);
                                acc = __fadd_rn(acc, prod);

                                // Advance pointers
                                in_ptr += input_cin_stride;
                                w_ptr += weight_cin_stride;
                            }
                        }
                    }
                }
            }
        }
    }

    // 5. Write Output
    // Output index: [b, c_out, h_out, w_out]
    // Since grid z=b, y=c_out, x=spatial, and we calculated spatial_idx...
    size_t out_idx = ((size_t)b * C_out * H_out * W_out) + 
                     ((size_t)c_out * H_out * W_out) + 
                     spatial_idx;
                     
    output[out_idx] = acc;
}

// Host Wrapper
void launch_conv_transpose2d_fp32(const float* input, const float* weight, const float* bias, float* output,
    int B, int C_in, int H_in, int W_in, int C_out, int K_H, int K_W, int H_out, int W_out,
    int stride_h, int stride_w, int pad_h, int pad_w, int out_pad_h, int out_pad_w, int dil_h, int dil_w) {

    // Total spatial pixels per channel per batch
    int total_spatial = H_out * W_out;
    
    // Block dimension: Standard occupancy tuning
    int blockSize = 256;
    
    // Grid Dimensions strategy:
    // X: Spatial Dimensions (Flattened H*W)
    // Y: Output Channels
    // Z: Batch Size
    dim3 gridDim;
    gridDim.x = (total_spatial + blockSize - 1) / blockSize;
    gridDim.y = C_out;
    gridDim.z = B;

    // Hardware Limit Check: Grid X dimension is limited to 2^31 - 1
    // If H_out*W_out is huge, standard grid logic might fail, 
    // but for typical feature maps this is fine.
    
    conv_transpose2d_fp32_opt_kernel<<<gridDim, blockSize>>>(
        input, weight, bias, output,
        B, C_in, H_in, W_in, C_out, K_H, K_W, H_out, W_out,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w
    );
    
    // Check for launch errors (optional but recommended)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}
//[N=64, Cin=64, Cout=64, 64x64, Stride=2] 23.118 ms, Avg Power: 157.41 W