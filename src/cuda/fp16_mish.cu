#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 辅助函数：处理 2 个 half 元素 (packed in half2)
// 严格遵守所有约束：No FP32 FMA, No tanh, Use FP32 exp, Use FP16 vectorized RCP
__device__ __forceinline__ half2 mish_math_pair(half2 val) {
    // 阈值：当 x > 5.0f 时，Mish(x) 在 FP16 精度下等于 x。
    // 同时避免 E*E 在计算分母转换为 half 时溢出 (FP16 Max ~ 65504)
    const float THRESHOLD = 5.0f;

    // 1. 转为 float2 进行高精度计算 (约束: 为了计算精度转为 float)
    float2 x = __half22float2(val);

    // 为了避免 expf 溢出或分母过大导致 h2rcp 输入为 INF (导致结果为0)，
    // 我们在计算 exp 前 clamp 输入值。
    // 对于 > 5.0 的部分，最终结果会被覆盖为 x，所以这里的 clamp 不影响最终正确性，只保证中间计算数值稳定。
    float2 x_safe;
    x_safe.x = fminf(x.x, THRESHOLD);
    x_safe.y = fminf(x.y, THRESHOLD);

    // 2. 计算 Exp (约束4: 必须使用 FP32 __expf)
    float2 e;
    e.x = __expf(x_safe.x);
    e.y = __expf(x_safe.y);

    // 3. 展开 Tanh(Softplus(x))
    // Formula: tanh(ln(1+e^x)) = (e^x * (e^x + 2)) / (e^x * (e^x + 2) + 2)
    // Numerator N = E * (E + 2)
    // 约束1: 不要用 FP32 FMA，拆分为 mul_rn 和 add_rn
    float2 n;
    n.x = __fmul_rn(e.x, __fadd_rn(e.x, 2.0f));
    n.y = __fmul_rn(e.y, __fadd_rn(e.y, 2.0f));

    // Denominator D = N + 2
    float2 d;
    d.x = __fadd_rn(n.x, 2.0f);
    d.y = __fadd_rn(n.y, 2.0f);

    // 4. 计算倒数 (约束7: 必须转换为向量化的 FP16 使用 h2rcp)
    // 转回 half2
    half2 d_h2 = __floats2half2_rn(d.x, d.y);
    // 向量化倒数
    half2 r_h2 = h2rcp(d_h2); 
    // 转回 float2 继续后续乘法
    float2 r = __half22float2(r_h2);

    // 5. 组合结果 Result = x * (n * r)
    // 同样禁止 FMA
    float2 res;
    res.x = __fmul_rn(x_safe.x, __fmul_rn(n.x, r.x));
    res.y = __fmul_rn(x_safe.y, __fmul_rn(n.y, r.y));

    // 6. 处理 x > 5.0 的情况 (Res = x)
    // 使用三元运算符，编译器会优化为 selection 指令
    res.x = (x.x >= THRESHOLD) ? x.x : res.x;
    res.y = (x.y >= THRESHOLD) ? x.y : res.y;

    return __floats2half2_rn(res.x, res.y);
}

__global__ void mish_kernel_ga100_opt(const half* __restrict__ input, half* __restrict__ output, int n) {
    // 128-bit Vectorized Access (处理 int4 = 8 halves)
    int idx_vec = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_vec = blockDim.x * gridDim.x;
    
    // 强制转换为 int4 指针进行向量化加载
    const int4* in_vec_ptr = (const int4*)input;
    int4* out_vec_ptr = (int4*)output;
    
    int n_vec = n / 8; // int4 的数量

    // 主循环：每次处理 8 个 float16
    for (int i = idx_vec; i < n_vec; i += stride_vec) {
        int4 load_val = in_vec_ptr[i];
        int4 store_val;

        // 解包为 4 个 half2 并分别处理
        // 显式写出以增加指令级并行度 (ILP)
        half2 h0 = *reinterpret_cast<half2*>(&load_val.x);
        half2 h1 = *reinterpret_cast<half2*>(&load_val.y);
        half2 h2 = *reinterpret_cast<half2*>(&load_val.z);
        half2 h3 = *reinterpret_cast<half2*>(&load_val.w);

        h0 = mish_math_pair(h0);
        h1 = mish_math_pair(h1);
        h2 = mish_math_pair(h2);
        h3 = mish_math_pair(h3);

        *reinterpret_cast<half2*>(&store_val.x) = h0;
        *reinterpret_cast<half2*>(&store_val.y) = h1;
        *reinterpret_cast<half2*>(&store_val.z) = h2;
        *reinterpret_cast<half2*>(&store_val.w) = h3;

        out_vec_ptr[i] = store_val;
    }

    // 处理剩余元素 (Tail effect)
    // 即使是尾部元素，也要满足 h2rcp 的向量化约束
    int processed = n_vec * 8;
    int remaining = n - processed;
    int idx_scalar = processed + idx_vec * 8; // 注意：这里需要调整索引逻辑，简单起见我们让线程0处理尾部
    
    // 让一个线程块来处理尾部，或者简单的 Grid-Stride Scalar loop
    // 为了保持高性能和简洁，这里使用简单的标量循环，但内部凑成 half2 来满足 h2rcp 约束
    if (idx_vec == 0) { // 仅让第一个线程处理剩余（通常剩余 < 8 个，无需并行）
        for (int i = processed; i < n; i += 2) {
            half h_in0 = input[i];
            // 如果只剩1个，第2个填0或1均可，不影响第1个结果
            half h_in1 = (i + 1 < n) ? input[i + 1] : __float2half(0.0f);
            
            half2 h_packed = __halves2half2(h_in0, h_in1);
            half2 h_res = mish_math_pair(h_packed);
            
            output[i] = __low2half(h_res);
            if (i + 1 < n) {
                output[i + 1] = __high2half(h_res);
            }
        }
    }
}

void launch_mish_fp16(const void* input, void* output, int total_elements) {
    // 针对 GA100 优化 Block/Grid 设置
    // 每个线程处理 8 个元素，所以总线程数需求减少
    int vec_size = 8;
    int num_vec_elements = total_elements / vec_size;
    if (total_elements % vec_size != 0) num_vec_elements++;

    int threads = 256;
    int blocks = (num_vec_elements + threads - 1) / threads;

    // 限制最大 Blocks 数量以避免启动开销过大 (Tail effect handled inside)
    if (blocks > 65535) blocks = 65535;

    mish_kernel_ga100_opt<<<blocks, threads>>>(
        (const half*)input, 
        (half*)output, 
        total_elements
    );
}