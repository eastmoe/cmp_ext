import torch
import torch.nn.functional as F
import time
import os
import threading
import pynvml
# 导入自定义算子
import cmpext3

# -----------------------------------------------------------
# 2. 全局设置
# -----------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
print(f"[Info] TF32 disabled: Matmul={torch.backends.cuda.matmul.allow_tf32}, CuDNN={torch.backends.cudnn.allow_tf32}")

device = torch.device("cuda")
os.environ["USE_TENSOR_CORE"] = "0"

# -----------------------------------------------------------
# 功耗监控辅助类
# -----------------------------------------------------------
class PowerMonitor:
    def __init__(self, device_index=0, interval=0.01):
        self.device_index = device_index
        self.interval = interval
        self.stop_event = threading.Event()
        self.power_readings = []
        self.thread = None
        try:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except pynvml.NVMLError:
            self.handle = None
            print("[Warning] Could not get NVML handle. Power monitoring disabled.")

    def _monitor_loop(self):
        while not self.stop_event.is_set():
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self.power_readings.append(power_mw / 1000.0) 
            except pynvml.NVMLError:
                pass
            time.sleep(self.interval)

    def start(self):
        if self.handle is None: return
        self.stop_event.clear()
        self.power_readings = []
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()

    def stop(self):
        if self.handle is None: return
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
    
    def get_avg_power(self):
        if not self.power_readings:
            return 0.0
        return sum(self.power_readings) / len(self.power_readings)


# -----------------------------------------------------------
# 3. 基准测试工具函数
# -----------------------------------------------------------
def benchmark_op(name, torch_func, custom_func, args, n_warmup=10, n_iter=50, power_monitor=None):
    # 1. Warmup
    for _ in range(n_warmup):
        torch_func(*args)
        custom_func(*args)
    torch.cuda.synchronize()

    # 2. Measure Torch
    if power_monitor: power_monitor.start()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(n_iter):
        torch_func(*args)
    end_event.record()
    torch.cuda.synchronize()

    if power_monitor: power_monitor.stop()
    
    t_torch = start_event.elapsed_time(end_event) / n_iter 
    p_torch = power_monitor.get_avg_power() if power_monitor else 0.0

    # 3. Measure Custom
    if power_monitor: power_monitor.start()
    
    start_event.record()
    for _ in range(n_iter):
        custom_func(*args)
    end_event.record()
    torch.cuda.synchronize()
    
    if power_monitor: power_monitor.stop()
        
    t_custom = start_event.elapsed_time(end_event) / n_iter 
    p_custom = power_monitor.get_avg_power() if power_monitor else 0.0

    return t_torch, t_custom, p_torch, p_custom


def run_suite(dtype_name, dtype):
    print(f"\n{'='*20} Running Benchmarks for {dtype_name} {'='*20}")

    # 定义一个清理显存的辅助小函数
    def clean_memory(*tensors):
        for t in tensors:
            del t
        torch.cuda.empty_cache()
    
    # =======================================================
    # I. 数学函数 (Math Functions)
    # =======================================================
    print("--- [Section I: 数学函数 (Math Functions)] ---")
    
    # 1. Tanh
    N_elem = 8192
    input_act = torch.randn(N_elem, N_elem, device=device, dtype=dtype)

    t_torch, t_custom, p_torch, p_custom = benchmark_op(
        "Tanh",
        lambda x: torch.tanh(x),
        lambda x: cmpext3.tanh(x),
        (input_act,), n_iter=50, power_monitor=power_monitor
    )
    print(f"Op: Tanh [{N_elem}x{N_elem}]")
    print(f"  Torch         : {t_torch:8.3f} ms, Avg Power: {p_torch:6.2f} W")
    print(f"  Custom        : {t_custom:8.3f} ms, Avg Power: {p_custom:6.2f} W")
    print(f"  Speedup       : {t_torch / t_custom:8.1f} x")

    # 2. ERF
    t_torch, t_custom, p_torch, p_custom = benchmark_op(
        "ERF",
        lambda x: torch.erf(x),
        lambda x: cmpext3.erf(x),
        (input_act,), n_iter=50, power_monitor=power_monitor
    )
    print(f"Op: ERF [{N_elem}x{N_elem}]")
    print(f"  Torch         : {t_torch:8.3f} ms, Avg Power: {p_torch:6.2f} W")
    print(f"  Custom        : {t_custom:8.3f} ms, Avg Power: {p_custom:6.2f} W")
    print(f"  Speedup       : {t_torch / t_custom:8.1f} x")

    # Section I 结束，清理 input_act
    clean_memory(input_act)

    # =======================================================
    # II. 基本计算 (Basic Calculations)
    # =======================================================
    print("\n--- [Section II: 基本计算 (Basic Calculations)] ---")

    # 1. Linear
    M, K, N = 4096, 4096, 4096 
    input_tensor = torch.randn(M, K, device=device, dtype=dtype)
    weight = torch.randn(N, K, device=device, dtype=dtype) 
    bias = torch.randn(N, device=device, dtype=dtype)
    
    t_torch, t_custom, p_torch, p_custom = benchmark_op(
        "Linear",
        lambda i, w, b: F.linear(i, w, b),
        lambda i, w, b: cmpext3.linear(i, w, b),
        (input_tensor, weight, bias),
        power_monitor=power_monitor
    )
    print(f"Op: Linear [{M}x{K}] @ [{N}x{K}].T")
    print(f"  Torch         : {t_torch:8.3f} ms, Avg Power: {p_torch:6.2f} W")
    print(f"  Custom        : {t_custom:8.3f} ms, Avg Power: {p_custom:6.2f} W")
    print(f"  Speedup       : {t_torch / t_custom:8.1f} x")

    clean_memory(input_tensor, weight, bias) # 立即清理

    # 2. BMM [新增]
    B_bmm, M_bmm, K_bmm, N_bmm = 16, 1024, 1024, 1024
    input_bmm = torch.randn(B_bmm, M_bmm, K_bmm, device=device, dtype=dtype)
    mat2_bmm = torch.randn(B_bmm, K_bmm, N_bmm, device=device, dtype=dtype)

    t_torch, t_custom, p_torch, p_custom = benchmark_op(
        "BMM",
        lambda x, y: torch.bmm(x, y),
        lambda x, y: cmpext3.bmm(x, y),
        (input_bmm, mat2_bmm),
        power_monitor=power_monitor
    )
    print(f"Op: BMM [Batch={B_bmm}, {M_bmm}x{K_bmm} @ {K_bmm}x{N_bmm}]")
    print(f"  Torch         : {t_torch:8.3f} ms, Avg Power: {p_torch:6.2f} W")
    print(f"  Custom        : {t_custom:8.3f} ms, Avg Power: {p_custom:6.2f} W")
    print(f"  Speedup       : {t_torch / t_custom:8.1f} x")

    clean_memory(input_bmm, mat2_bmm) # 立即清理

    # 3. Conv2d
    N_b, C_in, H, W = 64, 128, 128, 128 
    C_out = 64
    k_size = 3
    stride, padding = 1, 1
    
    input_conv = torch.randn(N_b, C_in, H, W, device=device, dtype=dtype)
    weight_conv = torch.randn(C_out, C_in, k_size, k_size, device=device, dtype=dtype)
    bias_conv = torch.randn(C_out, device=device, dtype=dtype)

    t_torch, t_custom, p_torch, p_custom = benchmark_op(
        "Conv2d", 
        lambda x, w, b: F.conv2d(x, w, b, stride=stride, padding=padding), 
        lambda x, w, b: cmpext3.conv2d(x, w, b, stride, padding), 
        (input_conv, weight_conv, bias_conv), n_iter=20, power_monitor=power_monitor
    )
    print(f"Op: Conv2d [N={N_b}, C={C_in}, {H}x{W}]")
    print(f"  Torch         : {t_torch:8.3f} ms, Avg Power: {p_torch:6.2f} W")
    print(f"  Custom        : {t_custom:8.3f} ms, Avg Power: {p_custom:6.2f} W")
    print(f"  Speedup       : {t_torch / t_custom:8.1f} x")

    clean_memory(input_conv, weight_conv, bias_conv) # 立即清理


    # 4. ConvTranspose2d [New]
    # 设置参数: 通常用于上采样，这里模拟 2x 上采样
    N_t, C_in_t, H_t, W_t = 64, 64, 64, 64
    C_out_t = 64
    k_size_t = 3
    stride_t = 2
    padding_t = 1
    output_padding_t = 1
    dilation_t = 1
    groups_t = 1

    input_tconv = torch.randn(N_t, C_in_t, H_t, W_t, device=device, dtype=dtype)
    # 注意: ConvTranspose2d 权重形状为 (In, Out/Groups, K, K)
    weight_tconv = torch.randn(C_in_t, C_out_t // groups_t, k_size_t, k_size_t, device=device, dtype=dtype)
    bias_tconv = torch.randn(C_out_t, device=device, dtype=dtype)

    t_torch, t_custom, p_torch, p_custom = benchmark_op(
        "ConvTranspose2d",
        lambda x, w, b: F.conv_transpose2d(x, w, b, stride=stride_t, padding=padding_t, 
                                           output_padding=output_padding_t, dilation=dilation_t, groups=groups_t),
        lambda x, w, b: cmpext3.conv_transpose2d(x, w, b, stride_t, padding_t, 
                                                 output_padding_t, dilation_t, groups_t),
        (input_tconv, weight_tconv, bias_tconv), 
        n_iter=20, 
        power_monitor=power_monitor
    )
    print(f"Op: ConvTranspose2d [N={N_t}, Cin={C_in_t}, Cout={C_out_t}, {H_t}x{W_t}, Stride={stride_t}]")
    print(f"  Torch         : {t_torch:8.3f} ms, Avg Power: {p_torch:6.2f} W")
    print(f"  Custom        : {t_custom:8.3f} ms, Avg Power: {p_custom:6.2f} W")
    print(f"  Speedup       : {t_torch / t_custom:8.1f} x")

    clean_memory(input_tconv, weight_tconv, bias_tconv) # 立即清理


    # 5. Upsample Nearest [New]
    N_up, C_up, H_up, W_up = 4, 1280, 64, 64
    target_h, target_w = 256, 256 # 2x Upsample
    output_size_up = (target_h, target_w)

    input_upsample = torch.randn(N_up, C_up, H_up, W_up, device=device, dtype=dtype)

    t_torch, t_custom, p_torch, p_custom = benchmark_op(
        "UpsampleNearest",
        lambda x, s: F.interpolate(x, size=s, mode='nearest'),
        lambda x, s: cmpext3.upsample_scaling(x, s),
        (input_upsample, output_size_up),
        n_iter=20,
        power_monitor=power_monitor
    )
    print(f"Op: UpsampleNearest [N={N_up}, C={C_up}, {H_up}x{W_up} -> {target_h}x{target_w}]")
    print(f"  Torch         : {t_torch:8.3f} ms, Avg Power: {p_torch:6.2f} W")
    print(f"  Custom        : {t_custom:8.3f} ms, Avg Power: {p_custom:6.2f} W")
    print(f"  Speedup       : {t_torch / t_custom:8.1f} x")

    clean_memory(input_upsample) # 立即清理

    # 6. Attention
    B_attn, H_attn, L_attn, D_attn = 8, 8, 1024, 128
    sm_scale = 1.0 / (D_attn ** 0.5)

    q = torch.randn(B_attn, H_attn, L_attn, D_attn, device=device, dtype=dtype)
    k = torch.randn(B_attn, H_attn, L_attn, D_attn, device=device, dtype=dtype)
    v = torch.randn(B_attn, H_attn, L_attn, D_attn, device=device, dtype=dtype)

    t_torch, t_custom, p_torch, p_custom = benchmark_op(
        "Attention", 
        lambda q,k,v: F.scaled_dot_product_attention(q, k, v, scale=sm_scale), 
        lambda q,k,v: cmpext3.attention(q, k, v, scale=sm_scale), 
        (q, k, v), 
        n_iter=20, power_monitor=power_monitor
    )
    print(f"Op: Attention [B={B_attn}, H={H_attn}, L={L_attn}, D={D_attn}]")
    print(f"  Torch         : {t_torch:8.3f} ms, Avg Power: {p_torch:6.2f} W")
    print(f"  Custom        : {t_custom:8.3f} ms, Avg Power: {p_custom:6.2f} W")
    print(f"  Speedup       : {t_torch / t_custom:8.1f} x")

    clean_memory(q, k, v) # 立即清理


    # 7. Embedding
    # 词表大小与维度设置
    num_embeddings = 32000
    embedding_dim = 1024
    # 输入形状 [Batch, Seq_Len]
    bs_emb, seq_emb = 512, 1024
    
    # 注意: Embedding 的输入 input 必须是整数索引 (Long/Int)，不能是浮点数 (dtype)
    input_emb = torch.randint(0, num_embeddings, (bs_emb, seq_emb), device=device, dtype=torch.long)
    # 权重矩阵使用当前测试的浮点精度 (FP32/FP16/BF16)
    weight_emb = torch.randn(num_embeddings, embedding_dim, device=device, dtype=dtype)

    t_torch, t_custom, p_torch, p_custom = benchmark_op(
        "Embedding",
        lambda x, w: F.embedding(x, w),
        lambda x, w: cmpext3.embedding(x, w),
        (input_emb, weight_emb), 
        n_iter=50, 
        power_monitor=power_monitor
    )
    print(f"Op: Embedding [Vocab={num_embeddings}, Dim={embedding_dim}, Input=({bs_emb}, {seq_emb})]")
    print(f"  Torch         : {t_torch:8.3f} ms, Avg Power: {p_torch:6.2f} W")
    print(f"  Custom        : {t_custom:8.3f} ms, Avg Power: {p_custom:6.2f} W")
    print(f"  Speedup       : {t_torch / t_custom:8.1f} x")

    clean_memory(input_emb, weight_emb) # 立即清理

    # =======================================================
    # III. 激活函数 (Activation Functions)
    # =======================================================
    print("\n--- [Section III: 激活函数 (Activation Functions)] ---")

    # 重新生成 input_act (因为之前被删除了)
    input_act = torch.randn(N_elem, N_elem, device=device, dtype=dtype)

    # 1. GELU
    t_torch, t_custom, p_torch, p_custom = benchmark_op(
        "GELU",
        lambda x: F.gelu(x),
        lambda x: cmpext3.gelu(x),
        (input_act,), n_iter=50, power_monitor=power_monitor
    )
    print(f"Op: GELU [{N_elem}x{N_elem}]")
    print(f"  Torch         : {t_torch:8.3f} ms, Avg Power: {p_torch:6.2f} W")
    print(f"  Custom        : {t_custom:8.3f} ms, Avg Power: {p_custom:6.2f} W")
    print(f"  Speedup       : {t_torch / t_custom:8.1f} x")

    # 2. Softmax
    t_torch, t_custom, p_torch, p_custom = benchmark_op(
        "Softmax",
        lambda x: F.softmax(x, dim=-1),
        lambda x: cmpext3.softmax(x, dim=-1),
        (input_act,), n_iter=50, power_monitor=power_monitor
    )
    print(f"Op: Softmax [{N_elem}x{N_elem}]")
    print(f"  Torch         : {t_torch:8.3f} ms, Avg Power: {p_torch:6.2f} W")
    print(f"  Custom        : {t_custom:8.3f} ms, Avg Power: {p_custom:6.2f} W")
    print(f"  Speedup       : {t_torch / t_custom:8.1f} x")

    # 3. SiLU
    t_torch, t_custom, p_torch, p_custom = benchmark_op(
        "SiLU",
        lambda x: F.silu(x),
        lambda x: cmpext3.silu(x),
        (input_act,), n_iter=50, power_monitor=power_monitor
    )
    print(f"Op: SiLU [{N_elem}x{N_elem}]")
    print(f"  Torch         : {t_torch:8.3f} ms, Avg Power: {p_torch:6.2f} W")
    print(f"  Custom        : {t_custom:8.3f} ms, Avg Power: {p_custom:6.2f} W")
    print(f"  Speedup       : {t_torch / t_custom:8.1f} x")

    # 4. Swish
    beta_val = 10.0  # 定义 beta 值
    t_torch, t_custom, p_torch, p_custom = benchmark_op(
        "Swish",
        lambda x: x * torch.sigmoid(beta_val * x),  # Torch 原生实现
        lambda x: cmpext3.swish(x, beta_val),       # 自定义算子
        (input_act,), n_iter=50, power_monitor=power_monitor
    )
    print(f"Op: Swish [{N_elem}x{N_elem}] (beta={beta_val})")
    print(f"  Torch         : {t_torch:8.3f} ms, Avg Power: {p_torch:6.2f} W")
    print(f"  Custom        : {t_custom:8.3f} ms, Avg Power: {p_custom:6.2f} W")
    print(f"  Speedup       : {t_torch / t_custom:8.1f} x")

    # 5. Mish
    t_torch, t_custom, p_torch, p_custom = benchmark_op(
        "Mish",
        lambda x: F.mish(x),                        # Torch 原生实现
        lambda x: cmpext3.mish(x),                  # 自定义算子
        (input_act,), n_iter=50, power_monitor=power_monitor
    )
    print(f"Op: Mish [{N_elem}x{N_elem}]")
    print(f"  Torch         : {t_torch:8.3f} ms, Avg Power: {p_torch:6.2f} W")
    print(f"  Custom        : {t_custom:8.3f} ms, Avg Power: {p_custom:6.2f} W")
    print(f"  Speedup       : {t_torch / t_custom:8.1f} x")

    # 6. Softplus
    sp_beta = 1.0
    sp_threshold = 20.0
    t_torch, t_custom, p_torch, p_custom = benchmark_op(
        "Softplus",
        lambda x: F.softplus(x, beta=sp_beta, threshold=sp_threshold), # Torch 原生实现
        lambda x: cmpext3.softplus(x, sp_beta, sp_threshold),          # 自定义算子
        (input_act,), n_iter=50, power_monitor=power_monitor
    )
    print(f"Op: Softplus [{N_elem}x{N_elem}] (beta={sp_beta}, threshold={sp_threshold})")
    print(f"  Torch         : {t_torch:8.3f} ms, Avg Power: {p_torch:6.2f} W")
    print(f"  Custom        : {t_custom:8.3f} ms, Avg Power: {p_custom:6.2f} W")
    print(f"  Speedup       : {t_torch / t_custom:8.1f} x")


    # 7. Softsign
    t_torch, t_custom, p_torch, p_custom = benchmark_op(
        "Softsign",
        lambda x: F.softsign(x),                    # Torch 原生实现
        lambda x: cmpext3.softsign(x),              # 自定义算子
        (input_act,), n_iter=50, power_monitor=power_monitor
    )
    print(f"Op: Softsign [{N_elem}x{N_elem}]")
    print(f"  Torch         : {t_torch:8.3f} ms, Avg Power: {p_torch:6.2f} W")
    print(f"  Custom        : {t_custom:8.3f} ms, Avg Power: {p_custom:6.2f} W")
    print(f"  Speedup       : {t_torch / t_custom:8.1f} x")

    # 8. Softshrink
    lambd_val = 0.5
    t_torch, t_custom, p_torch, p_custom = benchmark_op(
        "Softshrink",
        lambda x: F.softshrink(x, lambd=lambd_val),     # Torch 原生实现
        lambda x: cmpext3.softshrink(x, lambd_val),     # 自定义算子
        (input_act,), n_iter=50, power_monitor=power_monitor
    )
    print(f"Op: Softshrink [{N_elem}x{N_elem}] (lambd={lambd_val})")
    print(f"  Torch         : {t_torch:8.3f} ms, Avg Power: {p_torch:6.2f} W")
    print(f"  Custom        : {t_custom:8.3f} ms, Avg Power: {p_custom:6.2f} W")
    print(f"  Speedup       : {t_torch / t_custom:8.1f} x")

    clean_memory(input_act) # 立即清理

    # =======================================================
    # IV. 归一化 (Normalization)
    # =======================================================
    print("\n--- [Section IV: 归一化 (Normalization)] ---")

    # 1. GroupNorm
    N_gn, C_gn, H_gn, W_gn = 64, 512, 128, 128
    groups = 32
    input_gn = torch.randn(N_gn, C_gn, H_gn, W_gn, device=device, dtype=dtype)
    weight_gn = torch.randn(C_gn, device=device, dtype=dtype)
    bias_gn = torch.randn(C_gn, device=device, dtype=dtype)

    t_torch, t_custom, p_torch, p_custom = benchmark_op(
        "GroupNorm",
        lambda x, w, b: F.group_norm(x, groups, w, b, eps=1e-5),
        lambda x, w, b: cmpext3.group_norm(x, groups, w, b, 1e-5),
        (input_gn, weight_gn, bias_gn), n_iter=20,
        power_monitor=power_monitor
    )
    print(f"Op: GroupNorm [N={N_gn}, C={C_gn}, {H_gn}x{W_gn}]")
    print(f"  Torch         : {t_torch:8.3f} ms, Avg Power: {p_torch:6.2f} W")
    print(f"  Custom        : {t_custom:8.3f} ms, Avg Power: {p_custom:6.2f} W")
    print(f"  Speedup       : {t_torch / t_custom:8.1f} x")

    clean_memory(input_gn, weight_gn, bias_gn) # 立即清理

    # 2. LayerNorm
    B_ln, L_ln, D_ln = 64, 128, 1024 
    input_ln = torch.randn(B_ln, L_ln, D_ln, device=device, dtype=dtype)
    normalized_shape = (D_ln,) 
    weight_ln = torch.randn(normalized_shape, device=device, dtype=dtype)
    bias_ln = torch.randn(normalized_shape, device=device, dtype=dtype)

    t_torch, t_custom, p_torch, p_custom = benchmark_op(
        "LayerNorm",
        lambda x, s, w, b: F.layer_norm(x, s, w, b, eps=1e-5),
        lambda x, s, w, b: cmpext3.layer_norm(x, s, w, b, 1e-5),
        (input_ln, normalized_shape, weight_ln, bias_ln),
        n_iter=50,
        power_monitor=power_monitor
    )
    print(f"Op: LayerNorm [In={B_ln}x{L_ln}x{D_ln}, Norm={normalized_shape}]")
    print(f"  Torch         : {t_torch:8.3f} ms, Avg Power: {p_torch:6.2f} W")
    print(f"  Custom        : {t_custom:8.3f} ms, Avg Power: {p_custom:6.2f} W")
    print(f"  Speedup       : {t_torch / t_custom:8.1f} x")

    clean_memory(input_ln, weight_ln, bias_ln) # 立即清理


    # 3. RMSNorm
    B_rms, L_rms, D_rms = 64, 128, 1024
    input_rms = torch.randn(B_rms, L_rms, D_rms, device=device, dtype=dtype)
    normalized_shape_rms = (D_rms,)
    weight_rms = torch.randn(normalized_shape_rms, device=device, dtype=dtype)

    # PyTorch 原生实现 (作为基准对比)
    # RMSNorm = x * weight * rsqrt(mean(x^2) + eps)
    def torch_rmsnorm_func(x, shape, weight, eps):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

    t_torch, t_custom, p_torch, p_custom = benchmark_op(
        "RMSNorm",
        lambda x, s, w: torch_rmsnorm_func(x, s, w, 1e-6),
        lambda x, s, w: cmpext3.rmsnorm(x, s, w, 1e-6),
        (input_rms, normalized_shape_rms, weight_rms),
        n_iter=50,
        power_monitor=power_monitor
    )
    print(f"Op: RMSNorm   [In={B_rms}x{L_rms}x{D_rms}, Norm={normalized_shape_rms}]")
    print(f"  Torch         : {t_torch:8.3f} ms, Avg Power: {p_torch:6.2f} W")
    print(f"  Custom        : {t_custom:8.3f} ms, Avg Power: {p_custom:6.2f} W")
    print(f"  Speedup       : {t_torch / t_custom:8.1f} x")

    clean_memory(input_rms, weight_rms) # 立即清理




    # 最后的兜底清理    
    torch.cuda.empty_cache()

# 初始化 NVML
try:
    pynvml.nvmlInit()
    power_monitor = PowerMonitor(device_index=0, interval=0.01)
except pynvml.NVMLError:
    print("[Warning] NVML Init failed. Running without power monitoring.")
    power_monitor = None

# -----------------------------------------------------------
# 4. 主程序
# -----------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark Script")
    parser.add_argument("--fp32", action="store_true", help="Run FP32 benchmarks only")
    parser.add_argument("--fp16", action="store_true", help="Run FP16 benchmarks only")
    parser.add_argument("--bf16", action="store_true", help="Run BF16 benchmarks only")
    args = parser.parse_args()

    # 如果没有指定任何参数，默认运行所有精度
    run_all = not (args.fp32 or args.fp16 or args.bf16)

    props = torch.cuda.get_device_properties(0)
    print(f"  Name:                {props.name}")
    print(f"  Compute Capability:  {props.major}.{props.minor}")
    
    # 1. FP32
    if run_all or args.fp32:
        run_suite("FP32", torch.float32)
    
    # 2. FP16
    if run_all or args.fp16:
        run_suite("FP16", torch.float16)
    
    # 3. BF16
    if run_all or args.bf16:
        if torch.cuda.is_bf16_supported():
            run_suite("BF16", torch.bfloat16)
        else:
            print("\n[Info] BF16 not supported on this device, skipping.")

    if power_monitor:
        pynvml.nvmlShutdown()
