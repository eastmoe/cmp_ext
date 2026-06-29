import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn.functional as F


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import cmpext3  # noqa: E402


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

DEFAULT_TOLERANCES = {
    "fp32": (2e-3, 2e-3),
    "fp16": (5e-2, 5e-2),
    "bf16": (8e-2, 8e-2),
}

STRICT_TOLERANCES = {
    "fp32": (0.0, 0.0),
    "fp16": (0.0, 0.0),
    "bf16": (0.0, 0.0),
}


@dataclass(frozen=True)
class AccuracyCase:
    op: str
    case: str
    runner: Callable[[torch.dtype, torch.device, torch.Generator], tuple[torch.Tensor, torch.Tensor]]
    dtypes: tuple[str, ...] = ("fp32", "fp16", "bf16")
    tolerances: dict[str, tuple[float, float]] | None = None


@dataclass
class CaseResult:
    dtype_name: str
    op: str
    case: str
    status: str
    max_abs: float | None = None
    max_rel: float | None = None
    mean_abs: float | None = None
    rmse: float | None = None
    finite_mismatch: int | None = None
    message: str = ""


def _randn(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    gen: torch.Generator,
    scale: float = 1.0,
) -> torch.Tensor:
    return (torch.randn(shape, device=device, generator=gen, dtype=torch.float32) * scale).to(dtype)


def _tensor_from_values(
    values: Iterable[float],
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(list(values), device=device, dtype=torch.float32).reshape(shape).to(dtype)


def _ideal_unary(
    x: torch.Tensor,
    func: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    return func(x.float()).to(x.dtype)


def _rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)


def _attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(q.dtype)


def _make_unary_case(
    op: str,
    case: str,
    values: Iterable[float],
    shape: tuple[int, ...],
    ref_func: Callable[[torch.Tensor], torch.Tensor],
    custom_func: Callable[[torch.Tensor], torch.Tensor],
) -> AccuracyCase:
    def runner(dtype: torch.dtype, device: torch.device, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        del gen
        x = _tensor_from_values(values, shape, dtype, device)
        return _ideal_unary(x, ref_func), custom_func(x)

    return AccuracyCase(op, case, runner)


def build_cases() -> list[AccuracyCase]:
    unary_values = [
        -20.0,
        -10.0,
        -5.0,
        -2.0,
        -1.0,
        -0.5,
        -1e-3,
        0.0,
        1e-3,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        0.125,
    ]
    erf_values = [-4.0, -3.0, -2.0, -1.0, -0.5, -1e-3, 0.0, 1e-3, 0.5, 1.0, 2.0, 3.0, 4.0, 0.25, -0.25, 1.5]
    softplus_values = [-30.0, -20.0, -5.0, -1.0, -1e-3, 0.0, 1e-3, 1.0, 5.0, 19.5, 20.0, 20.5, 30.0, 0.5, -0.5, 10.0]
    shrink_values = [-2.0, -0.5001, -0.5, -0.4999, -1e-3, 0.0, 1e-3, 0.4999, 0.5, 0.5001, 2.0, 1.0, -1.0, 0.25, -0.25, 3.0]

    cases: list[AccuracyCase] = [
        _make_unary_case("tanh", "boundary", unary_values, (4, 4), torch.tanh, cmpext3.tanh),
        _make_unary_case("erf", "boundary", erf_values, (4, 4), torch.erf, cmpext3.erf),
        _make_unary_case("gelu", "boundary", unary_values, (4, 4), F.gelu, cmpext3.gelu),
        _make_unary_case("silu", "boundary", unary_values, (4, 4), F.silu, cmpext3.silu),
        _make_unary_case("mish", "boundary", unary_values, (4, 4), F.mish, cmpext3.mish),
        _make_unary_case("softsign", "boundary", unary_values, (4, 4), F.softsign, cmpext3.softsign),
        _make_unary_case(
            "softplus",
            "threshold",
            softplus_values,
            (4, 4),
            lambda x: F.softplus(x, beta=1.0, threshold=20.0),
            lambda x: cmpext3.softplus(x, 1.0, 20.0),
        ),
        _make_unary_case(
            "softshrink",
            "lambda_boundary",
            shrink_values,
            (4, 4),
            lambda x: F.softshrink(x, lambd=0.5),
            lambda x: cmpext3.softshrink(x, 0.5),
        ),
    ]

    def swish_runner(dtype: torch.dtype, device: torch.device, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        del gen
        beta = 10.0
        x = _tensor_from_values(unary_values, (4, 4), dtype, device)
        ref = (x.float() * torch.sigmoid(beta * x.float())).to(dtype)
        return ref, cmpext3.swish(x, beta)

    cases.append(AccuracyCase("swish", "beta_10_boundary", swish_runner))

    def softmax_last_runner(dtype: torch.dtype, device: torch.device, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        del gen
        values = [
            -20.0,
            -1.0,
            0.0,
            20.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -5.0,
            -2.0,
            2.0,
            5.0,
            1e-3,
            -1e-3,
            0.5,
            -0.5,
        ]
        x = _tensor_from_values(values, (4, 4), dtype, device)
        return torch.softmax(x.float(), dim=-1).to(dtype), cmpext3.softmax(x, dim=-1)

    cases.append(AccuracyCase("softmax", "last_dim_boundary", softmax_last_runner))

    def softmax_mid_runner(dtype: torch.dtype, device: torch.device, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        x = _randn((2, 4, 4), dtype, device, gen, scale=3.0)
        return torch.softmax(x.float(), dim=1).to(dtype), cmpext3.softmax(x, dim=1)

    cases.append(AccuracyCase("softmax", "middle_dim_random", softmax_mid_runner))

    def linear_runner(dtype: torch.dtype, device: torch.device, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        x = _randn((2, 3, 8), dtype, device, gen, scale=0.5)
        w = _randn((8, 8), dtype, device, gen, scale=0.5)
        b = _randn((8,), dtype, device, gen, scale=0.25)
        ref = F.linear(x.float(), w.float(), b.float()).to(dtype)
        return ref, cmpext3.linear(x, w, b)

    cases.append(AccuracyCase("linear", "batched_bias_aligned8", linear_runner))

    def linear_no_bias_runner(dtype: torch.dtype, device: torch.device, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        x = _randn((3, 8), dtype, device, gen, scale=0.5)
        w = _randn((8, 8), dtype, device, gen, scale=0.5)
        ref = F.linear(x.float(), w.float(), None).to(dtype)
        return ref, cmpext3.linear(x, w, None)

    cases.append(AccuracyCase("linear", "no_bias", linear_no_bias_runner))

    def bmm_runner(dtype: torch.dtype, device: torch.device, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        x = _randn((2, 3, 8), dtype, device, gen, scale=0.5)
        y = _randn((2, 8, 8), dtype, device, gen, scale=0.5)
        ref = torch.bmm(x.float(), y.float()).to(dtype)
        return ref, cmpext3.bmm(x, y)

    cases.append(AccuracyCase("bmm", "small_aligned8", bmm_runner))

    def conv2d_runner(dtype: torch.dtype, device: torch.device, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        x = _randn((2, 3, 5, 6), dtype, device, gen, scale=0.5)
        w = _randn((4, 3, 3, 3), dtype, device, gen, scale=0.25)
        b = _randn((4,), dtype, device, gen, scale=0.1)
        ref = F.conv2d(x.float(), w.float(), b.float(), stride=1, padding=1).to(dtype)
        return ref, cmpext3.conv2d(x, w, b, 1, 1)

    cases.append(AccuracyCase("conv2d", "k3_padding_bias", conv2d_runner))

    def conv2d_1x1_runner(dtype: torch.dtype, device: torch.device, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        x = _randn((1, 8, 4, 4), dtype, device, gen, scale=0.5)
        w = _randn((8, 8, 1, 1), dtype, device, gen, scale=0.25)
        ref = F.conv2d(x.float(), w.float(), None, stride=1, padding=0).to(dtype)
        return ref, cmpext3.conv2d(x, w, None, 1, 0)

    cases.append(AccuracyCase("conv2d", "k1_matmul_path", conv2d_1x1_runner))

    def conv_transpose_runner(dtype: torch.dtype, device: torch.device, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        x = _randn((2, 3, 4, 5), dtype, device, gen, scale=0.5)
        w = _randn((3, 4, 3, 3), dtype, device, gen, scale=0.25)
        b = _randn((4,), dtype, device, gen, scale=0.1)
        ref = F.conv_transpose2d(x.float(), w.float(), b.float(), stride=2, padding=1, output_padding=1).to(dtype)
        return ref, cmpext3.conv_transpose2d(x, w, b, 2, 1, 1)

    cases.append(AccuracyCase("conv_transpose2d", "stride2_padding_bias", conv_transpose_runner))

    def upsample_scale_runner(dtype: torch.dtype, device: torch.device, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        x = _randn((1, 2, 3, 4), dtype, device, gen, scale=1.0)
        ref = F.interpolate(x.float(), scale_factor=2, mode="nearest").to(dtype)
        return ref, cmpext3.upsample_scaling(x, 2)

    cases.append(AccuracyCase("upsample_scaling", "scale_factor_2", upsample_scale_runner, tolerances=STRICT_TOLERANCES))

    def upsample_size_runner(dtype: torch.dtype, device: torch.device, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        x = _randn((1, 2, 3, 4), dtype, device, gen, scale=1.0)
        ref = F.interpolate(x.float(), size=(6, 8), mode="nearest").to(dtype)
        return ref, cmpext3.upsample_scaling(x, (6, 8))

    cases.append(AccuracyCase("upsample_scaling", "output_size_6x8", upsample_size_runner, tolerances=STRICT_TOLERANCES))

    def attention_runner(dtype: torch.dtype, device: torch.device, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        q = _randn((1, 1, 4, 128), dtype, device, gen, scale=0.25)
        k = _randn((1, 1, 4, 128), dtype, device, gen, scale=0.25)
        v = _randn((1, 1, 4, 128), dtype, device, gen, scale=0.25)
        scale = 1.0 / math.sqrt(q.size(-1))
        return _attention_ref(q, k, v, scale), cmpext3.attention(q, k, v, scale=scale)

    cases.append(AccuracyCase("attention", "small_manual_ref", attention_runner))

    def embedding_runner(dtype: torch.dtype, device: torch.device, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        weight = _randn((8, 8), dtype, device, gen, scale=0.5)
        weight[0].zero_()
        indices = torch.tensor([[0, 1, 7, 3], [2, 0, 4, 5]], device=device, dtype=torch.long)
        ref = F.embedding(indices, weight.float(), padding_idx=0).to(dtype)
        return ref, cmpext3.embedding(indices, weight, padding_idx=0)

    cases.append(AccuracyCase("embedding", "padding_idx_zeroed", embedding_runner, tolerances=STRICT_TOLERANCES))

    def group_norm_runner(dtype: torch.dtype, device: torch.device, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        x = _randn((2, 4, 4, 4), dtype, device, gen, scale=0.5).contiguous()
        w = _randn((4,), dtype, device, gen, scale=0.25)
        b = _randn((4,), dtype, device, gen, scale=0.1)
        ref = F.group_norm(x.float(), 2, w.float(), b.float(), eps=1e-5).to(dtype)
        return ref, cmpext3.group_norm(x, 2, w, b, 1e-5)

    cases.append(AccuracyCase("group_norm", "groups2_weight_bias", group_norm_runner))

    def layer_norm_runner(dtype: torch.dtype, device: torch.device, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        x = _randn((2, 3, 8), dtype, device, gen, scale=0.5).contiguous()
        w = _randn((8,), dtype, device, gen, scale=0.25)
        b = _randn((8,), dtype, device, gen, scale=0.1)
        ref = F.layer_norm(x.float(), (8,), w.float(), b.float(), eps=1e-5).to(dtype)
        return ref, cmpext3.layer_norm(x, (8,), w, b, 1e-5)

    cases.append(AccuracyCase("layer_norm", "last_dim_weight_bias", layer_norm_runner))

    def rmsnorm_runner(dtype: torch.dtype, device: torch.device, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        x = _randn((2, 3, 8), dtype, device, gen, scale=0.5).contiguous()
        w = _randn((8,), dtype, device, gen, scale=0.25)
        return _rmsnorm_ref(x, w, 1e-6), cmpext3.rmsnorm(x, (8,), w, 1e-6)

    cases.append(AccuracyCase("rmsnorm", "last_dim_weight", rmsnorm_runner))

    return cases


def compare_tensors(
    ref: torch.Tensor,
    out: torch.Tensor,
    atol: float,
    rtol: float,
) -> tuple[bool, float, float, float, float, int, str]:
    if ref.shape != out.shape:
        return False, math.inf, math.inf, math.inf, math.inf, 0, f"shape mismatch: ref={tuple(ref.shape)} out={tuple(out.shape)}"

    ref32 = ref.detach().to(torch.float32)
    out32 = out.detach().to(torch.float32)
    ref_finite = torch.isfinite(ref32)
    out_finite = torch.isfinite(out32)
    finite_mismatch = int((ref_finite != out_finite).sum().item())
    finite = ref_finite & out_finite

    if finite.any():
        diff = (out32[finite] - ref32[finite]).abs()
        denom = ref32[finite].abs().clamp_min(1e-12)
        max_abs = float(diff.max().item())
        max_rel = float((diff / denom).max().item())
        mean_abs = float(diff.mean().item())
        rmse = float(torch.sqrt(torch.mean(diff * diff)).item())
    else:
        max_abs = max_rel = mean_abs = rmse = 0.0

    within = bool(torch.all((out32[finite] - ref32[finite]).abs() <= (atol + rtol * ref32[finite].abs())).item()) if finite.any() else True
    ok = finite_mismatch == 0 and within
    return ok, max_abs, max_rel, mean_abs, rmse, finite_mismatch, ""


def run_case(
    case: AccuracyCase,
    dtype_name: str,
    device: torch.device,
    seed: int,
    atol_override: float | None,
    rtol_override: float | None,
) -> CaseResult:
    dtype = DTYPES[dtype_name]

    try:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        ref, out = case.runner(dtype, device, gen)
        torch.cuda.synchronize(device)
    except Exception as exc:
        return CaseResult(dtype_name, case.op, case.case, "ERROR", message=str(exc))

    tolerances = case.tolerances or DEFAULT_TOLERANCES
    atol, rtol = tolerances[dtype_name]
    if atol_override is not None:
        atol = atol_override
    if rtol_override is not None:
        rtol = rtol_override

    ok, max_abs, max_rel, mean_abs, rmse, finite_mismatch, message = compare_tensors(ref, out, atol, rtol)
    status = "PASS" if ok else "FAIL"
    if not message:
        message = f"tol(atol={atol:g}, rtol={rtol:g})"
    return CaseResult(dtype_name, case.op, case.case, status, max_abs, max_rel, mean_abs, rmse, finite_mismatch, message)


def is_cuda_context_error(message: str) -> bool:
    lowered = message.lower()
    return "cuda error:" in lowered or "device-side assert" in lowered or "illegal memory access" in lowered


def selected_dtypes(args: argparse.Namespace) -> list[str]:
    requested = []
    for name in ("fp32", "fp16", "bf16"):
        if getattr(args, name):
            requested.append(name)
    if not requested:
        requested = ["fp32", "fp16", "bf16"]
    if "bf16" in requested and not torch.cuda.is_bf16_supported():
        requested.remove("bf16")
        print("[SKIP] bf16: torch.cuda.is_bf16_supported() is False")
    return requested


def filter_cases(cases: list[AccuracyCase], ops_arg: str | None) -> list[AccuracyCase]:
    if not ops_arg:
        return cases
    wanted = {item.strip() for item in ops_arg.split(",") if item.strip()}
    return [case for case in cases if case.op in wanted]


def print_result(result: CaseResult) -> None:
    name = f"{result.dtype_name:<4} {result.op:<18} {result.case:<24}"
    if result.status in {"ERROR", "SKIP"}:
        print(f"[{result.status:<5}] {name} {result.message}")
        return
    print(
        f"[{result.status:<5}] {name} "
        f"max_abs={result.max_abs:.6g} max_rel={result.max_rel:.6g} "
        f"mean_abs={result.mean_abs:.6g} rmse={result.rmse:.6g} "
        f"finite_mismatch={result.finite_mismatch} {result.message}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Accuracy and boundary tests for cmpext3 operators.")
    parser.add_argument("--fp32", action="store_true", help="Run FP32 cases only, unless combined with other dtype flags.")
    parser.add_argument("--fp16", action="store_true", help="Run FP16 cases only, unless combined with other dtype flags.")
    parser.add_argument("--bf16", action="store_true", help="Run BF16 cases only, unless combined with other dtype flags.")
    parser.add_argument("--ops", type=str, default=None, help="Comma-separated operator filter, e.g. linear,softmax,tanh.")
    parser.add_argument("--list", action="store_true", help="List available cases and exit.")
    parser.add_argument("--seed", type=int, default=20260629, help="Random seed used for generated inputs.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--atol", type=float, default=None, help="Override absolute tolerance for all cases.")
    parser.add_argument("--rtol", type=float, default=None, help="Override relative tolerance for all cases.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed or errored case.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available; cmpext3 accuracy tests require CUDA tensors.")
        return 2

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    cases = filter_cases(build_cases(), args.ops)

    if args.list:
        for case in cases:
            print(f"{case.op:<18} {case.case}")
        return 0

    if not cases:
        print("[ERROR] No cases selected.")
        return 2

    dtypes = selected_dtypes(args)
    if not dtypes:
        print("[ERROR] No supported dtypes selected.")
        return 2

    props = torch.cuda.get_device_properties(device)
    print(f"[Info] Device: {props.name} (cc {props.major}.{props.minor})")
    print("[Info] Reference: float32 PyTorch expression rounded to the tested dtype")
    print(f"[Info] TF32 disabled: matmul={torch.backends.cuda.matmul.allow_tf32}, cudnn={torch.backends.cudnn.allow_tf32}")

    total = 0
    failed = 0
    skipped = 0

    for dtype_name in dtypes:
        for index, case in enumerate(cases):
            if dtype_name not in case.dtypes:
                skipped += 1
                print_result(CaseResult(dtype_name, case.op, case.case, "SKIP", message="dtype not enabled for this case"))
                continue
            total += 1
            result = run_case(case, dtype_name, device, args.seed + index, args.atol, args.rtol)
            print_result(result)
            if result.status != "PASS":
                failed += 1
                if result.status == "ERROR" and is_cuda_context_error(result.message):
                    print("[Info] Stopping because the CUDA context may be invalid after this error.")
                    print(f"\n[Summary] total={total} failed={failed} skipped={skipped}")
                    return 1
                if args.fail_fast:
                    print(f"\n[Summary] total={total} failed={failed} skipped={skipped}")
                    return 1

    print(f"\n[Summary] total={total} failed={failed} skipped={skipped}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
