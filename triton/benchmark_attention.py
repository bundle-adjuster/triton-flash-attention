#!/usr/bin/env python3
"""
Compare PyTorch `scaled_dot_product_attention` (SDPA) with this repo's Triton attention.

- Times both forward passes on CUDA using `torch.cuda.Event`.
- By default the printed ms/iter is **steady-state** (warmup runs first but is not counted).
- Use `--amortize-warmup` to report ms/call averaged over **warmup + repeat** (includes compile/autotune effects).
- Optional `--profile` prints a `torch.profiler` summary (CUDA kernel names and times).
- Optional `--chrome-trace DIR` writes separate Chrome trace files for SDPA and Triton runs.

Examples:

  python benchmark_attention.py --batch 2 --heads 16 --seq-len 4096 --head-dim 64 --causal

  python benchmark_attention.py --profile --sdpa-backend flash --warmup 30 --repeat 100

  python benchmark_attention.py --amortize-warmup --warmup 30 --repeat 50

  python benchmark_attention.py --chrome-trace ./traces --seq-len 2048
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from typing import Callable

import torch
import torch.nn.functional as F

from flash_attention import TritonAttention


def _sdpa_backend_context(backend: str) -> contextlib.AbstractContextManager[None]:
    """Restrict SDPA to a single backend when supported (for profiling / apples-to-apples)."""
    if backend == "default":
        return contextlib.nullcontext()

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        import torch as _torch

        if backend == "math":
            return _torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            )
        if backend == "flash":
            return _torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_mem_efficient=False,
                enable_math=False,
            )
        if backend == "efficient":
            return _torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=True,
                enable_math=False,
            )
        raise ValueError(
            f"Unknown or unsupported SDPA backend {backend!r} for this PyTorch build "
            "(install a newer torch with torch.nn.attention for full backend control)."
        )

    name_map = {
        "math": SDPBackend.MATH,
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
    }
    if hasattr(SDPBackend, "CUDNN_ATTENTION"):
        name_map["cudnn"] = SDPBackend.CUDNN_ATTENTION
    if backend not in name_map:
        raise ValueError(f"Unknown SDPA backend {backend!r}; choose from {sorted(name_map)} or 'default'.")
    return sdpa_kernel(name_map[backend])


def make_tensors(
    batch: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    g = torch.Generator(device=device)
    g.manual_seed(0)
    q = torch.randn(
        batch, num_heads, seq_len, head_dim, dtype=dtype, device=device, generator=g
    )
    k = torch.randn(
        batch, num_heads, seq_len, head_dim, dtype=dtype, device=device, generator=g
    )
    v = torch.randn(
        batch, num_heads, seq_len, head_dim, dtype=dtype, device=device, generator=g
    )
    softmax_scale = head_dim**-0.5
    return q, k, v, softmax_scale


def sdpa_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    softmax_scale: float,
    sdpa_backend_name: str,
) -> torch.Tensor:
    with _sdpa_backend_context(sdpa_backend_name):
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, scale=softmax_scale
        )


def triton_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    softmax_scale: float,
) -> torch.Tensor:
    return TritonAttention.apply(q, k, v, causal, softmax_scale)


def bench_cuda_ms(
    fn: Callable[[], None],
    *,
    warmup: int,
    repeat: int,
    amortize_warmup: bool,
) -> float:
    """Return mean ms per forward call.

    If amortize_warmup is False (default steady timing): warmup iterations run, then CUDA
    events time only the `repeat` loop; reported value is ms_total / repeat.

    If amortize_warmup is True: one CUDA event pair wraps warmup and repeat; reported
    value is ms_total / (warmup + repeat).
    """
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    if amortize_warmup:
        t0 = time.perf_counter()
        start.record()
        for _ in range(warmup):
            fn()
        for _ in range(repeat):
            fn()
        end.record()
        torch.cuda.synchronize()
        elapsed_wall = time.perf_counter() - t0
        n = warmup + repeat
        ms_total = start.elapsed_time(end)
        per_ms = ms_total / n if n else 0.0
        if per_ms <= 0 or per_ms > (elapsed_wall * 1000 / n) * 2:
            return (elapsed_wall / n) * 1000.0 if n else 0.0
        return per_ms

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    elapsed_wall = time.perf_counter() - t0
    ms_total = start.elapsed_time(end)
    per_iter_ms = ms_total / repeat if repeat else 0.0
    if per_iter_ms <= 0 or (repeat and per_iter_ms > (elapsed_wall * 1000 / repeat) * 2):
        return (elapsed_wall / repeat) * 1000.0 if repeat else 0.0
    return per_iter_ms


def forward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    softmax_scale: float,
) -> torch.Tensor:
    """Materialized attention (O(S^2)); for small seq_len correctness checks only."""
    mask = torch.tril(torch.ones(q.size(2), q.size(2), device=q.device, dtype=torch.bool))
    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    if causal:
        scores = scores.masked_fill(~mask, float("-inf"))
    p = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(p, v)


def run_profiler(
    fn: Callable[[], None],
    *,
    name: str,
    iterations: int,
    chrome_path: str | None,
) -> None:
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        torch.cuda.synchronize()
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()

    print(f"\n=== Profiler: {name} ({iterations} iters) ===")
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
            row_limit=25,
        )
    )
    if chrome_path:
        parent = os.path.dirname(os.path.abspath(chrome_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        prof.export_chrome_trace(chrome_path)
        print(f"Wrote Chrome trace: {chrome_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--causal", action="store_true", default=True)
    p.add_argument("--no-causal", action="store_false", dest="causal")
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--repeat", type=int, default=50)
    p.add_argument(
        "--amortize-warmup",
        action="store_true",
        help=(
            "Include warmup in the reported average: time (warmup + repeat) calls with one "
            "CUDA timer and divide by warmup+repeat. Default: warmup runs but timing uses "
            "only the repeat loop (steady-state / ignore-warmup for the printed ms)."
        ),
    )
    p.add_argument("--check-correctness", action="store_true", help="Compare forwards vs materialized ref (seq_len <= 4096 recommended)")
    sdpa_choices = ["default", "math", "flash", "efficient"]
    try:
        from torch.nn.attention import SDPBackend

        if hasattr(SDPBackend, "CUDNN_ATTENTION"):
            sdpa_choices.append("cudnn")
    except ImportError:
        pass
    p.add_argument(
        "--sdpa-backend",
        default="default",
        choices=sdpa_choices,
        help="Which SDPA backend to force (when supported). 'default' lets PyTorch choose.",
    )
    p.add_argument("--profile", action="store_true", help="Run torch.profiler and print kernel summary for each implementation")
    p.add_argument("--profile-iters", type=int, default=5, help="Iterations inside each profiler session")
    p.add_argument(
        "--chrome-trace",
        metavar="DIR",
        default=None,
        help="Directory to write trace_sdpa.json and trace_triton.json (Chrome://tracing)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.repeat < 1:
        print("error: --repeat must be at least 1", file=sys.stderr)
        return 2
    if args.amortize_warmup and args.warmup + args.repeat < 1:
        print("error: --amortize-warmup requires warmup + repeat >= 1", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("CUDA is required for this benchmark.", file=sys.stderr)
        return 1

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = torch.device("cuda")

    q, k, v, scale = make_tensors(
        args.batch, args.heads, args.seq_len, args.head_dim, dtype, device
    )

    if args.check_correctness:
        if args.seq_len > 8192:
            print("Warning: correctness check uses O(S^2) memory; consider smaller --seq-len.", file=sys.stderr)
        with torch.no_grad():
            ref = forward_reference(q, k, v, causal=args.causal, softmax_scale=scale)
            y_sdpa = sdpa_forward(
                q, k, v, causal=args.causal, softmax_scale=scale, sdpa_backend_name=args.sdpa_backend
            )
            y_tri = triton_forward(q, k, v, causal=args.causal, softmax_scale=scale)
        atol, rtol = 1e-2, 0.0
        ok_sdpa = torch.allclose(ref, y_sdpa, atol=atol, rtol=rtol)
        ok_tri = torch.allclose(ref, y_tri, atol=atol, rtol=rtol)
        ok_pair = torch.allclose(y_sdpa, y_tri, atol=atol, rtol=rtol)
        print(f"correctness vs materialized ref: SDPA={ok_sdpa}, Triton={ok_tri}")
        print(f"SDPA vs Triton (same atol/rtol): {ok_pair}")
        if not (ok_sdpa and ok_tri):
            def _max_abs(a, b):
                return (a.float() - b.float()).abs().max().item()

            print(f"max |ref - sdpa|: {_max_abs(ref, y_sdpa)}")
            print(f"max |ref - triton|: {_max_abs(ref, y_tri)}")

    # Timing (forward only, no autograd).
    def run_sdpa():
        sdpa_forward(
            q, k, v, causal=args.causal, softmax_scale=scale, sdpa_backend_name=args.sdpa_backend
        )

    def run_triton():
        triton_forward(q, k, v, causal=args.causal, softmax_scale=scale)

    with torch.no_grad():
        ms_sdpa = bench_cuda_ms(
            run_sdpa,
            warmup=args.warmup,
            repeat=args.repeat,
            amortize_warmup=args.amortize_warmup,
        )
        ms_tri = bench_cuda_ms(
            run_triton,
            warmup=args.warmup,
            repeat=args.repeat,
            amortize_warmup=args.amortize_warmup,
        )

    timing_note = (
        f"amortized over warmup+repeat={args.warmup + args.repeat}"
        if args.amortize_warmup
        else f"steady over repeat={args.repeat} (warmup={args.warmup} not counted)"
    )
    print(
        f"shape (B,H,S,D)=({args.batch},{args.heads},{args.seq_len},{args.head_dim}) "
        f"dtype={args.dtype} causal={args.causal} sdpa_backend={args.sdpa_backend!r}"
    )
    print(f"timing: {timing_note}")
    print(f"SDPA forward:   {ms_sdpa:.4f} ms/call")
    print(f"Triton forward: {ms_tri:.4f} ms/call")
    if ms_tri > 0:
        print(f"speedup (SDPA / Triton): {ms_sdpa / ms_tri:.3f}x")

    if args.profile:
        chrome_sdpa = chrome_tri = None
        if args.chrome_trace:
            os.makedirs(args.chrome_trace, exist_ok=True)
            chrome_sdpa = os.path.join(args.chrome_trace, "trace_sdpa.json")
            chrome_tri = os.path.join(args.chrome_trace, "trace_triton.json")

        with torch.no_grad():
            run_profiler(
                run_sdpa,
                name=f"PyTorch SDPA ({args.sdpa_backend})",
                iterations=args.profile_iters,
                chrome_path=chrome_sdpa,
            )
            run_profiler(
                run_triton,
                name="TritonAttention",
                iterations=args.profile_iters,
                chrome_path=chrome_tri,
            )
    elif args.chrome_trace:
        os.makedirs(args.chrome_trace, exist_ok=True)
        with torch.no_grad():
            run_profiler(
                run_sdpa,
                name=f"PyTorch SDPA ({args.sdpa_backend})",
                iterations=args.profile_iters,
                chrome_path=os.path.join(args.chrome_trace, "trace_sdpa.json"),
            )
            run_profiler(
                run_triton,
                name="TritonAttention",
                iterations=args.profile_iters,
                chrome_path=os.path.join(args.chrome_trace, "trace_triton.json"),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
