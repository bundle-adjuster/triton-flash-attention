#!/usr/bin/env python3
"""
Benchmark full FlashAttention implementations (FA2 and FA1 legacy API when available)
against this repository's Triton implementation and PyTorch SDPA flash backend.

Notes:
- FA2 comes from the `flash-attn` package (`flash_attn_func`).
- FA1 is exposed only on some flash-attn builds via legacy symbols.
- If an implementation is unavailable in your environment, it is skipped with a message.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import sys
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from flash_attention import TritonAttention


@dataclass
class BenchCfg:
    batch: int
    heads: int
    seq_len: int
    head_dim: int
    causal: bool
    dtype: torch.dtype
    warmup: int
    repeat: int
    amortize_warmup: bool
    profile: bool
    profile_iters: int
    chrome_trace_dir: str | None


def bench_cuda_ms(
    fn: Callable[[], None], warmup: int, repeat: int, amortize_warmup: bool
) -> float:
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    if amortize_warmup:
        n = warmup + repeat
        if n < 1:
            raise ValueError("warmup+repeat must be >= 1")
        start.record()
        for _ in range(warmup):
            fn()
        for _ in range(repeat):
            fn()
        end.record()
        torch.cuda.synchronize()
        total = start.elapsed_time(end)
        return total / n

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    total = start.elapsed_time(end)
    return total / repeat


def run_profiler(
    name: str,
    fn: Callable[[], None],
    iters: int,
    chrome_trace: str | None,
) -> None:
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        torch.cuda.synchronize()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()

    print(f"\n=== Profiler: {name} ({iters} iters) ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
    if chrome_trace is not None:
        os.makedirs(os.path.dirname(os.path.abspath(chrome_trace)), exist_ok=True)
        prof.export_chrome_trace(chrome_trace)
        print(f"Wrote trace: {chrome_trace}")


def make_inputs(cfg: BenchCfg) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    q = torch.randn(
        cfg.batch,
        cfg.heads,
        cfg.seq_len,
        cfg.head_dim,
        device="cuda",
        dtype=cfg.dtype,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = cfg.head_dim**-0.5
    return q, k, v, scale


def make_torch_flash_runner(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, scale: float
) -> Callable[[], None]:
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception:
        return lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)

    def _run() -> None:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)

    return _run


def make_repo_triton_runner(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, scale: float
) -> Callable[[], None]:
    return lambda: TritonAttention.apply(q, k, v, causal, scale)


def _try_load_flash_attn():
    try:
        return importlib.import_module("flash_attn.flash_attn_interface")
    except Exception:
        return None


def make_fa2_runner(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, scale: float
) -> tuple[str, Callable[[], None]] | None:
    mod = _try_load_flash_attn()
    if mod is None:
        return None
    if not hasattr(mod, "flash_attn_func"):
        return None

    # flash-attn expects [B, S, H, D]
    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()

    def _run() -> None:
        mod.flash_attn_func(
            q_bshd,
            k_bshd,
            v_bshd,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=causal,
        )

    return ("flash-attn FA2", _run)


def make_fa1_legacy_runner(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, scale: float
) -> tuple[str, Callable[[], None]] | None:
    mod = _try_load_flash_attn()
    if mod is None:
        return None

    # Legacy FA1 symbol sometimes exists in older/newer builds for compatibility.
    legacy_name = None
    for cand in ("flash_attn_unpadded_func", "flash_attn_varlen_func"):
        if hasattr(mod, cand):
            legacy_name = cand
            break
    if legacy_name is None:
        return None

    # Use varlen-style packing (single sequence length per sample).
    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()
    b, s, h, d = q_bshd.shape
    q_flat = q_bshd.reshape(b * s, h, d)
    k_flat = k_bshd.reshape(b * s, h, d)
    v_flat = v_bshd.reshape(b * s, h, d)
    cu_seqlens = torch.arange(0, (b + 1) * s, step=s, device=q.device, dtype=torch.int32)
    max_seqlen = s
    legacy = getattr(mod, legacy_name)

    def _run() -> None:
        # Signature compatibility differs across flash-attn versions.
        # Try common variants and fail clearly if this build has a different ABI.
        tried = []
        for variant in ("varlen_v2", "unpadded_v1"):
            try:
                if variant == "varlen_v2":
                    legacy(
                        q_flat,
                        k_flat,
                        v_flat,
                        cu_seqlens,
                        cu_seqlens,
                        max_seqlen,
                        max_seqlen,
                        0.0,
                        scale,
                        causal,
                    )
                else:
                    legacy(
                        q_flat,
                        k_flat,
                        v_flat,
                        cu_seqlens,
                        cu_seqlens,
                        max_seqlen,
                        max_seqlen,
                        0.0,
                        scale,
                        False,
                        causal,
                    )
                return
            except TypeError as e:
                tried.append(str(e))
        raise RuntimeError(
            f"Legacy FA1 symbol '{legacy_name}' exists but call ABI did not match this script.\n"
            + "\n".join(tried)
        )

    return (f"flash-attn legacy ({legacy_name})", _run)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--causal", action="store_true", default=True)
    p.add_argument("--no-causal", action="store_false", dest="causal")
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeat", type=int, default=50)
    p.add_argument("--amortize-warmup", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile-iters", type=int, default=5)
    p.add_argument("--chrome-trace-dir", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA is required.", file=sys.stderr)
        return 1

    cfg = BenchCfg(
        batch=args.batch,
        heads=args.heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        causal=args.causal,
        dtype=torch.float16 if args.dtype == "float16" else torch.bfloat16,
        warmup=args.warmup,
        repeat=args.repeat,
        amortize_warmup=args.amortize_warmup,
        profile=args.profile,
        profile_iters=args.profile_iters,
        chrome_trace_dir=args.chrome_trace_dir,
    )

    q, k, v, scale = make_inputs(cfg)

    runners: list[tuple[str, Callable[[], None]]] = []
    runners.append(("repo triton", make_repo_triton_runner(q, k, v, cfg.causal, scale)))
    runners.append(("torch sdpa flash", make_torch_flash_runner(q, k, v, cfg.causal, scale)))

    fa2 = make_fa2_runner(q, k, v, cfg.causal, scale)
    if fa2 is not None:
        runners.append(fa2)
    else:
        print("Skipping flash-attn FA2: package/symbol not available.")

    fa1 = make_fa1_legacy_runner(q, k, v, cfg.causal, scale)
    if fa1 is not None:
        runners.append(fa1)
    else:
        print("Skipping FA1 legacy: symbol/API not available in this flash-attn build.")

    print(
        f"shape (B,H,S,D)=({cfg.batch},{cfg.heads},{cfg.seq_len},{cfg.head_dim}) "
        f"dtype={args.dtype} causal={cfg.causal}"
    )
    timing_mode = (
        f"amortized over warmup+repeat={cfg.warmup + cfg.repeat}"
        if cfg.amortize_warmup
        else f"steady over repeat={cfg.repeat} (warmup={cfg.warmup} not counted)"
    )
    print(f"timing: {timing_mode}")

    with torch.no_grad():
        results: list[tuple[str, float]] = []
        for name, fn in runners:
            ms = bench_cuda_ms(fn, cfg.warmup, cfg.repeat, cfg.amortize_warmup)
            print(f"{name}: {ms:.4f} ms/call")
            results.append((name, ms))

            if cfg.profile:
                trace = None
                if cfg.chrome_trace_dir:
                    safe = name.replace(" ", "_").replace("/", "_")
                    trace = os.path.join(cfg.chrome_trace_dir, f"trace_{safe}.json")
                run_profiler(name, fn, cfg.profile_iters, trace)

    # relative speedups vs fastest implementation
    fastest_name, fastest_ms = min(results, key=lambda x: x[1])
    print(f"fastest: {fastest_name} ({fastest_ms:.4f} ms/call)")
    for name, ms in results:
        print(f"relative {name}: {ms / fastest_ms:.3f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

