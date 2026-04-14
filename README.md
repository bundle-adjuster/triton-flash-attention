# Flash Attention implemented with Triton

Implements the Flash Attention 2 algorithm, based on the code published by OpenAI's team at [Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)

It also includes some cuda examples as shown in the video.

Install the requirements at `triton/requirements.txt` to launch the Python file. Adjust the `BATCH_SIZE`, `NUM_HEADS`, `SEQ_LEN`, `HEAD_DIM` to make sure your computer doesn't explode.

The *naive* implementation materializes a `SEQ_LEN x SEQ_LEN` tensor, so it may be the bottleneck in running this code. Just disable it and try to push the `SEQ_LEN` of the Flash Attention to the limit supported by your hardware.

## Benchmarking (PyTorch SDPA vs Triton)

From the repo root, install dependencies, then run the benchmark script **from the `triton/` directory** (so `flash_attention` imports correctly):

```bash
pip install -r triton/requirements.txt
cd triton
python benchmark_attention.py --batch 2 --heads 16 --seq-len 4096 --head-dim 64 --causal
```

Useful options:

- `--warmup N` / `--repeat M` — number of untimed warmup iterations and timed iterations (defaults: 25 and 50).
- **Steady timing (default):** warmup runs first, but the printed `ms/call` uses CUDA events **only over `--repeat`** (warmup is **not** included). Use this to measure sustained kernel time after JIT/autotune settles.
- **`--amortize-warmup`:** one timer covers **both** warmup and repeat; the printed `ms/call` is total time divided by **`warmup + repeat`**. Use this to see end-to-end cost per call including compilation and autotuning amortized across iterations (especially relevant for Triton autotune on first shapes).
- `--check-correctness` — forward check against a materialized reference (keep `--seq-len` moderate; memory grows like sequence length squared).
- `--profile` — `torch.profiler` summary per backend; `--chrome-trace DIR` — Chrome trace files for SDPA and Triton.

```bash
python benchmark_attention.py --warmup 30 --repeat 100
python benchmark_attention.py --amortize-warmup --warmup 30 --repeat 50
python benchmark_attention.py --profile --sdpa-backend flash
```

## Full FA1 / FA2 benchmarking

For full-kernel comparisons, use `triton/benchmark_fa1_fa2.py`. It benchmarks:

- this repo's Triton attention (`TritonAttention`)
- PyTorch SDPA flash backend
- FlashAttention v2 from `flash-attn` (`flash_attn_func`) when installed
- legacy FA1 API if exposed by your installed `flash-attn` build

Install and run:

```bash
pip install flash-attn
cd triton
python benchmark_fa1_fa2.py --batch 2 --heads 16 --seq-len 1024 --head-dim 64 --causal --warmup 20 --repeat 50
```

Optional:

- `--amortize-warmup` include warmup in reported ms/call
- `--profile --profile-iters N` print `torch.profiler` summaries
- `--chrome-trace-dir DIR` export per-backend trace JSON files
- `--sweep-seq-lens 512,1024,2048,4096` run a paper-style sequence-length sweep and print a summary table
- `--dtypes float16,bfloat16,float32` run multiple dtypes in one invocation (for unsupported backend+dtype pairs, the script prints a skip message)

Notes:

- FA1 support depends on the exact `flash-attn` version/build (legacy symbol/API availability).
- If FA1/FA2 symbols are missing, the script prints a skip message and continues.

## CUDA FlashAttention benchmark

The `cuda/` folder now includes `flash_attention_cuda.cu`, a CUDA forward-attention benchmark with:

- `CUDA flash attention` kernel (streaming / online softmax, FlashAttention-style, no full score matrix materialization)
- `CUDA regular attention` kernel (multi-pass softmax baseline on CUDA)
- optional output comparison between flash and regular CUDA kernels
- warmup/repeat timing with optional warmup amortization
- optional profiler start/stop hooks

Build and run:

```bash
cd cuda
make flash_attention_cuda.out
./flash_attention_cuda.out --batch 2 --heads 16 --seq-len 1024 --head-dim 64 --causal --warmup 20 --repeat 50
```

Useful flags:

- `--amortize-warmup` include warmup in reported ms/call (otherwise warmup is excluded from timing)
- `--check` / `--no-check` enable/disable `max |flash - regular|` output validation
- `--profile` wraps benchmark loops with `cudaProfilerStart/Stop`

For external profiling:

```bash
nsys profile -o fa_cuda_report ./flash_attention_cuda.out --batch 2 --heads 16 --seq-len 1024 --head-dim 64 --repeat 50 --no-check --profile
ncu --set full ./flash_attention_cuda.out --batch 2 --heads 16 --seq-len 1024 --head-dim 64 --repeat 20 --no-check
```

## CUTLASS attention benchmark

The `cuda/` folder also includes `cutlass_attention.cu`, a unified benchmark that runs all three kernels on the same `(Q, K, V)` inputs:

- CUTLASS attention
- CUDA flash attention (online softmax)
- optional CUDA regular attention (disabled by default because it is very slow)

CUTLASS path uses:

- `QK^T` (attention scores)
- softmax (custom CUDA kernel, optional causal mask)
- `P @ V` (output projection)

It supports the same benchmark/profiling options as the CUDA benchmark:

- `--warmup` / `--repeat`
- `--amortize-warmup`
- `--check` / `--no-check` (prints `max |cutlass - flash|`)
- `--include-regular` also run regular CUDA attention and print regular-vs-flash speedup/error
- `--profile`

Build requirements:

- CUTLASS checkout available locally
- `CUTLASS_PATH` pointing to that checkout (or `/usr/local/cutlass`)

```bash
git clone https://github.com/NVIDIA/cutlass.git
cd cuda
make cutlass_attention.out CUTLASS_PATH=/path/to/cutlass
./cutlass_attention.out --batch 2 --heads 16 --seq-len 1024 --head-dim 64 --causal --warmup 20 --repeat 50
```

Profile examples:

```bash
nsys profile -o cutlass_attention_report ./cutlass_attention.out --batch 2 --heads 16 --seq-len 1024 --head-dim 64 --repeat 50 --no-check --profile
ncu --set full ./cutlass_attention.out --batch 2 --heads 16 --seq-len 1024 --head-dim 64 --repeat 20 --no-check
```

Not tested on AMD, so let me know!

## Exercise 1: autotuning the backwards pass

Can you apply autotuning configs to the backwards pass like done for the forward pass?

## Exercise 2: how to make Flash Attention faster

As you can see, during the backwards pass we are going through the entire `SEQ_LEN` even when the attention calculation is `causal`, can you avoid going through all tokens that would not contribute to any change in `dK`, `dQ` and `dV` when the attention calculation is causal?
