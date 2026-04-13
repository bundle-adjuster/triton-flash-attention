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

Not tested on AMD, so let me know!

## Exercise 1: autotuning the backwards pass

Can you apply autotuning configs to the backwards pass like done for the forward pass?

## Exercise 2: how to make Flash Attention faster

As you can see, during the backwards pass we are going through the entire `SEQ_LEN` even when the attention calculation is `causal`, can you avoid going through all tokens that would not contribute to any change in `dK`, `dQ` and `dV` when the attention calculation is causal?
