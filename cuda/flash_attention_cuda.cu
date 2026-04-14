#include <assert.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <math.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "cuda_common.cuh"

namespace {

constexpr int MAX_HEAD_DIM = 128;

struct Args {
    int batch = 2;
    int heads = 16;
    int seq_len = 1024;
    int head_dim = 64;
    int warmup = 20;
    int repeat = 50;
    int causal = 1;
    int check = 0;
    int amortize_warmup = 0;
    int profile = 0;
};

void print_usage(const char* prog) {
    printf(
        "Usage: %s [options]\n"
        "  --batch N               (default: 2)\n"
        "  --heads N               (default: 16)\n"
        "  --seq-len N             (default: 1024)\n"
        "  --head-dim N            (default: 64, max: %d)\n"
        "  --warmup N              (default: 20)\n"
        "  --repeat N              (default: 50)\n"
        "  --causal / --no-causal  (default: causal)\n"
        "  --check / --no-check    compare regular CUDA vs flash CUDA outputs (default: no-check)\n"
        "  --amortize-warmup       include warmup calls in reported ms/call\n"
        "  --profile               wrap benchmark loop with cudaProfilerStart/Stop\n"
        "  -h, --help              show this message\n",
        prog,
        MAX_HEAD_DIM);
}

bool next_value(int argc, char** argv, int& i, int* out) {
    if (i + 1 >= argc) return false;
    *out = atoi(argv[++i]);
    return true;
}

bool parse_args(int argc, char** argv, Args* args) {
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (strcmp(a, "--batch") == 0) {
            if (!next_value(argc, argv, i, &args->batch)) return false;
        } else if (strcmp(a, "--heads") == 0) {
            if (!next_value(argc, argv, i, &args->heads)) return false;
        } else if (strcmp(a, "--seq-len") == 0) {
            if (!next_value(argc, argv, i, &args->seq_len)) return false;
        } else if (strcmp(a, "--head-dim") == 0) {
            if (!next_value(argc, argv, i, &args->head_dim)) return false;
        } else if (strcmp(a, "--warmup") == 0) {
            if (!next_value(argc, argv, i, &args->warmup)) return false;
        } else if (strcmp(a, "--repeat") == 0) {
            if (!next_value(argc, argv, i, &args->repeat)) return false;
        } else if (strcmp(a, "--causal") == 0) {
            args->causal = 1;
        } else if (strcmp(a, "--no-causal") == 0) {
            args->causal = 0;
        } else if (strcmp(a, "--check") == 0) {
            args->check = 1;
        } else if (strcmp(a, "--no-check") == 0) {
            args->check = 0;
        } else if (strcmp(a, "--amortize-warmup") == 0) {
            args->amortize_warmup = 1;
        } else if (strcmp(a, "--profile") == 0) {
            args->profile = 1;
        } else if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else {
            printf("Unknown arg: %s\n", a);
            return false;
        }
    }
    return true;
}

__host__ __device__ __forceinline__ int linear_idx_4d(
    int b,
    int h,
    int s,
    int d,
    int num_heads,
    int seq_len,
    int head_dim) {
    return (((b * num_heads + h) * seq_len + s) * head_dim + d);
}

__global__ void attention_forward_naive_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int batch,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    int causal) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_q = batch * num_heads * seq_len;
    if (tid >= total_q) return;

    int q_idx = tid % seq_len;
    int tmp = tid / seq_len;
    int h = tmp % num_heads;
    int b = tmp / num_heads;

    int max_k = causal ? (q_idx + 1) : seq_len;

    float row_max = -INFINITY;
    for (int k_idx = 0; k_idx < max_k; ++k_idx) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            float q = Q[linear_idx_4d(b, h, q_idx, d, num_heads, seq_len, head_dim)];
            float k = K[linear_idx_4d(b, h, k_idx, d, num_heads, seq_len, head_dim)];
            score += q * k;
        }
        score *= scale;
        row_max = fmaxf(row_max, score);
    }

    float denom = 0.0f;
    for (int k_idx = 0; k_idx < max_k; ++k_idx) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            float q = Q[linear_idx_4d(b, h, q_idx, d, num_heads, seq_len, head_dim)];
            float k = K[linear_idx_4d(b, h, k_idx, d, num_heads, seq_len, head_dim)];
            score += q * k;
        }
        score *= scale;
        denom += expf(score - row_max);
    }

    for (int d = 0; d < head_dim; ++d) {
        float out = 0.0f;
        for (int k_idx = 0; k_idx < max_k; ++k_idx) {
            float score = 0.0f;
            for (int kdim = 0; kdim < head_dim; ++kdim) {
                float q = Q[linear_idx_4d(b, h, q_idx, kdim, num_heads, seq_len, head_dim)];
                float k = K[linear_idx_4d(b, h, k_idx, kdim, num_heads, seq_len, head_dim)];
                score += q * k;
            }
            score *= scale;
            float p = expf(score - row_max) / denom;
            float v = V[linear_idx_4d(b, h, k_idx, d, num_heads, seq_len, head_dim)];
            out += p * v;
        }
        O[linear_idx_4d(b, h, q_idx, d, num_heads, seq_len, head_dim)] = out;
    }
}

__global__ void attention_forward_online_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int batch,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    int causal) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_q = batch * num_heads * seq_len;
    if (tid >= total_q) return;

    int q_idx = tid % seq_len;
    int tmp = tid / seq_len;
    int h = tmp % num_heads;
    int b = tmp / num_heads;

    float out_local[MAX_HEAD_DIM];
    for (int d = 0; d < head_dim; ++d) out_local[d] = 0.0f;

    float m = -INFINITY;
    float l = 0.0f;

    int max_k = causal ? (q_idx + 1) : seq_len;
    for (int k_idx = 0; k_idx < max_k; ++k_idx) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            float q = Q[linear_idx_4d(b, h, q_idx, d, num_heads, seq_len, head_dim)];
            float k = K[linear_idx_4d(b, h, k_idx, d, num_heads, seq_len, head_dim)];
            score += q * k;
        }
        score *= scale;

        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float beta = expf(score - m_new);

        for (int d = 0; d < head_dim; ++d) {
            float v = V[linear_idx_4d(b, h, k_idx, d, num_heads, seq_len, head_dim)];
            out_local[d] = out_local[d] * alpha + beta * v;
        }
        l = l * alpha + beta;
        m = m_new;
    }

    float inv_l = 1.0f / l;
    for (int d = 0; d < head_dim; ++d) {
        O[linear_idx_4d(b, h, q_idx, d, num_heads, seq_len, head_dim)] = out_local[d] * inv_l;
    }
}

float benchmark_kernel(
    const char* label,
    void (*kernel_launcher)(const float*, const float*, const float*, float*, int, int, int, int, float, int),
    const float* d_q,
    const float* d_k,
    const float* d_v,
    float* d_out,
    const Args& args,
    float scale) {
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    auto launch = [&]() {
        kernel_launcher(
            d_q,
            d_k,
            d_v,
            d_out,
            args.batch,
            args.heads,
            args.seq_len,
            args.head_dim,
            scale,
            args.causal);
    };

    float ms_per_call = 0.0f;
    CUDA_CHECK(cudaDeviceSynchronize());
    if (args.profile) {
        CUDA_CHECK(cudaProfilerStart());
    }

    if (args.amortize_warmup) {
        int total = args.warmup + args.repeat;
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < args.warmup; ++i) launch();
        for (int i = 0; i < args.repeat; ++i) launch();
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        float ms_total = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_total, start, end));
        ms_per_call = ms_total / total;
    } else {
        for (int i = 0; i < args.warmup; ++i) launch();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < args.repeat; ++i) launch();
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        float ms_total = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_total, start, end));
        ms_per_call = ms_total / args.repeat;
    }

    if (args.profile) {
        CUDA_CHECK(cudaProfilerStop());
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));

    printf("%s: %.4f ms/call\n", label, ms_per_call);
    return ms_per_call;
}

void launch_attention_regular(
    const float* d_q,
    const float* d_k,
    const float* d_v,
    float* d_out,
    int batch,
    int heads,
    int seq_len,
    int head_dim,
    float scale,
    int causal) {
    int total = batch * heads * seq_len;
    constexpr int block = 128;
    int grid = (total + block - 1) / block;
    attention_forward_naive_kernel<<<grid, block>>>(
        d_q, d_k, d_v, d_out, batch, heads, seq_len, head_dim, scale, causal);
    CUDA_CHECK(cudaPeekAtLastError());
}

void launch_attention_flash(
    const float* d_q,
    const float* d_k,
    const float* d_v,
    float* d_out,
    int batch,
    int heads,
    int seq_len,
    int head_dim,
    float scale,
    int causal) {
    int total = batch * heads * seq_len;
    constexpr int block = 128;
    int grid = (total + block - 1) / block;
    attention_forward_online_kernel<<<grid, block>>>(
        d_q, d_k, d_v, d_out, batch, heads, seq_len, head_dim, scale, causal);
    CUDA_CHECK(cudaPeekAtLastError());
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, &args)) {
        print_usage(argv[0]);
        return 1;
    }

    if (args.head_dim <= 0 || args.head_dim > MAX_HEAD_DIM) {
        printf("head_dim must be in [1, %d]\n", MAX_HEAD_DIM);
        return 1;
    }
    if (args.batch <= 0 || args.heads <= 0 || args.seq_len <= 0 || args.warmup < 0 ||
        args.repeat <= 0) {
        printf("invalid args\n");
        return 1;
    }

    const int64_t numel = static_cast<int64_t>(args.batch) * args.heads * args.seq_len * args.head_dim;
    const size_t bytes = static_cast<size_t>(numel) * sizeof(float);
    const float scale = 1.0f / sqrtf(static_cast<float>(args.head_dim));

    std::vector<float> h_q(numel), h_k(numel), h_v(numel);
    std::vector<float> h_out_flash(numel), h_out_regular(numel);

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int64_t i = 0; i < numel; ++i) {
        h_q[i] = dist(rng);
        h_k[i] = dist(rng);
        h_v[i] = dist(rng);
    }

    float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, bytes));
    CUDA_CHECK(cudaMalloc(&d_k, bytes));
    CUDA_CHECK(cudaMalloc(&d_v, bytes));
    CUDA_CHECK(cudaMalloc(&d_o, bytes));

    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), bytes, cudaMemcpyHostToDevice));

    printf(
        "shape (B,H,S,D)=(%d,%d,%d,%d) causal=%d\n",
        args.batch,
        args.heads,
        args.seq_len,
        args.head_dim,
        args.causal);
    if (args.amortize_warmup) {
        printf("timing: amortized over warmup+repeat=%d\n", args.warmup + args.repeat);
    } else {
        printf("timing: steady over repeat=%d (warmup=%d not counted)\n", args.repeat, args.warmup);
    }
    if (args.profile) {
        printf("profiling enabled via cudaProfilerStart/Stop (use nsys/ncu around this run).\n");
    }

    float ms_flash = benchmark_kernel(
        "CUDA flash attention (online softmax)",
        launch_attention_flash,
        d_q,
        d_k,
        d_v,
        d_o,
        args,
        scale);
    CUDA_CHECK(cudaMemcpy(h_out_flash.data(), d_o, bytes, cudaMemcpyDeviceToHost));

    float ms_regular = benchmark_kernel(
        "CUDA regular attention",
        launch_attention_regular,
        d_q,
        d_k,
        d_v,
        d_o,
        args,
        scale);
    CUDA_CHECK(cudaMemcpy(h_out_regular.data(), d_o, bytes, cudaMemcpyDeviceToHost));

    if (args.check) {
        float diff_pair = max_abs_diff(h_out_flash, h_out_regular);
        printf("max |flash - regular|: %.6f\n", diff_pair);
    }

    printf("speedup (regular / flash): %.3fx\n", ms_regular / ms_flash);

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_o));
    return 0;
}
