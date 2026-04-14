#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cutlass/gemm/device/gemm.h>

#include "cuda_common.cuh"

namespace {

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
    int include_regular = 0;
};

__host__ __device__ __forceinline__ int idx4(
    int b, int h, int s, int d, int num_heads, int seq_len, int head_dim) {
    return (((b * num_heads + h) * seq_len + s) * head_dim + d);
}

void usage(const char* prog) {
    printf(
        "Usage: %s [options]\n"
        "  --batch N               (default: 2)\n"
        "  --heads N               (default: 16)\n"
        "  --seq-len N             (default: 1024)\n"
        "  --head-dim N            (default: 64)\n"
        "  --warmup N              (default: 20)\n"
        "  --repeat N              (default: 50)\n"
        "  --causal / --no-causal  (default: causal)\n"
        "  --check / --no-check    compare CUTLASS vs flash outputs (default: no-check)\n"
        "  --amortize-warmup       include warmup in ms/call\n"
        "  --profile               wrap measured region with cudaProfilerStart/Stop\n"
        "  --include-regular       also run regular CUDA attention (very slow for large seq)\n"
        "  -h, --help              show this message\n",
        prog);
}

bool parse_int_arg(int argc, char** argv, int& i, int* out) {
    if (i + 1 >= argc) return false;
    *out = atoi(argv[++i]);
    return true;
}

bool parse_args(int argc, char** argv, Args* args) {
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (!strcmp(a, "--batch")) {
            if (!parse_int_arg(argc, argv, i, &args->batch)) return false;
        } else if (!strcmp(a, "--heads")) {
            if (!parse_int_arg(argc, argv, i, &args->heads)) return false;
        } else if (!strcmp(a, "--seq-len")) {
            if (!parse_int_arg(argc, argv, i, &args->seq_len)) return false;
        } else if (!strcmp(a, "--head-dim")) {
            if (!parse_int_arg(argc, argv, i, &args->head_dim)) return false;
        } else if (!strcmp(a, "--warmup")) {
            if (!parse_int_arg(argc, argv, i, &args->warmup)) return false;
        } else if (!strcmp(a, "--repeat")) {
            if (!parse_int_arg(argc, argv, i, &args->repeat)) return false;
        } else if (!strcmp(a, "--causal")) {
            args->causal = 1;
        } else if (!strcmp(a, "--no-causal")) {
            args->causal = 0;
        } else if (!strcmp(a, "--check")) {
            args->check = 1;
        } else if (!strcmp(a, "--no-check")) {
            args->check = 0;
        } else if (!strcmp(a, "--amortize-warmup")) {
            args->amortize_warmup = 1;
        } else if (!strcmp(a, "--profile")) {
            args->profile = 1;
        } else if (!strcmp(a, "--include-regular")) {
            args->include_regular = 1;
        } else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(argv[0]);
            exit(0);
        } else {
            return false;
        }
    }
    return true;
}

__global__ void softmax_kernel(
    const float* scores, float* probs, int seq_len, int causal) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;

    int k_limit = causal ? (row + 1) : seq_len;
    float m = -INFINITY;
    int base = row * seq_len;
    for (int k = 0; k < k_limit; ++k) {
        m = fmaxf(m, scores[base + k]);
    }
    float denom = 0.0f;
    for (int k = 0; k < k_limit; ++k) {
        denom += expf(scores[base + k] - m);
    }
    float inv_denom = 1.0f / denom;
    for (int k = 0; k < k_limit; ++k) {
        probs[base + k] = expf(scores[base + k] - m) * inv_denom;
    }
    for (int k = k_limit; k < seq_len; ++k) {
        probs[base + k] = 0.0f;
    }
}

constexpr int MAX_HEAD_DIM = 128;

__global__ void attention_forward_flash_kernel(
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
            score += Q[idx4(b, h, q_idx, d, num_heads, seq_len, head_dim)] *
                     K[idx4(b, h, k_idx, d, num_heads, seq_len, head_dim)];
        }
        score *= scale;
        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float beta = expf(score - m_new);
        for (int d = 0; d < head_dim; ++d) {
            float v = V[idx4(b, h, k_idx, d, num_heads, seq_len, head_dim)];
            out_local[d] = out_local[d] * alpha + beta * v;
        }
        l = l * alpha + beta;
        m = m_new;
    }

    float inv_l = 1.0f / l;
    for (int d = 0; d < head_dim; ++d) {
        O[idx4(b, h, q_idx, d, num_heads, seq_len, head_dim)] = out_local[d] * inv_l;
    }
}

__global__ void attention_forward_regular_kernel(
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
            score += Q[idx4(b, h, q_idx, d, num_heads, seq_len, head_dim)] *
                     K[idx4(b, h, k_idx, d, num_heads, seq_len, head_dim)];
        }
        score *= scale;
        row_max = fmaxf(row_max, score);
    }

    float denom = 0.0f;
    for (int k_idx = 0; k_idx < max_k; ++k_idx) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += Q[idx4(b, h, q_idx, d, num_heads, seq_len, head_dim)] *
                     K[idx4(b, h, k_idx, d, num_heads, seq_len, head_dim)];
        }
        score *= scale;
        denom += expf(score - row_max);
    }

    for (int d = 0; d < head_dim; ++d) {
        float out = 0.0f;
        for (int k_idx = 0; k_idx < max_k; ++k_idx) {
            float score = 0.0f;
            for (int kd = 0; kd < head_dim; ++kd) {
                score += Q[idx4(b, h, q_idx, kd, num_heads, seq_len, head_dim)] *
                         K[idx4(b, h, k_idx, kd, num_heads, seq_len, head_dim)];
            }
            score *= scale;
            float p = expf(score - row_max) / denom;
            out += p * V[idx4(b, h, k_idx, d, num_heads, seq_len, head_dim)];
        }
        O[idx4(b, h, q_idx, d, num_heads, seq_len, head_dim)] = out;
    }
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float max_d = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_d = fmaxf(max_d, fabsf(a[i] - b[i]));
    }
    return max_d;
}

void run_cutlass_attention(
    const float* d_q,
    const float* d_k,
    const float* d_v,
    float* d_o,
    float* d_scores,
    float* d_probs,
    const Args& args,
    float scale) {
    using GemmQK = cutlass::gemm::device::Gemm<
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::layout::ColumnMajor,
        float,
        cutlass::layout::RowMajor,
        float>;

    using GemmPV = cutlass::gemm::device::Gemm<
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::layout::RowMajor,
        float>;

    GemmQK gemm_qk;
    GemmPV gemm_pv;

    const int head_qkv = args.seq_len * args.head_dim;
    const int head_scores = args.seq_len * args.seq_len;

    for (int b = 0; b < args.batch; ++b) {
        for (int h = 0; h < args.heads; ++h) {
            int bh = b * args.heads + h;
            const float* q_ptr = d_q + bh * head_qkv;
            const float* k_ptr = d_k + bh * head_qkv;
            const float* v_ptr = d_v + bh * head_qkv;
            float* o_ptr = d_o + bh * head_qkv;
            float* s_ptr = d_scores + bh * head_scores;
            float* p_ptr = d_probs + bh * head_scores;

            typename GemmQK::Arguments qk_args(
                {args.seq_len, args.seq_len, args.head_dim},
                {q_ptr, args.head_dim},
                {k_ptr, args.head_dim},
                {s_ptr, args.seq_len},
                {s_ptr, args.seq_len},
                {scale, 0.0f});
            cutlass::Status st1 = gemm_qk(qk_args);
            if (st1 != cutlass::Status::kSuccess) {
                printf("CUTLASS QK GEMM failed (b=%d, h=%d)\n", b, h);
                exit(1);
            }

            int threads = 128;
            int blocks = (args.seq_len + threads - 1) / threads;
            softmax_kernel<<<blocks, threads>>>(s_ptr, p_ptr, args.seq_len, args.causal);
            CUDA_CHECK(cudaPeekAtLastError());

            typename GemmPV::Arguments pv_args(
                {args.seq_len, args.head_dim, args.seq_len},
                {p_ptr, args.seq_len},
                {v_ptr, args.head_dim},
                {o_ptr, args.head_dim},
                {o_ptr, args.head_dim},
                {1.0f, 0.0f});
            cutlass::Status st2 = gemm_pv(pv_args);
            if (st2 != cutlass::Status::kSuccess) {
                printf("CUTLASS PV GEMM failed (b=%d, h=%d)\n", b, h);
                exit(1);
            }
        }
    }
}

float benchmark_cutlass(
    const float* d_q,
    const float* d_k,
    const float* d_v,
    float* d_o,
    float* d_scores,
    float* d_probs,
    const Args& args,
    float scale) {
    auto call = [&]() {
        run_cutlass_attention(d_q, d_k, d_v, d_o, d_scores, d_probs, args, scale);
    };

    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    CUDA_CHECK(cudaDeviceSynchronize());
    if (args.profile) CUDA_CHECK(cudaProfilerStart());

    float ms = 0.0f;
    if (args.amortize_warmup) {
        int n = args.warmup + args.repeat;
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < args.warmup; ++i) call();
        for (int i = 0; i < args.repeat; ++i) call();
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
        ms /= n;
    } else {
        for (int i = 0; i < args.warmup; ++i) call();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < args.repeat; ++i) call();
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
        ms /= args.repeat;
    }

    if (args.profile) CUDA_CHECK(cudaProfilerStop());
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    return ms;
}

float benchmark_flash(
    const float* d_q,
    const float* d_k,
    const float* d_v,
    float* d_o,
    const Args& args,
    float scale) {
    auto call = [&]() {
        int total = args.batch * args.heads * args.seq_len;
        constexpr int block = 128;
        int grid = (total + block - 1) / block;
        attention_forward_flash_kernel<<<grid, block>>>(
            d_q,
            d_k,
            d_v,
            d_o,
            args.batch,
            args.heads,
            args.seq_len,
            args.head_dim,
            scale,
            args.causal);
        CUDA_CHECK(cudaPeekAtLastError());
    };

    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    CUDA_CHECK(cudaDeviceSynchronize());
    if (args.profile) CUDA_CHECK(cudaProfilerStart());

    float ms = 0.0f;
    if (args.amortize_warmup) {
        int n = args.warmup + args.repeat;
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < args.warmup; ++i) call();
        for (int i = 0; i < args.repeat; ++i) call();
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
        ms /= n;
    } else {
        for (int i = 0; i < args.warmup; ++i) call();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < args.repeat; ++i) call();
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
        ms /= args.repeat;
    }

    if (args.profile) CUDA_CHECK(cudaProfilerStop());
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    return ms;
}

float benchmark_regular(
    const float* d_q,
    const float* d_k,
    const float* d_v,
    float* d_o,
    const Args& args,
    float scale) {
    auto call = [&]() {
        int total = args.batch * args.heads * args.seq_len;
        constexpr int block = 128;
        int grid = (total + block - 1) / block;
        attention_forward_regular_kernel<<<grid, block>>>(
            d_q,
            d_k,
            d_v,
            d_o,
            args.batch,
            args.heads,
            args.seq_len,
            args.head_dim,
            scale,
            args.causal);
        CUDA_CHECK(cudaPeekAtLastError());
    };

    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    CUDA_CHECK(cudaDeviceSynchronize());
    if (args.profile) CUDA_CHECK(cudaProfilerStart());

    float ms = 0.0f;
    if (args.amortize_warmup) {
        int n = args.warmup + args.repeat;
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < args.warmup; ++i) call();
        for (int i = 0; i < args.repeat; ++i) call();
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
        ms /= n;
    } else {
        for (int i = 0; i < args.warmup; ++i) call();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < args.repeat; ++i) call();
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
        ms /= args.repeat;
    }

    if (args.profile) CUDA_CHECK(cudaProfilerStop());
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    return ms;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, &args)) {
        usage(argv[0]);
        return 1;
    }
    if (args.batch <= 0 || args.heads <= 0 || args.seq_len <= 0 || args.head_dim <= 0 ||
        args.warmup < 0 || args.repeat <= 0) {
        printf("Invalid args.\n");
        return 1;
    }
    if (args.head_dim > MAX_HEAD_DIM) {
        printf("head_dim must be <= %d\n", MAX_HEAD_DIM);
        return 1;
    }

    int64_t numel = static_cast<int64_t>(args.batch) * args.heads * args.seq_len * args.head_dim;
    int64_t score_el = static_cast<int64_t>(args.batch) * args.heads * args.seq_len * args.seq_len;
    size_t bytes_qkv = static_cast<size_t>(numel) * sizeof(float);
    size_t bytes_scores = static_cast<size_t>(score_el) * sizeof(float);
    float scale = 1.0f / sqrtf(static_cast<float>(args.head_dim));

    std::vector<float> h_q(numel), h_k(numel), h_v(numel), h_out_cutlass(numel), h_out_flash(numel);
    std::vector<float> h_out_regular;
    if (args.include_regular) {
        h_out_regular.resize(numel);
    }
    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int64_t i = 0; i < numel; ++i) {
        h_q[i] = dist(gen);
        h_k[i] = dist(gen);
        h_v[i] = dist(gen);
    }

    float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr, *d_scores = nullptr, *d_probs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, bytes_qkv));
    CUDA_CHECK(cudaMalloc(&d_k, bytes_qkv));
    CUDA_CHECK(cudaMalloc(&d_v, bytes_qkv));
    CUDA_CHECK(cudaMalloc(&d_o, bytes_qkv));
    CUDA_CHECK(cudaMalloc(&d_scores, bytes_scores));
    CUDA_CHECK(cudaMalloc(&d_probs, bytes_scores));
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), bytes_qkv, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), bytes_qkv, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), bytes_qkv, cudaMemcpyHostToDevice));

    printf("shape (B,H,S,D)=(%d,%d,%d,%d) causal=%d\n", args.batch, args.heads, args.seq_len, args.head_dim, args.causal);
    if (args.amortize_warmup) {
        printf("timing: amortized over warmup+repeat=%d\n", args.warmup + args.repeat);
    } else {
        printf("timing: steady over repeat=%d (warmup=%d not counted)\n", args.repeat, args.warmup);
    }
    if (args.profile) {
        printf("profiling enabled via cudaProfilerStart/Stop\n");
    }
    if (!args.include_regular) {
        printf("running CUTLASS + flash only (pass --include-regular to also benchmark regular CUDA attention)\n");
    }

    printf("running CUTLASS benchmark...\n");
    float ms_cutlass = benchmark_cutlass(d_q, d_k, d_v, d_o, d_scores, d_probs, args, scale);
    CUDA_CHECK(cudaMemcpy(h_out_cutlass.data(), d_o, bytes_qkv, cudaMemcpyDeviceToHost));
    printf("CUTLASS attention: %.4f ms/call\n", ms_cutlass);

    printf("running flash benchmark...\n");
    float ms_flash = benchmark_flash(d_q, d_k, d_v, d_o, args, scale);
    CUDA_CHECK(cudaMemcpy(h_out_flash.data(), d_o, bytes_qkv, cudaMemcpyDeviceToHost));
    printf("CUDA flash attention: %.4f ms/call\n", ms_flash);

    float ms_regular = 0.0f;
    if (args.include_regular) {
        printf("running regular benchmark (this can be very slow)...\n");
        ms_regular = benchmark_regular(d_q, d_k, d_v, d_o, args, scale);
        CUDA_CHECK(cudaMemcpy(h_out_regular.data(), d_o, bytes_qkv, cudaMemcpyDeviceToHost));
        printf("CUDA regular attention: %.4f ms/call\n", ms_regular);
    }

    if (args.check) {
        float diff = max_abs_diff(h_out_cutlass, h_out_flash);
        printf("max |cutlass - flash|: %.6f\n", diff);
        if (args.include_regular && !h_out_regular.empty()) {
            float diff_reg_flash = max_abs_diff(h_out_regular, h_out_flash);
            printf("max |regular - flash|: %.6f\n", diff_reg_flash);
        }
    }
    printf("speedup (cutlass / flash): %.3fx\n", ms_cutlass / ms_flash);
    if (args.include_regular && !h_out_regular.empty()) {
        printf("speedup (regular / flash): %.3fx\n", ms_regular / ms_flash);
    }

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_o));
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_probs));
    return 0;
}
