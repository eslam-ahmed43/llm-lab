# Exp 1: Serving and batching on a single T4

*Qwen2.5-1.5B-Instruct, fp16, Kaggle Tesla T4. Last updated 2026-10-02.*

## Abstract

We compare four ways of serving the same model under concurrent load: a one-request-at-a-time Transformers server, a Transformers server with static batching, vLLM 0.30.0, and vLLM with automatic prefix caching. At 64 concurrent requests and 128-token prompts, static batching raises throughput 13.9x over the naive server, and vLLM adds another 3.8x on top of that. The gap narrows as prompts grow (2.1x at 2048 tokens) because prefill becomes the bottleneck on a T4. Time to first token under load differs even more: 0.816 s for vLLM against 16.8 s for static batching and 150.4 s for the naive server. Our first vLLM run was invalidated by prefix caching combined with identical prompts; we kept it as a measurement of the best case for caching and re-ran vLLM with caching disabled.

## 1. Setup

**Hardware and software.** Tesla T4 (15360 MiB, driver 580.178.04) on Kaggle, one GPU pinned with `CUDA_VISIBLE_DEVICES=0`. vLLM 0.30.0 runs in its own virtualenv (torch 2.13.0+cu132). The Transformers servers and the client use transformers 5.0.0, torch 2.10.0+cu128, Python 3.12.13. The model is Qwen/Qwen2.5-1.5B-Instruct in fp16, because the T4 has no bf16.

**Backends.**

| Name | Description |
|---|---|
| `hf_naive` | FastAPI server, one `generate()` at a time behind a lock, `TextIteratorStreamer` |
| `hf_batched` | Static batching: wait up to 20 ms, group up to 16 requests, run one left-padded `generate()`; the next batch starts only when the previous one has finished. Tokens are streamed per request |
| `vllm` | vLLM 0.30.0, default scheduler, `--max-model-len 4096 --gpu-memory-utilization 0.85`, prefix caching **disabled** |
| `vllm_prefixcache` | Same as `vllm` with automatic prefix caching enabled (the vLLM default) |

**Workload.** A closed-loop load generator (`src/bench_inference.py`) sends streaming `/v1/completions` requests to an OpenAI-compatible server. Prompts have 128, 512 or 2048 tokens (a fixed sentence repeated, identical for every request), each request produces exactly 128 tokens (`ignore_eos`), and concurrency is 1, 4, 16 or 64. Each level sends `max(8, k x concurrency)` requests with k = 4 for vLLM, 1 for `hf_naive` and 2 for `hf_batched` (the slower backends get fewer requests to keep the run time reasonable). One warm-up pass precedes each run. Every (backend, prompt length) pair is run 3 times; tables show the mean and, for throughput, the standard deviation over those 3 runs.

**Metrics.**

- *Throughput*: total generated tokens divided by the wall time of the concurrency level.
- *TTFT*: time from sending the request to receiving the first non-empty streamed chunk. **It includes queueing time**, which is the point: under load, waiting is part of the user-visible latency. The naive server streams at word boundaries, vLLM and the batched server per token, so naive TTFT can be slightly pessimistic at concurrency 1.
- *Peak memory*: maximum `nvidia-smi` memory.used sampled every 250 ms. It is cumulative over a server's lifetime (the PyTorch allocator does not give memory back), and vLLM reserves its KV cache up front, so this metric is not a like-for-like efficiency comparison.

## 2. Results

### Prompt length 128 tokens

Output throughput (tokens/s, mean ± std over 3 repeats):

| Concurrency | HF naive | HF batched | vLLM | vLLM + prefix cache |
|---:|---:|---:|---:|---:|
| 1 | 26.8 ± 0.1 | 27.4 ± 0.1 | 53.1 ± 8.3 | 64.0 ± 0.7 |
| 4 | 26.7 ± 0.2 | 109.4 ± 0.8 | 235.1 ± 4.5 | 247.9 ± 0.6 |
| 16 | 26.9 ± 0.2 | 372.4 ± 2.9 | 705.9 ± 16.6 | 799.8 ± 4.1 |
| 64 | 26.8 ± 0.2 | 372.7 ± 2.6 | 1419.2 ± 6.0 | 1800.9 ± 11.1 |

TTFT in seconds (p50 / p95, mean over 3 repeats):

| Concurrency | HF naive | HF batched | vLLM | vLLM + prefix cache |
|---:|---:|---:|---:|---:|
| 1 | 0.045 / 0.048 | 0.063 / 0.073 | 0.043 / 0.045 | 0.026 / 0.033 |
| 4 | 14.4 / 14.5 | 0.107 / 0.108 | 0.111 / 0.120 | 0.047 / 0.058 |
| 16 | 35.6 / 71.4 | 0.375 / 0.380 | 0.371 / 0.392 | 0.085 / 0.097 |
| 64 | 150.4 / 286.9 | 16.8 / 16.9 | 0.816 / 1.48 | 0.213 / 0.317 |

### Prompt length 512 tokens

Output throughput (tokens/s, mean ± std over 3 repeats):

| Concurrency | HF naive | HF batched | vLLM | vLLM + prefix cache |
|---:|---:|---:|---:|---:|
| 1 | 26.1 ± 0.2 | 27.0 ± 0.1 | 52.3 ± 1.4 | 62.7 ± 0.3 |
| 4 | 25.9 ± 0.4 | 103.3 ± 0.4 | 186.3 ± 0.3 | 227.0 ± 0.9 |
| 16 | 26.0 ± 0.2 | 190.5 ± 0.2 | 407.5 ± 0.9 | 647.0 ± 3.6 |
| 64 | 26.2 ± 0.1 | 189.6 ± 0.0 | 605.6 ± 0.3 | 1246.3 ± 6.3 |

TTFT in seconds (p50 / p95, mean over 3 repeats):

| Concurrency | HF naive | HF batched | vLLM | vLLM + prefix cache |
|---:|---:|---:|---:|---:|
| 1 | 0.116 / 0.120 | 0.131 / 0.136 | 0.130 / 0.137 | 0.031 / 0.038 |
| 4 | 14.8 / 15.0 | 0.468 / 0.472 | 0.467 / 0.471 | 0.058 / 0.067 |
| 16 | 37.3 / 73.9 | 1.72 / 1.72 | 1.05 / 1.87 | 0.130 / 0.147 |
| 64 | 154.3 / 293.1 | 34.1 / 34.2 | 1.38 / 6.09 | 0.288 / 0.450 |

### Prompt length 2048 tokens

Output throughput (tokens/s, mean ± std over 3 repeats):

| Concurrency | HF naive | HF batched | vLLM | vLLM + prefix cache |
|---:|---:|---:|---:|---:|
| 1 | 23.0 ± 0.1 | 23.7 ± 0.1 | 39.2 ± 0.2 | 56.4 ± 0.5 |
| 4 | 23.2 ± 0.1 | 45.0 ± 0.0 | 77.5 ± 0.1 | 179.6 ± 2.7 |
| 16 | 23.1 ± 0.1 | 52.8 ± 0.1 | 103.9 ± 0.1 | 384.6 ± 5.4 |
| 64 | 23.1 ± 0.1 | 52.8 ± 0.0 | 112.6 ± 0.1 | 560.9 ± 0.8 |

TTFT in seconds (p50 / p95, mean over 3 repeats):

| Concurrency | HF naive | HF batched | vLLM | vLLM + prefix cache |
|---:|---:|---:|---:|---:|
| 1 | 0.792 / 0.842 | 0.789 / 0.821 | 0.935 / 0.956 | 0.050 / 0.051 |
| 4 | 17.3 / 17.4 | 2.98 / 3.00 | 2.81 / 3.81 | 0.102 / 0.129 |
| 16 | 42.5 / 83.8 | 12.7 / 12.9 | 3.68 / 12.8 | 0.214 / 0.334 |
| 64 | 175.1 / 332.6 | 129.1 / 129.2 | 3.96 / 49.7 | 0.546 / 1.07 |

### Peak GPU memory (GiB, highest value over all levels of a prompt length)

| Prompt tokens | HF naive | HF batched | vLLM | vLLM + prefix cache |
|---:|---:|---:|---:|---:|
| 128 | 3.2 | 3.5 | 12.6 | 12.5 |
| 512 | 3.3 | 4.3 | 12.6 | 12.5 |
| 2048 | 3.8 | 12.6 | 12.6 | 12.5 |

## 3. Analysis

### 3.1 Batching dominates at short prompts

The naive server is serialized, so its throughput does not move with load: about 26 tok/s at 128 tokens regardless of concurrency. Static batching changes the picture (13.9x at 128 tokens, 7.2x at 512, 2.3x at 2048 for concurrency 64). The decode step time hints at why. For `hf_batched` at 128-token prompts, the step time estimated from p50 latency and TTFT is about 36 ms at concurrency 1 and 40 ms at concurrency 16: sixteen times more tokens per step for roughly 11% more time. That suggests the per-step cost is dominated by work that does not grow with the number of sequences (reading the weights, launching kernels), and batching amortizes it.

Batched throughput is identical at concurrency 16 and 64 (372 and 373 tok/s at 128 tokens) because the batch size is capped at 16. The extra requests at concurrency 64 simply wait, which is why batched TTFT jumps to 16.8 s.

### 3.2 vLLM versus static batching

At concurrency 64, vLLM is 3.8x / 3.2x / 2.1x faster than static batching for 128 / 512 / 2048-token prompts, and 53x / 23x / 4.9x faster than the naive server. Three effects are visible in the data:

- **Continuous batching removes the batch cap and the wait for the slowest sequence.** vLLM keeps scaling past 16 concurrent requests (27x throughput from concurrency 1 to 64 at 128 tokens) and its TTFT stays at 0.816 s instead of 16.8 s.
- **Lower per-step overhead.** At concurrency 1 and 128-token prompts vLLM generates 53 tok/s against 27 for Transformers. We did not isolate the cause here. Exp 6 later measured part of it (see `REPORT_EXP6.md`): in a plain eager loop the GPU sits idle for roughly 30% of each step waiting for kernel launches, and CUDA graphs recover part of that.
- **Memory is managed per token, not per padded batch.** `hf_batched` reached 12.6 GiB (an allocator high-water mark) at 2048-token prompts with a batch of 16, close to the 15 GiB device limit, so a larger batch would likely run out of memory. vLLM's KV cache of 8.67 GiB holds 324,704 tokens (about 79 concurrent requests of 4096 tokens, as reported in its startup log).

### 3.3 Long prompts make the T4 prefill-bound

At 2048 tokens vLLM saturates early: 104 tok/s at concurrency 16 and 113 at 64, a total scale-up of only 2.9x from concurrency 1. A single 2048-token prefill takes 0.94 s, which is about 2190 prompt tokens/s. At concurrency 64 the initial burst needs 64 x 2048 = 131,072 prompt tokens, roughly 60 s of pure prefill at that rate, the same order as the observed p95 TTFT of 49.7 s. At concurrency 4 the TTFT of vLLM and static batching is nearly identical (2.81 s versus 2.98 s): both are waiting for the same prefill compute, and vLLM only wins in the decode phase.

### 3.4 An unexpected single-stream result

For a lone 2048-token request, vLLM's TTFT (0.935 s) is about 18% slower than Transformers (0.792 s). At 128 and 512 tokens the two are within noise. A plausible explanation is the attention backend available on a Turing GPU (no FlashAttention 2), but we have not verified it. This is a good target for profiling.

### 3.5 Prefix caching (preliminary Exp 3)

The `vllm_prefixcache` run is the best case for caching, because every request carries the same prompt, so after the first request nearly the whole prompt comes from the cache. Compared with the uncached run:

| Prompt tokens | Throughput gain at c=64 | TTFT p50 at c=1 (no cache to cache) | TTFT p50 at c=64 (no cache to cache) |
|---:|---:|---:|---:|
| 128 | 1.27x | 0.043 s to 0.026 s | 0.816 s to 0.213 s |
| 512 | 2.06x | 0.130 s to 0.031 s | 1.383 s to 0.288 s |
| 2048 | 4.98x | 0.935 s to 0.050 s | 3.961 s to 0.546 s |

The gain grows with prompt length because the cache removes prefill work, which is exactly the cost that dominates long prompts. vLLM's log reported prefix-cache hit rates of this order in the cached run (for example 76.6% in one logged interval) and 0.0% in the uncached run. Real traffic will sit between the two: the benefit depends on the fraction of each prompt that is a shared prefix, which is what the planned Exp 3 sweep will measure.

## 4. A methodology problem we caught

The first vLLM run used identical prompts with automatic prefix caching on, which is vLLM's default. Two signals exposed it: vLLM's TTFT for a single 2048-token prompt was 0.05 s while Transformers needed 0.79 s for the same prefill (implausibly fast for a T4), and the server log reported non-zero prefix-cache hit rates. The comparison against Transformers, which has no prefix cache, was therefore unfair to Transformers. We re-ran vLLM with `--no-enable-prefix-caching` (a smoke test confirmed a hit rate of 0.0% and a realistic TTFT of 0.87 s), and kept the original numbers as `vllm_prefixcache`, where they now document the effect of caching. The Transformers backends are unaffected, since they have no prefix cache.

## 5. Practical takeaways

For a roughly 1.5B-parameter fp16 model on a 16 GB T4, 128 output tokens per request:

1. **Never serve concurrent users with a one-request-at-a-time loop.** Even simple static batching gives 2.3x to 14x more throughput.
2. **Use a continuous-batching engine for interactive traffic.** It gives 2.1x to 3.8x more throughput than static batching, and keeps queueing delay at seconds instead of tens of seconds.
3. **Expect prompt length to matter more than concurrency once prompts reach about 2k tokens.** Throughput saturates around concurrency 16 and the TTFT tail grows quickly, so limit concurrency or shorten prompts rather than adding load.
4. **Reuse prefixes when you can.** Shared system prompts or documents can give several times more throughput on long prompts.
5. **Plan memory around the KV cache, not the weights.** The weights take about 3 GiB; the rest of the card is cache capacity, and vLLM reports it directly in its startup log.

## 6. Limitations

- One model, one GPU type, one precision. Conclusions about ratios may shift on other hardware.
- All requests in a level share one prompt and one output length (128 tokens, forced). Mixed output lengths would expose head-of-line blocking in static batching, which this workload hides (it is the best case for static batching).
- The number of requests per level differs between backends (k = 4, 1, 2), and small levels (8 requests at low concurrency) give coarse p95 values.
- `hf_batched` uses an arbitrary batch cap of 16 and a 20 ms wait window. It also pays that 20 ms in TTFT at concurrency 1.
- Peak memory is cumulative over a server's lifetime and vLLM pre-allocates, so it cannot be compared across backends as an efficiency measure.
- Run-to-run variation is small in most cells, but vLLM at concurrency 1 with 128-token prompts varied between about 47 and 62 tok/s across repeats; treat that cell as roughly ±15%.
- The raw `hf_batched` result files were rebuilt from the printed log of the original run (see `results/hf_batched_SOURCE.txt`); the values keep full float precision.
- Causes given in 3.2 and 3.4 are hypotheses; none was isolated with a profiler.

## 7. Reproducibility

Everything needed is in the repository: the client (`src/bench_inference.py`), the three server configurations, the raw JSON results (`results/*_p{128,512,2048}_r{0,1,2}.json`), the aggregated tables (`results/all_runs.csv`, `results/summary.csv`), the environment records (`results/env*.json`) and the server logs. The Kaggle notebooks that drove the runs skip any result file that already exists, so interrupted sessions can resume.

## 8. What comes next

- **Exp 2, quantization**: done, see `REPORT_EXP2.md`. Next there: AWQ or GPTQ checkpoints and quantization combined with vLLM serving.
- **Exp 3, prefix caching sweep**: vary the fraction of each prompt that is a shared prefix, with the cache on and off.
