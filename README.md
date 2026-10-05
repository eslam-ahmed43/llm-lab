# llm-lab: LLM inference experiments on a single T4

Measured, reproducible experiments on LLM serving, compression, fine-tuning and model internals, run on one free Kaggle **Tesla T4 (15 GiB)** with **Qwen2.5-1.5B-Instruct** in fp16. Every number in the reports traces back to a raw result file and a script in this repo.

## Status

| Experiment | Question | Status |
|---|---|---|
| **Exp 1: Serving and batching** | How much do batching and the serving engine matter under concurrency? | Done: [REPORT.md](REPORT.md) |
| **Exp 2: Quantization** | What do INT8 / INT4 cost and buy in memory, latency, throughput and quality? | Done: [REPORT_EXP2.md](REPORT_EXP2.md) |
| **Exp 3: Prefix caching** | How much does a shared prefix help, and how does it scale with prefix length? | Preliminary: the extreme case (100% shared prompt) is in the Exp 1 report. The shared-prefix-length sweep is planned |
| **Exp 4: LoRA and QLoRA fine-tuning** | How do rank, adapted layers and a 4-bit base trade accuracy, memory and time? | Done: [REPORT_EXP4.md](REPORT_EXP4.md) |
| **Exp 5: Transformer internals** | Does a from-scratch decoder match Hugging Face, and does our LoRA match PEFT? | Done: [REPORT_EXP5.md](REPORT_EXP5.md) |
| **Exp 6: Decode profiling** | Where does single-stream decode time go, and do CUDA graphs remove the overhead? | Done: [REPORT_EXP6.md](REPORT_EXP6.md) |

## Exp 1 in one table

Output throughput in tokens/s at **64 concurrent requests**, 128 generated tokens per request (mean of 3 repeats):

| Prompt tokens | HF naive | HF batched | vLLM | vLLM + prefix cache |
|---:|---:|---:|---:|---:|
| 128 | 27 | 373 | 1419 | 1801 |
| 512 | 26 | 190 | 606 | 1246 |
| 2048 | 23 | 53 | 113 | 561 |

Time to first token (p50) at 64 concurrent requests, queueing included:

| Prompt tokens | HF naive | HF batched | vLLM | vLLM + prefix cache |
|---:|---:|---:|---:|---:|
| 128 | 150.4 s | 16.8 s | 0.816 s | 0.213 s |
| 512 | 154.3 s | 34.1 s | 1.38 s | 0.288 s |
| 2048 | 175.1 s | 129.1 s | 3.96 s | 0.546 s |

![Throughput vs concurrency](plots/throughput_vs_concurrency.png)
![TTFT vs concurrency](plots/ttft_vs_concurrency.png)

## Exp 1: what the numbers say

- **Batching is worth far more than any single kernel.** One-request-at-a-time serving stays flat at 23 to 27 tok/s no matter the load. Static batching (up to 16 requests) multiplies throughput by 13.9x / 7.2x / 2.3x for 128 / 512 / 2048-token prompts.
- **A continuous-batching engine beats static batching by 3.8x / 3.2x / 2.1x** (128 / 512 / 2048 tokens), and keeps TTFT in the sub-second to few-second range where static batching queues for 17 to 129 s.
- **The advantage shrinks as prompts get longer.** At 2048 tokens the T4 is prefill-bound (about 2190 prompt tokens/s) and vLLM saturates by concurrency 16.
- **Prefix caching is the biggest single lever when prompts share a prefix:** 1.27x / 2.06x / 4.98x throughput at 64 concurrent requests for 128 / 512 / 2048 tokens when every request uses the same prompt. That is an upper bound, not a typical gain.
- **A methodology bug was caught and fixed.** The first vLLM run used identical prompts with prefix caching on, which inflated its numbers. It is kept as `vllm_prefixcache` and re-measured with caching off (see REPORT.md, section 4).

## Exp 2 in one table

Qwen2.5-1.5B-Instruct on a T4 with bitsandbytes. Throughput is generated tokens/s at batch 1 and batch 16 (512-token prompts, 128 new tokens):

| Method | Weights (GiB) | WikiText-2 PPL | ARC-Easy acc | Throughput, batch 1 | Throughput, batch 16 |
|---|---:|---:|---:|---:|---:|
| FP16 | 2.88 | 9.65 | 0.750 | 27.6 | 185.3 |
| INT8 (bitsandbytes) | 1.66 | 9.70 | 0.740 | 7.0 | 78.8 |
| NF4 (bitsandbytes) | 1.04 | 10.42 | 0.694 | 20.4 | 172.8 |

- **Neither format made inference faster.** NF4 reaches 0.74x of FP16 throughput at batch 1 and 0.93x at batch 16. bitsandbytes INT8 decodes 2.6x to 4.0x slower.
- **INT8 is nearly lossless but a poor trade here:** perplexity +0.6%, but only 42% less weight memory for a large slowdown.
- **NF4 saves the most memory** (2.8x smaller weights) for about 8% higher perplexity and 5.6 points lower ARC-Easy accuracy.
- **The savings are below the nominal 2x and 4x** because the 0.23B embedding parameters stay in FP16.

![Quantization overview](plots/quant_overview.png)

## Exp 4 in one table

LoRA written from scratch, Qwen2.5-1.5B-Instruct fine-tuned on 6-way emotion classification, 3 seeds per setup, 600 steps on a T4:

| Setup | Trainable params | Test accuracy (%) | Peak memory (GiB) | Train time (s) |
|---|---:|---:|---:|---:|
| No fine-tuning | 0 | 51.2 | - | - |
| LoRA, attention, rank 8 | 2.2M | 91.18 ± 0.16 | 7.37 | 316 |
| LoRA, all linear, rank 4 | 4.6M | 92.17 ± 0.10 | 9.07 | 397 |
| LoRA, all linear, rank 8 | 9.2M | 92.80 ± 0.28 | 9.15 | 395 |
| LoRA, all linear, rank 32 | 36.9M | 92.85 ± 0.35 | 9.66 | 405 |
| QLoRA, all linear, rank 8 | 9.2M | 92.78 ± 0.18 | 7.40 | 427 |

- **Fine-tuning takes accuracy from 51% to about 93%** while training under 2.5% of the parameters.
- **With all linear layers adapted, rank 8 is enough:** ranks 8, 16 and 32 are within 0.15 points of each other, so rank 32 pays 4x the parameters for nothing measurable.
- **With attention only, rank matters:** accuracy rises from 89.3% to 92.5% between rank 4 and 32.
- **QLoRA matches LoRA at rank 8** (92.78% against 92.80%) with 19% less peak memory and 8% more training time.
- Exp 4 ran on a newer Kaggle image (Python 3.13, torch 2.11, transformers 5.16) than Exp 1 and 2; the versions are stored in each result file.

![LoRA overview](plots/lora_overview.png)

## Exp 5 in one table

A Qwen2-style decoder written from scratch in PyTorch (RMSNorm, rotary embeddings, grouped-query attention, SwiGLU, KV cache), loaded with the real Qwen2.5-1.5B weights and compared with Hugging Face on a T4:

| | Max abs logit difference | Greedy outputs identical (3 prompts, 48 tokens) | Decode (ms per token) |
|---|---:|---:|---:|
| fp32, manual attention | 1.65e-05 | 3 of 3 | - |
| fp32, fused attention | 2.10e-05 | 3 of 3 | - |
| fp16, manual attention | 4.88e-02 | 2 of 3 | 33.0 |
| fp16, fused attention | 3.32e-02 | 2 of 3 | 27.4 |
| Hugging Face `generate` (fp16) | reference | reference | 37.0 |

- **In fp32 the from-scratch model matches Hugging Face to rounding error** (about 1e-6 relative), layer by layer.
- **In fp16 one prompt diverges at token 33,** where the reference's two best logits are only 0.016 apart, consistent with a near-tie flipped by rounding.
- **A real bug was caught:** the first manual attention returned NaN in fp16 because the scores overflowed; scaling q and k separately fixed it.
- **The plain decode loop is 1.35x faster than `generate`** for a single stream (batch of one only), which quantifies the framework overhead.
- **The LoRA layer from Exp 4 gives exactly the same logits as PEFT** with identical weights, and merging adapters into the weights matches (difference 9e-05, rounding).

## Exp 6 in one table

The decode step of the from-scratch model of Exp 5 (fp16, batch of one, T4), run four ways and profiled:

| Variant | Step time (ms) | CPU time to enqueue a step (ms) |
|---|---:|---:|
| Eager, growing cache, host sync per token | 27.1 | 27.1 |
| Eager, growing cache, no host sync | 24.5 | 24.5 |
| Eager, preallocated cache | 25.1 | 25.1 |
| Preallocated cache + CUDA graph | 21.2 | 20.3 |

- **The eager loop is CPU-bound:** enqueueing takes as long as the step, and the GPU is busy for only 17.5 of 24.5 ms (71%).
- **CUDA graphs make the step 1.28x faster** than the Exp 5 loop and GPU-bound.
- **The GPU work is mostly streaming weights:** the three largest kernels take 12.95 ms against 12.7 ms to read the weights once at the measured 243 GB/s.
- **Budget:** about 20 of Hugging Face's 37 ms per token is not GPU work, and vLLM's step (derived from Exp 1) is close to the GPU-work level.

![Decode time budget](plots/decode_budget.png)

## Setup (Exp 1 and 2)

| | |
|---|---|
| GPU | Tesla T4, 15360 MiB, driver 580.178.04 (Kaggle, one GPU pinned) |
| Model | Qwen/Qwen2.5-1.5B-Instruct, fp16 (the T4 has no bf16) |
| vLLM | 0.30.0 (torch 2.13.0+cu132), max-model-len 4096, gpu-memory-utilization 0.85 |
| Transformers servers | transformers 5.0.0, torch 2.10.0+cu128, Python 3.12.13 |
| Workload | closed loop, streaming `/v1/completions`, prompts of 128 / 512 / 2048 tokens, 128 forced output tokens, concurrency 1 / 4 / 16 / 64, 3 repeats |

## Repository layout

```
llm-lab/
├── README.md
├── REPORT.md                  # full Exp 1 report with all tables and caveats
├── REPORT_EXP2.md             # full Exp 2 (quantization) report
├── REPORT_EXP4.md             # full Exp 4 (LoRA / QLoRA) report
├── REPORT_EXP5.md             # full Exp 5 (transformer internals) report
├── REPORT_EXP6.md             # full Exp 6 (decode profiling) report
├── src/
│   ├── bench_inference.py     # load generator (TTFT, latency, throughput, peak memory)
│   ├── hf_server.py           # Transformers baseline: one generate() at a time
│   ├── hf_batched_server.py   # Transformers baseline: static batching
│   ├── quantize.py            # Exp 2: FP16 / INT8 / NF4 study (run + summarize)
│   ├── train_lora.py          # Exp 4: LoRA / QLoRA from scratch (run + summarize)
│   ├── plot_lora.py           # Exp 4 figure from results/lora_summary.csv
│   ├── exp5_internals.py      # Exp 5: from-scratch Qwen2 vs Hugging Face, LoRA vs PEFT
│   ├── exp6_profiling.py      # Exp 6: decode step overhead, CUDA graphs, profiler
│   └── plot_exp6.py           # Exp 6 figure from the results of Exp 1, 5 and 6
├── results/                   # raw JSON per run, all_runs.csv, summary.csv, quant_*.json, quant_summary.csv, lora_*.json, lora_summary.csv, internals_*.json, profiling_exp6.json, env files
├── plots/
└── logs/                      # server logs (KV cache size, prefix-cache hit rates)
```

## Reproduce

On Kaggle (GPU T4 x2, internet on), pin one GPU with `CUDA_VISIBLE_DEVICES=0`, start a server, then run the client:

```bash
# vLLM with prefix caching disabled
vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype float16 --max-model-len 4096 \
    --gpu-memory-utilization 0.85 --no-enable-prefix-caching --port 8000

python src/bench_inference.py --model Qwen/Qwen2.5-1.5B-Instruct --backend-name vllm \
    --prompt-len 512 --max-tokens 128 --requests-per-level 4 --out results/vllm_p512_r0.json
```

The Transformers baselines are started with `python src/hf_server.py --model ...` and `python src/hf_batched_server.py --model ... --max-batch 16 --max-wait-ms 20`, and benchmarked with the same client.

## Reproduce Exp 2

```bash
python src/quantize.py run --method fp16 --out results/quant_fp16.json
python src/quantize.py run --method bnb-int8 --out results/quant_bnb-int8.json
python src/quantize.py run --method bnb-nf4 --out results/quant_bnb-nf4.json
python src/quantize.py summarize --results-dir results --out results/quant_summary.csv
```

## Reproduce Exp 4

```bash
python src/train_lora.py run --mode lora --rank 8 --targets all --seed 0 --out results/lora_lora_r8_all_s0.json
python src/train_lora.py run --mode qlora --rank 8 --targets all --seed 0 --out results/lora_qlora_r8_all_s0.json
python src/train_lora.py summarize --results-dir results --out results/lora_summary.csv
python src/plot_lora.py --summary results/lora_summary.csv --out plots/lora_overview.png
```

## Reproduce Exp 5

```bash
python src/exp5_internals.py selftest --out results/internals_selftest.json
python src/exp5_internals.py qwen-verify --dtypes fp32 fp16 --bench --out results/internals_qwen_verify.json
python src/exp5_internals.py lora-peft --rank 8 --out results/internals_lora_peft.json
```

## Reproduce Exp 6

```bash
python src/exp6_profiling.py run --check --bench --profile --repeats 5 --out results/profiling_exp6.json
python src/plot_exp6.py --results results --out plots/decode_budget.png
```

## Known caveats

- The raw `hf_batched_*.json` files were lost with a Kaggle draft session and rebuilt from the run's printed log (full float precision). `results/hf_batched_SOURCE.txt` documents this. Re-running that backend takes about 20 minutes.
- All requests share one prompt per level and one output length, so head-of-line blocking is not exercised. See REPORT.md, section 6, for the full list.
