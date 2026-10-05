# Exp 5: Transformer internals, verified against Hugging Face

*A Qwen2-style decoder written from scratch in PyTorch, checked against Hugging Face on Qwen2.5-1.5B-Instruct, and the LoRA layer from Exp 4 checked against PEFT. Kaggle Tesla T4. Last updated 2026-10-04.*

## Abstract

We wrote a decoder-only transformer in plain PyTorch (RMSNorm, rotary embeddings, grouped-query attention, SwiGLU MLP, KV cache, greedy decoding), loaded the real Qwen2.5-1.5B-Instruct weights into it and compared it with the Hugging Face implementation. In fp32 the two agree to rounding error (largest logit difference about 2e-5 on logits up to 22, identical greedy outputs on all 3 prompts for 48 tokens). In fp16 the differences are larger, as expected for 16-bit arithmetic (about 3e-2 to 5e-2), and one of three prompts diverges at token 33, where the reference's own two best logits are only 0.016 apart. A first version of the manual attention produced NaN in fp16 because the attention scores overflowed; scaling q and k separately fixed it. In single-stream decoding on the T4, the lean implementation runs at 27.4 ms per token against 37.0 for Hugging Face `generate`, which quantifies the framework overhead of generic generation. Separately, the LoRA layer from Exp 4 produces exactly the same logits as PEFT when both use the same adapter weights, and merging the adapters into the base weights gives matching outputs.

## 1. Setup

**What was implemented** (`src/exp5_internals.py`, module names mirror Hugging Face so its weights load directly):

- RMSNorm with the statistics computed in fp32.
- Rotary position embeddings in the half-rotation convention, with cos and sin computed in fp32 and cast to the activation dtype.
- Grouped-query attention: 12 query heads sharing 2 key/value heads of dimension 128 (hidden size 1536), biases on the q, k and v projections, a causal mask with an offset for cached decoding, and the compact KV heads stored in the cache.
- SwiGLU MLP with 8,960 hidden units, pre-norm residual blocks, 28 layers, a tied output head.
- Greedy decoding with and without the KV cache. Two attention paths: a manual one (scores, mask, fp32 softmax) and one that calls PyTorch's fused `scaled_dot_product_attention`.

**Verification protocol.** Three short prompts, 48 new tokens each, batch of one. We compare: the logits for every position, the output of every decoder layer (stored for the first prompt, to locate where a mismatch would start), the greedy tokens against Hugging Face, and the logits produced with the KV cache against a full recomputation. The instruct model's default generation settings include a repetition penalty, so the Hugging Face side is run with `repetition_penalty=1.0` to make both sides plain greedy decoding. For tokens that differ, we also record the gap between the reference's two best logits at that step. Before the real model, a self-test on a tiny random Qwen2 model checks the logic without any download.

**Speed.** fp16, batch of one, 512-token prompt, 128 new tokens, one warm-up, 3 timed repeats. Decode latency is `(total time - prefill time) / 127`. Hugging Face uses `generate`; the own implementation uses a plain Python loop with the KV cache.

**LoRA against PEFT.** fp32, rank 8, alpha 16, adapters on all linear layers, no dropout. The adapter weights of the own implementation (with `B` filled with small random values, so the adapters change the output) are copied into a PEFT model, and the logits are compared on one prompt, before and after merging the adapters into the base weights.

**Environment.** Tesla T4, Python 3.13.15, torch 2.11.0+cu128, transformers 5.16.1, peft 0.20.0. PEFT 0.20.0 refuses to run with the `torchao` 0.10.0 that comes preinstalled in the Kaggle image, so `torchao`, which this project does not use, was uninstalled before the PEFT comparison.

## 2. Results

### 2.1 Correctness against Hugging Face

Logits and top-1 agreement are for the first prompt; the largest logit is about 22. The per-prompt and per-layer numbers are in `results/internals_qwen_verify.json`.

| Precision | Attention | Max abs logit difference | Relative to the largest logit | Top-1 agreement | Greedy outputs identical (of 3 prompts) | First divergence |
|---|---|---:|---:|---:|---:|---|
| fp32 | manual | 1.65e-05 | 7.5e-07 | 1.000 | 3 | none |
| fp32 | sdpa | 2.10e-05 | 9.5e-07 | 1.000 | 3 | none |
| fp16 | manual | 4.88e-02 | 2.2e-03 | 1.000 | 2 | token 33 (prompt 2) |
| fp16 | sdpa | 3.32e-02 | 1.5e-03 | 1.000 | 2 | token 33 (prompt 2) |

The self-test on the tiny random model gives a largest logit difference of 1.8e-6 (manual) and 0.0 (fused), identical greedy outputs, and a KV-cache result within 1.3e-6 of the full recomputation.

### 2.2 Single-stream speed (fp16, T4)

| | Prefill, 512 tokens (s) | Decode (ms per token) | Tokens/s | Decode speed-up vs Hugging Face |
|---|---:|---:|---:|---:|
| Hugging Face `generate` | 0.105 | 37.0 | 26.6 | 1.00x |
| Own implementation, manual attention | 0.102 | 33.0 | 29.8 | 1.12x |
| Own implementation, fused attention (SDPA) | 0.087 | 27.4 | 35.8 | 1.35x |

The same measurement in an earlier run of this notebook gave 36.7 ms per token for Hugging Face and 27.3 for the fused path, so these figures reproduce to about 1%. (The manual-attention speed in that earlier run is not used, because that version produced NaN.)

### 2.3 LoRA against PEFT (fp32, rank 8)

| Comparison | Max abs logit difference | Top-1 agreement |
|---|---:|---:|
| Base model against model with adapters (sanity: the adapters matter) | 8.54 | 0.909 |
| **Own LoRA against PEFT** (same adapter weights) | **0.00** | 1.000 |
| Own LoRA, unmerged against merged | 9.0e-05 | 1.000 |
| PEFT, unmerged against merged | 9.0e-05 | 1.000 |
| Own merged against PEFT merged | 0.00 | 1.000 |

Trainable parameters: 9,232,384 for both, on 196 adapted modules.

## 3. Analysis

### 3.1 fp32: the implementation is correct

The largest difference of about 2e-5 on logits of magnitude 22 is a relative error near 1e-6, which is float32 rounding. It is the same for the manual and the fused attention, the tokens are identical, the KV cache reproduces the full computation, and the check passed for all 3 prompts for 48 tokens. The per-layer differences for the first prompt are stored in the results file to locate a mismatch if one appears; agreement at rounding level in the final logits means a mismatch in any layer would have propagated to them.

### 3.2 fp16: larger differences, one near-tie

In fp16 the largest logit difference is 3.3e-02 for the fused attention and 4.9e-02 for the manual one, a relative error of about 1.5e-3 to 2.2e-3, as expected for a format with roughly three decimal digits. Top-1 agreement on the first prompt is still 1.000. On one prompt (of three) the greedy output diverges at token 33, for both attention paths, and at that step the reference's two best logits differ by 0.016, which is smaller than the typical fp16 logit error. That is consistent with a near-tie flipped by rounding rather than with a defect, and the same implementation matches exactly in fp32. We have only one such case, so we do not claim this explains every possible divergence.

### 3.3 A bug that only fp16 could reveal

The first version of the manual attention computed `(q kᵀ) x scale` and returned NaN in fp16 (maximum difference `nan`, top-1 agreement 0.000), while it was exact in fp32 and the fused path was fine in fp16. The likely cause is that the unscaled scores exceed the largest fp16 value (65,504) and become infinite, which turns the softmax into NaN. Scaling `q` and `k` by `head_dim^-0.25` each before the product, as PyTorch's reference attention does, removed the NaN, which supports this explanation without isolating it. The lesson is general: tests at a precision other than the deployed one can pass while the deployed path is broken. The self-test could not catch it either, since it runs in fp32.

### 3.4 What the speed comparison shows

On this T4, the plain loop decodes at 27.4 ms per token, 1.35x faster than Hugging Face `generate` (37.0 ms), and prefill is 1.21x faster. The own implementation handles only a batch of one with no padding, sampling or stopping criteria, so this is not a claim that it is better. It measures what the generic machinery (logits processing, cache management, per-step bookkeeping) costs when the model is small: about 9.6 ms per token out of 37.0. Hugging Face `generate` here runs at 26.6 tokens/s, in line with the naive Transformers server in Exp 1 (26.8 tokens/s), and vLLM reached 53 tokens/s at concurrency 1 there. A large part of the single-stream gap between those systems is therefore overhead rather than GPU work. Exp 6 tests the remaining candidates (kernel-launch gaps and CUDA graphs) and finds that about half of Hugging Face's per-token time is CPU-side overhead; see `REPORT_EXP6.md`. The fused attention is faster than the manual path (33.0 against 27.4 ms), probably because it replaces several small kernels and intermediate tensors with one, but that is a hypothesis.

### 3.5 The LoRA layer from Exp 4 matches PEFT

With identical adapter weights the logits are exactly equal (difference 0.0), the parameter counts are identical, and merging the adapters into the base weights changes the output by about 9e-5 in both implementations, which is floating-point rounding when adding the low-rank update into the weights. The two merged models agree with each other exactly. The comparison is not vacuous: the adapters move the logits by up to 8.5 and change the top-1 prediction at about 9% of the positions compared with the base model (top-1 agreement 0.909). This closes the limitation noted in Exp 4: the forward pass of the from-scratch LoRA is the same as the library's.

## 4. Limitations

- The correctness numbers are for three short prompts and 48 generated tokens; they show agreement, not a proof of equivalence on all inputs. Only batch size 1 and sequences without padding are supported.
- fp16 and fp32 only, one model (Qwen2.5-1.5B-Instruct), one GPU type.
- The speed comparison is single-stream, with one prompt length (512) and one output length (128). The own implementation omits features that the Hugging Face path provides, so the comparison measures overhead, not quality of engineering.
- The LoRA comparison covers one rank, one prompt, fp32, random (not trained) adapters and no dropout. It checks the forward pass and merging with copied weights; it does not compare initialization or training dynamics with PEFT.
- The diagnosis of the fp16 NaN and the explanations in 3.4 are hypotheses supported by the fix and by the numbers, not isolated with a profiler or a controlled experiment.

## 5. Reproducibility

`src/exp5_internals.py` has three subcommands and writes JSON files (`results/internals_selftest.json`, `internals_qwen_verify.json`, `internals_lora_peft.json`, each with an environment block). The Kaggle notebook clones this repository and runs them.

```bash
pip uninstall -y torchao   # only if PEFT complains about the preinstalled torchao
python src/exp5_internals.py selftest --out results/internals_selftest.json
python src/exp5_internals.py qwen-verify --dtypes fp32 fp16 --bench --out results/internals_qwen_verify.json
python src/exp5_internals.py lora-peft --rank 8 --out results/internals_lora_peft.json
```

## 6. What comes next

- Train a small GPT from scratch with the same building blocks, to cover the training side of the transformer.
- Compare the initialization and a short training run of the own LoRA with PEFT, not only the forward pass.
- Done in Exp 6: profiling of the single-stream decode step. Still open: why the fused attention path is faster than the manual one.
