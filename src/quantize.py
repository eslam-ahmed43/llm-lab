"""Quantization study for causal LMs on a single GPU (Exp 2).

Compares FP16 against bitsandbytes INT8 / NF4 / FP4 and pre-quantized checkpoints
(for example AWQ or GPTQ repos) on four axes, all measured with the same code path:

  1. memory     : weight footprint after load, peak allocated memory during generation
  2. latency    : prefill time and per-token decode latency
  3. throughput : generated tokens/s for several batch sizes (static batching, HF generate)
  4. quality    : WikiText-2 perplexity and ARC-Easy accuracy

Each method runs in its own process and writes one JSON file, so GPU memory and CUDA
state never leak between runs. Merge the JSON files with the `summarize` subcommand.

Examples:
  python quantize.py run --method fp16 --out results/quant_fp16.json
  python quantize.py run --method bnb-int8 --out results/quant_bnb-int8.json
  python quantize.py run --method bnb-nf4 --out results/quant_bnb-nf4.json
  python quantize.py run --method prequant --name awq-int4 \
      --model Qwen/Qwen2.5-1.5B-Instruct-AWQ --out results/quant_awq-int4.json
  python quantize.py summarize --results-dir results --out results/quant_summary.csv
"""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

GIB = 1024 ** 3
METHODS = ["fp16", "bnb-int8", "bnb-nf4", "bnb-fp4", "prequant"]


# --------------------------------------------------------------------------- loading
def build_quant_config(method):
    """Return a BitsAndBytesConfig for bitsandbytes methods, None otherwise."""
    if method == "bnb-int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if method in ("bnb-nf4", "bnb-fp4"):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=method.split("-")[1],  # "nf4" or "fp4"
            bnb_4bit_compute_dtype=torch.float16,       # T4 has no bf16
            bnb_4bit_use_double_quant=False,
        )
    return None


def load_model(model_id, method):
    kwargs = {"device_map": {"": 0}}
    quant = build_quant_config(method)
    if quant is not None:
        kwargs["quantization_config"] = quant
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, **kwargs)
    except TypeError:
        # Older transformers versions use torch_dtype instead of dtype
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, **kwargs)
    return model.eval()


def env_info():
    import transformers
    info = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
    }
    try:
        import bitsandbytes
        info["bitsandbytes"] = bitsandbytes.__version__
    except Exception:
        info["bitsandbytes"] = None
    return info


def mean_std(values):
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


# --------------------------------------------------------------------------- speed
def make_prompt_ids(tok, n_tokens, batch_size, device):
    """Fixed prompt of n_tokens tokens, replicated batch_size times (no padding needed)."""
    base = "The quick brown fox jumps over the lazy dog. "
    ids = tok(base * (n_tokens // 5 + 10), add_special_tokens=False)["input_ids"][:n_tokens]
    return torch.tensor([ids] * batch_size, device=device)


@torch.inference_mode()
def timed_generate(model, input_ids, new_tokens, pad_id):
    """Wall-clock seconds for one greedy generate() call producing exactly new_tokens tokens."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.generate(
        input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
        max_new_tokens=new_tokens, min_new_tokens=new_tokens,
        do_sample=False, pad_token_id=pad_id,
    )
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def bench_speed(model, tok, batch_sizes, prompt_len, new_tokens, repeats):
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rows = []
    for bs in batch_sizes:
        try:
            input_ids = make_prompt_ids(tok, prompt_len, bs, model.device)
            timed_generate(model, input_ids, 8, pad_id)  # warmup (kernel compilation, caches)
            torch.cuda.reset_peak_memory_stats()

            prefill, total = [], []
            for _ in range(repeats):
                prefill.append(timed_generate(model, input_ids, 1, pad_id))
                total.append(timed_generate(model, input_ids, new_tokens, pad_id))

            # decode latency = (total - prefill) spread over the remaining new_tokens - 1 steps
            decode_ms = [(t - p) / (new_tokens - 1) * 1000 for p, t in zip(prefill, total)]
            throughput = [bs * new_tokens / t for t in total]
            row = {"batch_size": bs, "prompt_len": prompt_len, "new_tokens": new_tokens,
                   "repeats": repeats,
                   "peak_mem_gib": torch.cuda.max_memory_allocated() / GIB}
            for name, values in [("prefill_s", prefill), ("decode_ms_per_token", decode_ms),
                                 ("throughput_tok_s", throughput)]:
                row[f"{name}_mean"], row[f"{name}_std"] = mean_std(values)
            rows.append(row)
            print(f"  bs={bs}: prefill {row['prefill_s_mean']:.3f}s, "
                  f"decode {row['decode_ms_per_token_mean']:.1f} ms/tok, "
                  f"{row['throughput_tok_s_mean']:.1f} tok/s, "
                  f"peak {row['peak_mem_gib']:.2f} GiB", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rows.append({"batch_size": bs, "prompt_len": prompt_len, "error": "OOM"})
            print(f"  bs={bs}: OOM", flush=True)
    return rows


# --------------------------------------------------------------------------- quality
def load_wikitext_test():
    from datasets import load_dataset
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(ds["text"])


@torch.inference_mode()
def eval_perplexity(model, tok, seqlen, max_windows):
    """Perplexity over non-overlapping windows of seqlen tokens (no BOS, no stride overlap).

    Numbers are only comparable between methods run with this exact protocol, not with
    published perplexities that use a different windowing scheme.
    """
    ids = tok(load_wikitext_test(), return_tensors="pt").input_ids[0]
    n_windows = ids.numel() // seqlen
    if max_windows:
        n_windows = min(n_windows, max_windows)
    nll, n_tokens = 0.0, 0
    for i in range(n_windows):
        window = ids[i * seqlen:(i + 1) * seqlen].unsqueeze(0).to(model.device)
        logits = model(window).logits[:, :-1].float()
        labels = window[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="sum")
        nll += loss.item()
        n_tokens += labels.numel()
    return {"wikitext2_ppl": math.exp(nll / n_tokens), "ppl_tokens": n_tokens,
            "ppl_windows": n_windows, "ppl_seqlen": seqlen}


@torch.inference_mode()
def eval_arc_easy(model, tok, n_questions):
    """Zero-shot multiple choice: pick the answer with the highest log-likelihood.

    acc      : argmax of the summed log-probability of the answer tokens
    acc_norm : argmax of the log-probability divided by the answer length in characters
    """
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    n = min(n_questions, len(ds))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    correct = correct_norm = 0

    for ex in ds.select(range(n)):
        ctx_ids = tok(f"Question: {ex['question']}\nAnswer:").input_ids
        seqs, cont_lens, char_lens = [], [], []
        for choice in ex["choices"]["text"]:
            cont = " " + choice
            cont_ids = tok(cont, add_special_tokens=False).input_ids
            seqs.append(ctx_ids + cont_ids)
            cont_lens.append(len(cont_ids))
            char_lens.append(len(cont))

        max_len = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), max_len), pad, dtype=torch.long)
        attn = torch.zeros_like(input_ids)
        for i, s in enumerate(seqs):  # right padding, the causal mask keeps real tokens clean
            input_ids[i, :len(s)] = torch.tensor(s)
            attn[i, :len(s)] = 1

        logits = model(input_ids=input_ids.to(model.device),
                       attention_mask=attn.to(model.device)).logits.float()
        logprobs = torch.log_softmax(logits, dim=-1)

        scores = []
        for i, s in enumerate(seqs):
            start = len(s) - cont_lens[i]
            target = torch.tensor(s[start:], device=logprobs.device).unsqueeze(-1)
            scores.append(logprobs[i, start - 1:len(s) - 1].gather(-1, target).sum().item())

        gold = ex["choices"]["label"].index(ex["answerKey"])
        correct += int(max(range(len(scores)), key=lambda i: scores[i]) == gold)
        normed = [sc / cl for sc, cl in zip(scores, char_lens)]
        correct_norm += int(max(range(len(normed)), key=lambda i: normed[i]) == gold)

    return {"arc_easy_acc": correct / n, "arc_easy_acc_norm": correct_norm / n, "arc_easy_n": n}


# --------------------------------------------------------------------------- commands
def cmd_run(args):
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} already exists, skipping (use --force to overwrite)")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    name = args.name or args.method

    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"[{name}] loading {args.model}", flush=True)
    t0 = time.perf_counter()
    model = load_model(args.model, args.method)
    torch.cuda.synchronize()
    result = {
        "method": name, "base_method": args.method, "model": args.model,
        "env": env_info(),
        "load": {
            "seconds": time.perf_counter() - t0,
            "footprint_gib": model.get_memory_footprint() / GIB,
            "allocated_gib": torch.cuda.memory_allocated() / GIB,
        },
    }
    print(f"[{name}] weights: {result['load']['footprint_gib']:.2f} GiB", flush=True)

    print(f"[{name}] speed benchmark", flush=True)
    result["speed"] = bench_speed(model, tok, args.batch_sizes, args.prompt_len,
                                  args.new_tokens, args.repeats)

    result["quality"] = {}
    if not args.skip_ppl:
        print(f"[{name}] WikiText-2 perplexity", flush=True)
        result["quality"].update(eval_perplexity(model, tok, args.ppl_seqlen, args.ppl_max_windows))
        print(f"  ppl = {result['quality']['wikitext2_ppl']:.3f}", flush=True)
    if args.arc_n > 0:
        print(f"[{name}] ARC-Easy ({args.arc_n} questions)", flush=True)
        result["quality"].update(eval_arc_easy(model, tok, args.arc_n))
        print(f"  acc = {result['quality']['arc_easy_acc']:.3f}, "
              f"acc_norm = {result['quality']['arc_easy_acc_norm']:.3f}", flush=True)

    out.write_text(json.dumps(result, indent=2))
    print(f"[{name}] saved {out}")


def cmd_summarize(args):
    import pandas as pd

    rows = []
    for path in sorted(Path(args.results_dir).glob("quant_*.json")):
        r = json.loads(path.read_text())
        base = {"method": r["method"], "weights_gib": r["load"]["footprint_gib"],
                "load_s": r["load"]["seconds"], **r.get("quality", {})}
        for s in r["speed"]:
            rows.append({**base, **s})
    if not rows:
        print("No quant_*.json files found")
        return
    df = pd.DataFrame(rows)

    # Relative columns against the FP16 reference, when it exists
    ref = df[df.method == "fp16"]
    if not ref.empty:
        if "wikitext2_ppl" in df:
            df["ppl_delta_pct"] = (df.wikitext2_ppl / ref.wikitext2_ppl.iloc[0] - 1) * 100
        df["weights_vs_fp16"] = df.weights_gib / ref.weights_gib.iloc[0]
        if "throughput_tok_s_mean" in df:
            ref_thr = ref.set_index("batch_size").throughput_tok_s_mean
            df["speedup_vs_fp16"] = df.throughput_tok_s_mean / df.batch_size.map(ref_thr)

    df.to_csv(args.out, index=False)
    cols = [c for c in ["method", "batch_size", "weights_gib", "weights_vs_fp16",
                        "prefill_s_mean", "decode_ms_per_token_mean", "throughput_tok_s_mean",
                        "speedup_vs_fp16", "peak_mem_gib", "wikitext2_ppl", "ppl_delta_pct",
                        "arc_easy_acc", "arc_easy_acc_norm"] if c in df.columns]
    print(df[cols].round(3).to_string(index=False))
    print(f"\nSaved {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="benchmark one quantization method")
    run.add_argument("--method", required=True, choices=METHODS)
    run.add_argument("--name", help="label stored in the JSON (default: the method); use it for prequant")
    run.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    run.add_argument("--out", required=True)
    run.add_argument("--force", action="store_true")
    run.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16])
    run.add_argument("--prompt-len", type=int, default=512)
    run.add_argument("--new-tokens", type=int, default=128)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--ppl-seqlen", type=int, default=2048)
    run.add_argument("--ppl-max-windows", type=int, default=0, help="0 = whole test set")
    run.add_argument("--skip-ppl", action="store_true")
    run.add_argument("--arc-n", type=int, default=500, help="0 disables ARC-Easy")
    run.set_defaults(func=cmd_run)

    summ = sub.add_parser("summarize", help="merge quant_*.json into one table")
    summ.add_argument("--results-dir", default="results")
    summ.add_argument("--out", default="results/quant_summary.csv")
    summ.set_defaults(func=cmd_summarize)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
