"""LoRA and QLoRA fine-tuning study (Exp 4), with LoRA implemented from scratch in PyTorch.

Question: how do the LoRA rank, the set of adapted modules and 4-bit base weights (QLoRA)
trade off task quality, trainable parameters, GPU memory and training time?

Task: 6-way emotion classification (dair-ai/emotion). The model sees an instruction and a text and
must continue with the emotion name. Quality is measured on the test split by comparing the
next-token logits of the six label words (no generation), reporting accuracy and macro-F1.

Design choices, kept fixed across runs so that the differences come from the variables under study:
  * the same prompt template, data order per seed, optimizer, learning rate schedule and step budget
  * alpha = 2 * rank, so the LoRA scaling factor alpha / rank is constant (2.0)
  * LoRA matrices live in fp32 and are trained with fp16 autocast and loss scaling (the T4 has no bf16)
  * the loss is computed only on the answer tokens

Each run writes one JSON file. Merge them with the `summarize` subcommand.

Examples:
  python train_lora.py run --mode zeroshot --out results/lora_zeroshot.json
  python train_lora.py run --mode lora --rank 8 --targets all --seed 0 --out results/lora_lora_r8_all_s0.json
  python train_lora.py run --mode qlora --rank 8 --targets all --seed 0 --out results/lora_qlora_r8_all_s0.json
  python train_lora.py summarize --results-dir results --out results/lora_summary.csv
"""
import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

GIB = 1024 ** 3
INSTRUCTION = "Classify the emotion of the text as one of: sadness, joy, love, anger, fear, surprise."
TARGETS = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


# --------------------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B(A(dropout(x))), with the base layer W frozen.

    The base can be a regular fp16 linear layer or a bitsandbytes 4-bit / 8-bit layer, because the
    adapter only calls base(x) and adds its own output.
    """

    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # same init as the reference implementation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.base(x)
        update = F.linear(F.linear(self.dropout(x).to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return out + (update * self.scale).to(out.dtype)


def inject_lora(model, target_names, rank, alpha, dropout):
    """Freeze the model and wrap every target linear layer with a LoRALinear. Returns the count."""
    for p in model.parameters():
        p.requires_grad_(False)
    count = 0
    for _, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in target_names and isinstance(child, nn.Linear):
                setattr(parent, child_name, LoRALinear(child, rank, alpha, dropout))
                count += 1
    return count


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- data
def load_emotion():
    from datasets import load_dataset
    try:
        ds = load_dataset("dair-ai/emotion", "split")
    except Exception:
        ds = load_dataset("emotion")
    names = ds["train"].features["label"].names
    return ds["train"], ds["test"], names


def build_examples(tok, texts, labels, label_names, max_len):
    """Tokenize prompt and answer separately; the answer is ' <label>'."""
    out = []
    for text, label in zip(texts, labels):
        prompt = f"{INSTRUCTION}\nText: {text}\nEmotion:"
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tok(" " + label_names[label], add_special_tokens=False)["input_ids"]
        prompt_ids = prompt_ids[-(max_len - len(target_ids)):]  # rare: keep the end of an overlong prompt
        out.append({"prompt_ids": prompt_ids, "target_ids": target_ids, "gold": label})
    return out


def label_first_token_ids(tok, label_names):
    ids = [tok(" " + n, add_special_tokens=False)["input_ids"][0] for n in label_names]
    if len(set(ids)) != len(ids):
        raise ValueError("The first tokens of the label words are not distinct, scoring by first token is invalid")
    return ids


def collate_train(items, pad_id, device):
    seqs = [x["prompt_ids"] + x["target_ids"] for x in items]
    max_len = max(len(s) for s in seqs)
    ids = torch.full((len(items), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(items), max_len), dtype=torch.long)
    labels = torch.full((len(items), max_len), -100, dtype=torch.long)
    for i, (x, s) in enumerate(zip(items, seqs)):
        ids[i, :len(s)] = torch.tensor(s)
        attn[i, :len(s)] = 1
        labels[i, len(x["prompt_ids"]):len(s)] = torch.tensor(x["target_ids"])  # loss on answer tokens only
    return ids.to(device), attn.to(device), labels.to(device)


def collate_eval(items, pad_id, device):
    max_len = max(len(x["prompt_ids"]) for x in items)
    ids = torch.full((len(items), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(items), max_len), dtype=torch.long)
    last = torch.zeros(len(items), dtype=torch.long)
    for i, x in enumerate(items):
        n = len(x["prompt_ids"])
        ids[i, :n] = torch.tensor(x["prompt_ids"])
        attn[i, :n] = 1
        last[i] = n - 1
    return ids.to(device), attn.to(device), last.to(device)


# --------------------------------------------------------------------------- model
def load_model(model_id, mode):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    kwargs = {"device_map": {"": 0}}
    if mode == "qlora":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, **kwargs)
    model.config.use_cache = False
    return model


def env_info():
    """Library versions and GPU name, stored in every result file for reproducibility."""
    import platform
    import transformers
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0)}
    try:
        import bitsandbytes
        info["bitsandbytes"] = bitsandbytes.__version__
    except Exception:
        info["bitsandbytes"] = None
    return info


def hidden_states(model, ids, attn):
    return model.model(input_ids=ids, attention_mask=attn, use_cache=False).last_hidden_state


def answer_loss(model, ids, attn, labels):
    """Cross-entropy on the answer tokens only; the LM head is applied just to those positions."""
    hidden = hidden_states(model, ids, attn)
    target = labels[:, 1:]
    selected = target != -100
    logits = model.lm_head(hidden[:, :-1][selected]).float()
    return F.cross_entropy(logits, target[selected])


@torch.inference_mode()
def evaluate(model, items, label_ids, pad_id, batch_size, device):
    model.eval()
    label_ids = torch.tensor(label_ids, device=device)
    preds, golds = [], []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        ids, attn, last = collate_eval(chunk, pad_id, device)
        hidden = hidden_states(model, ids, attn)
        logits = model.lm_head(hidden[torch.arange(len(chunk), device=device), last]).float()
        preds += logits[:, label_ids].argmax(dim=-1).tolist()
        golds += [x["gold"] for x in chunk]
    return metrics(preds, golds, n_classes=len(label_ids))


def metrics(preds, golds, n_classes):
    acc = sum(int(p == g) for p, g in zip(preds, golds)) / len(golds)
    f1s = []
    for c in range(n_classes):
        tp = sum(1 for p, g in zip(preds, golds) if p == c and g == c)
        fp = sum(1 for p, g in zip(preds, golds) if p == c and g != c)
        fn = sum(1 for p, g in zip(preds, golds) if p != c and g == c)
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return {"accuracy": acc, "macro_f1": sum(f1s) / n_classes, "n": len(golds)}


def train(model, items, args, pad_id, device, use_amp):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    warmup = max(1, int(0.05 * args.steps))

    def lr_factor(step):  # linear warmup then linear decay to zero
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (args.steps - step) / max(1, args.steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = random.Random(args.seed)
    order, pos = [], 0
    losses, tokens = [], 0
    model.train()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(args.steps):
        batch = []
        while len(batch) < args.batch_size:
            if pos >= len(order):  # reshuffle at every epoch boundary
                order = list(range(len(items)))
                rng.shuffle(order)
                pos = 0
            batch.append(items[order[pos]])
            pos += 1
        ids, attn, labels = collate_train(batch, pad_id, device)
        tokens += int(attn.sum())
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss = answer_loss(model, ids, attn, labels)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        losses.append(loss.item())
        if (step + 1) % 100 == 0:
            print(f"  step {step + 1}/{args.steps} loss {statistics.mean(losses[-100:]):.4f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    return {"seconds": seconds, "tokens_per_s": tokens / seconds,
            "final_loss": statistics.mean(losses[-50:]),
            "loss_curve": [round(statistics.mean(losses[i:i + 25]), 4) for i in range(0, len(losses), 25)]}


# --------------------------------------------------------------------------- commands
def cmd_run(args):
    from transformers import AutoTokenizer
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} already exists, skipping (use --force to overwrite)")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    use_amp = True

    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    train_ds, test_ds, label_names = load_emotion()
    # list(...) makes the columns plain Python lists regardless of the datasets library version
    train_texts, train_labels = list(train_ds["text"]), list(train_ds["label"])
    test_texts, test_labels = list(test_ds["text"]), list(test_ds["label"])
    n_train = args.train_limit or len(train_texts)
    train_items = build_examples(tok, train_texts[:n_train], train_labels[:n_train], label_names, args.max_len)
    test_items = build_examples(tok, test_texts[:args.eval_n], test_labels[:args.eval_n], label_names, args.max_len)
    label_ids = label_first_token_ids(tok, label_names)

    tag = f"{args.mode}" + ("" if args.mode == "zeroshot" else f"_r{args.rank}_{args.targets}_s{args.seed}")
    print(f"[{tag}] loading {args.model}", flush=True)
    model = load_model(args.model, args.mode)
    result = {"mode": args.mode, "model": args.model, "seed": args.seed, "lr": args.lr,
              "steps": args.steps, "batch_size": args.batch_size, "env": env_info(),
              "load_footprint_gib": model.get_memory_footprint() / GIB}

    if args.mode != "zeroshot":
        alpha = args.alpha if args.alpha else 2 * args.rank
        # Reference outputs on one batch before adding adapters: B starts at zero, so they must not change
        ref_ids, ref_attn, _ = collate_eval(test_items[:8], pad_id, device)
        model.eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            ref = hidden_states(model, ref_ids, ref_attn).float()
        n_modules = inject_lora(model, TARGETS[args.targets], args.rank, alpha, args.dropout)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            diff = (hidden_states(model, ref_ids, ref_attn).float() - ref).abs().max().item()
        assert diff < 1e-3, f"LoRA injection changed the model output by {diff}"
        result.update({"rank": args.rank, "alpha": alpha, "targets": args.targets, "dropout": args.dropout,
                       "lora_modules": n_modules, "trainable_params": count_trainable(model),
                       "init_max_abs_diff": diff})
        print(f"[{tag}] {n_modules} adapted modules, {result['trainable_params']:,} trainable params", flush=True)
        torch.cuda.reset_peak_memory_stats()
        result["train"] = train(model, train_items, args, pad_id, device, use_amp)
        result["train"]["peak_mem_gib"] = torch.cuda.max_memory_allocated() / GIB
    else:
        result.update({"rank": 0, "trainable_params": 0})

    result["eval"] = evaluate(model, test_items, label_ids, pad_id, args.eval_batch_size, device)
    print(f"[{tag}] accuracy {result['eval']['accuracy']:.4f}, macro-F1 {result['eval']['macro_f1']:.4f}", flush=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[{tag}] saved {out}")


def cmd_summarize(args):
    import pandas as pd

    rows = []
    for path in sorted(Path(args.results_dir).glob("lora_*.json")):
        r = json.loads(path.read_text())
        if "eval" not in r:
            continue
        rows.append({
            "mode": r["mode"], "rank": r.get("rank", 0), "targets": r.get("targets", "-"), "seed": r["seed"],
            "trainable_params": r.get("trainable_params", 0),
            "accuracy": r["eval"]["accuracy"], "macro_f1": r["eval"]["macro_f1"],
            "peak_mem_gib": r.get("train", {}).get("peak_mem_gib"),
            "train_seconds": r.get("train", {}).get("seconds"),
            "tokens_per_s": r.get("train", {}).get("tokens_per_s"),
            "final_loss": r.get("train", {}).get("final_loss"),
        })
    if not rows:
        print("No lora_*.json files found")
        return
    df = pd.DataFrame(rows)
    df.to_csv(Path(args.out).with_name("lora_all_runs.csv"), index=False)
    agg = df.groupby(["mode", "targets", "rank"]).agg(
        runs=("seed", "count"), trainable_params=("trainable_params", "first"),
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
        f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
        peak_mem_gib=("peak_mem_gib", "mean"), train_seconds=("train_seconds", "mean"),
        tokens_per_s=("tokens_per_s", "mean"), final_loss=("final_loss", "mean"),
    ).reset_index().fillna(0)
    agg.to_csv(args.out, index=False)
    print(agg.round(4).to_string(index=False))
    print(f"\nSaved {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="train and evaluate one configuration")
    run.add_argument("--mode", required=True, choices=["zeroshot", "lora", "qlora"])
    run.add_argument("--rank", type=int, default=8)
    run.add_argument("--alpha", type=float, default=0, help="0 means 2 * rank")
    run.add_argument("--targets", choices=list(TARGETS), default="all")
    run.add_argument("--dropout", type=float, default=0.05)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--lr", type=float, default=2e-4)
    run.add_argument("--steps", type=int, default=600)
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--max-len", type=int, default=128)
    run.add_argument("--train-limit", type=int, default=0, help="0 means the whole train split")
    run.add_argument("--eval-n", type=int, default=2000)
    run.add_argument("--eval-batch-size", type=int, default=64)
    run.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    run.add_argument("--out", required=True)
    run.add_argument("--force", action="store_true")
    run.set_defaults(func=cmd_run)

    summ = sub.add_parser("summarize", help="merge lora_*.json into one table")
    summ.add_argument("--results-dir", default="results")
    summ.add_argument("--out", default="results/lora_summary.csv")
    summ.set_defaults(func=cmd_summarize)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
