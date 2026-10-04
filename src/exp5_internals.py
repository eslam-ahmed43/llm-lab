"""Exp 5: transformer internals, implemented from scratch and verified against Hugging Face.

Part 1: a Qwen2-style decoder written in plain PyTorch (RMSNorm, rotary embeddings, grouped-query
attention, SwiGLU MLP, KV cache, greedy decoding). It loads the real Qwen2.5 weights and is
compared with the Hugging Face model on logits, per-layer hidden states and generated tokens.

Part 2: the LoRA layer from train_lora.py is compared with the PEFT library using identical adapter
weights, and merging the adapters into the base weights is checked for equivalence.

Subcommands:
  selftest    tiny random model, no downloads, runs in seconds (use it first to catch bugs)
  qwen-verify real Qwen2.5-1.5B-Instruct: correctness for fp32 / fp16, optional speed benchmark
  lora-peft   own LoRA against PEFT, including adapter merging

Examples:
  python exp5_internals.py selftest --out results/internals_selftest.json
  python exp5_internals.py qwen-verify --dtypes fp32 --n-prompts 1 --new-tokens 8 --out results/internals_quick.json
  python exp5_internals.py qwen-verify --dtypes fp32 fp16 --bench --out results/internals_qwen_verify.json
  python exp5_internals.py lora-peft --rank 8 --out results/internals_lora_peft.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROMPTS = [
    "The capital of France is",
    "In machine learning, overfitting happens when a model",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
]
DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


# --------------------------------------------------------------------------- model
class RMSNorm(nn.Module):
    """x * rsqrt(mean(x^2) + eps) * weight, with the statistics computed in fp32."""

    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_tables(head_dim, theta, positions):
    """cos/sin tables for rotary embeddings, computed in fp32. Shape [T, head_dim]."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32) / head_dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    """x: [B, heads, T, head_dim]; the tables are cast to the dtype of x."""
    cos = cos.to(x.dtype)[None, None]
    sin = sin.to(x.dtype)[None, None]
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    """Grouped-query attention with rotary embeddings and an optional KV cache."""

    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // self.n_heads
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.attn_impl = "manual"

    def forward(self, x, cos, sin, past):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        cache = (k, v)  # the cache keeps the compact KV heads, not the repeated ones
        rep = self.n_heads // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        S = k.shape[2]
        if self.attn_impl == "sdpa":
            if T > 1 and S != T:
                raise ValueError("the sdpa path supports a full prefill or a single new token")
            out = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        else:
            # Scale q and k separately (head_dim ** -0.25 each) so that fp16 scores cannot overflow
            # before the softmax; the product still carries the usual 1 / sqrt(head_dim) factor.
            s = self.head_dim ** -0.25
            scores = torch.matmul(q * s, (k * s).transpose(-1, -2))
            q_pos = torch.arange(S - T, S, device=x.device)
            k_pos = torch.arange(S, device=x.device)
            scores = scores.masked_fill(k_pos[None, :] > q_pos[:, None], float("-inf"))
            probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
            out = torch.matmul(probs, v)
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out), cache


class MLP(nn.Module):
    """SwiGLU: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """Pre-norm residual block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, cfg):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, past):
        h, cache = self.self_attn(self.input_layernorm(x), cos, sin, past)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, cache


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class MiniQwen(nn.Module):
    """Module names mirror the Hugging Face Qwen2 model, so its state dict loads directly."""

    def __init__(self, cfg, rope_theta):
        super().__init__()
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.rope_theta = rope_theta
        self.model = Backbone(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def set_attention(self, impl):
        for layer in self.model.layers:
            layer.self_attn.attn_impl = impl

    def forward(self, input_ids, past=None):
        """Returns (logits, new cache, list with the output of every decoder layer)."""
        T = input_ids.shape[1]
        past_len = 0 if past is None else past[0][0].shape[2]
        positions = torch.arange(past_len, past_len + T, device=input_ids.device)
        cos, sin = rope_tables(self.head_dim, self.rope_theta, positions)
        x = self.model.embed_tokens(input_ids)
        new_past, layer_outputs = [], []
        for i, layer in enumerate(self.model.layers):
            x, cache = layer(x, cos, sin, None if past is None else past[i])
            new_past.append(cache)
            layer_outputs.append(x)
        logits = self.lm_head(self.model.norm(x))
        return logits, new_past, layer_outputs


@torch.inference_mode()
def greedy(model, ids, n_new, eos_ids=(), use_cache=True):
    """Greedy decoding for a batch of one. Returns the list of new token ids."""
    out = []
    cur = ids
    if use_cache:
        logits, past, _ = model(ids)
    for step in range(n_new):
        if not use_cache:
            logits, _, _ = model(cur)
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        token = int(nxt.item())
        out.append(token)
        if token in eos_ids or step == n_new - 1:
            break
        if use_cache:
            logits, past, _ = model(nxt, past)
        else:
            cur = torch.cat([cur, nxt], dim=1)
    return out


def rope_theta_of(cfg):
    theta = getattr(cfg, "rope_theta", None)
    if theta is None:
        theta = (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta")
    if theta is None:
        raise ValueError("could not find rope_theta in the model config")
    return float(theta)


def build_mini(cfg, state_dict, dtype, device, theta=None):
    """Create MiniQwen, load a Hugging Face state dict into it and move it to device / dtype."""
    model = MiniQwen(cfg, theta if theta is not None else rope_theta_of(cfg))
    result = model.load_state_dict(state_dict, strict=False)
    # A tied output head shares its weight with the embedding, so it may be absent from the state dict
    missing = [k for k in result.missing_keys if not (k == "lm_head.weight" and cfg.tie_word_embeddings)]
    if missing:
        raise RuntimeError(f"missing keys when loading the state dict: {missing[:5]}")
    unexpected = [k for k in result.unexpected_keys if "rotary" not in k]
    if unexpected:
        raise RuntimeError(f"unexpected keys when loading the state dict: {unexpected[:5]}")
    return model.to(dtype).to(device).eval()


# --------------------------------------------------------------------------- helpers
def env_info():
    import platform
    import transformers
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    try:
        import peft
        info["peft"] = peft.__version__
    except Exception:
        info["peft"] = None
    return info


def load_hf(model_id, dtype):
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)


def eos_set(generation_config):
    eos = getattr(generation_config, "eos_token_id", None)
    if eos is None:
        return set()
    return set(eos) if isinstance(eos, (list, tuple)) else {int(eos)}


def hf_greedy(hf, ids, n_new, pad_id):
    """Plain greedy decoding in Hugging Face; the repetition penalty in the instruct config is switched off.

    Returns the new tokens and, for every step, the gap between the two largest logits. A tiny gap
    means a near-tie, where a small numerical difference can flip the chosen token.
    """
    out = hf.generate(ids, max_new_tokens=n_new, do_sample=False, repetition_penalty=1.0,
                      pad_token_id=pad_id, output_scores=True, return_dict_in_generate=True)
    tokens = out.sequences[0, ids.shape[1]:].tolist()
    margins = []
    for step_scores in out.scores:
        top2 = step_scores[0].float().topk(2).values
        margins.append((top2[0] - top2[1]).item())
    return tokens, margins


def compare_logits(ref, mine):
    diff = (ref - mine).abs()
    return {"max_abs_diff": diff.max().item(), "mean_abs_diff": diff.mean().item(),
            "ref_abs_max": ref.abs().max().item(),
            "top1_agreement": (ref.argmax(-1) == mine.argmax(-1)).float().mean().item()}


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def decode_speed(prefill_fn, total_fn, n_new, repeats=3):
    prefill_fn()
    total_fn()  # warmup
    pre = statistics.mean(timed(prefill_fn) for _ in range(repeats))
    tot = statistics.mean(timed(total_fn) for _ in range(repeats))
    return {"prefill_s": pre, "decode_ms_per_token": (tot - pre) / (n_new - 1) * 1000, "tokens_per_s": n_new / tot}


def correctness(mini, impl, ref, eos_ids, n_new, device):
    """Compare MiniQwen against saved Hugging Face reference outputs for every prompt."""
    mini.set_attention(impl)
    prompts = []
    for i, item in enumerate(ref):
        ids = item["ids"].to(device)
        with torch.inference_mode():
            logits, _, layer_outputs = mini(ids)
            entry = compare_logits(item["logits"], logits.float().cpu())
            if i == 0:  # per-layer drift, only for the first prompt, to locate where a mismatch starts
                entry["layer_max_abs_diff"] = [(a - b.float().cpu()).abs().max().item()
                                               for a, b in zip(item["layers"], layer_outputs)]
                entry["first_nan_layer"] = next((j for j, v in enumerate(entry["layer_max_abs_diff"]) if v != v), None)
            full = logits[:, -1].float()
            _, past, _ = mini(ids[:, :-1])
            step = mini(ids[:, -1:], past)[0][:, -1].float()
            entry["cache_vs_full_max_abs_diff"] = (full - step).abs().max().item()
        mine_tokens = greedy(mini, ids, n_new, eos_ids)
        entry["greedy_identical"] = mine_tokens == item["gen"]
        d = first_divergence(mine_tokens, item["gen"])
        entry["greedy_first_divergence"] = d
        entry["ref_margin_at_divergence"] = item["margins"][d] if d is not None and d < len(item["margins"]) else None
        entry["prompt_tokens"] = ids.shape[1]
        prompts.append(entry)
    return {"prompts": prompts}


# --------------------------------------------------------------------------- commands
def cmd_selftest(args):
    """Tiny random Qwen2 model on the CPU: checks the implementation logic without any download."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=200, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      rms_norm_eps=1e-6, tie_word_embeddings=True)
    hf = Qwen2ForCausalLM(cfg).eval()
    for name, p in hf.named_parameters():  # random values everywhere, including biases and norm weights
        p.data.uniform_(0.5, 1.5) if "norm" in name else p.data.normal_(0, 0.1)
    try:
        theta = rope_theta_of(cfg)
    except ValueError:
        theta = 10000.0
    mini = build_mini(cfg, hf.state_dict(), torch.float32, "cpu", theta=theta)
    ids = torch.randint(0, 200, (1, 12))
    with torch.inference_mode():
        ref = hf(ids).logits
    report = {"env": env_info(), "rope_theta": theta, "impls": {}}
    for impl in ["manual", "sdpa"]:
        mini.set_attention(impl)
        with torch.inference_mode():
            logits, _, _ = mini(ids)
            full = logits[:, -1]
            _, past, _ = mini(ids[:, :-1])
            step = mini(ids[:, -1:], past)[0][:, -1]
        hf_tokens = hf.generate(ids, max_new_tokens=10, do_sample=False, repetition_penalty=1.0, pad_token_id=0)[0, 12:].tolist()
        mine_tokens = greedy(mini, ids, 10)
        entry = compare_logits(ref, logits)
        entry["cache_vs_full_max_abs_diff"] = (full - step).abs().max().item()
        entry["greedy_identical"] = mine_tokens == hf_tokens
        report["impls"][impl] = entry
        print(impl, {k: (round(v, 8) if isinstance(v, float) else v) for k, v in entry.items()}, flush=True)
    ok = all(e["max_abs_diff"] < 1e-4 and e["greedy_identical"] and e["cache_vs_full_max_abs_diff"] < 1e-4
             for e in report["impls"].values())
    report["passed"] = ok
    print("SELFTEST", "PASSED" if ok else "FAILED")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    if not ok:
        sys.exit(1)


def cmd_qwen_verify(args):
    from transformers import AutoConfig, AutoTokenizer
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    prompt_ids = [tok(p, return_tensors="pt").input_ids for p in PROMPTS[:args.n_prompts]]
    base = "The quick brown fox jumps over the lazy dog. "
    bench_ids = torch.tensor([tok(base * 110, add_special_tokens=False).input_ids[:512]], device=device)
    report = {"env": env_info(), "model": args.model, "new_tokens": args.new_tokens}

    for dtype_name in args.dtypes:
        dtype = DTYPES[dtype_name]
        print(f"[{dtype_name}] Hugging Face reference", flush=True)
        hf = load_hf(args.model, dtype).to(device).eval()
        eos_ids = eos_set(hf.generation_config)
        ref, layers_out, hooks = [], [], []
        for layer in hf.model.layers:
            hooks.append(layer.register_forward_hook(
                lambda m, i, o: layers_out.append((o[0] if isinstance(o, tuple) else o).detach().float().cpu())))
        for idx, ids in enumerate(prompt_ids):
            ids = ids.to(device)
            with torch.inference_mode():
                logits = hf(ids).logits.float().cpu()
            if idx == 0:
                for h in hooks:
                    h.remove()
            gen_tokens, margins = hf_greedy(hf, ids, args.new_tokens, pad_id)
            ref.append({"ids": ids.cpu(), "logits": logits, "layers": list(layers_out) if idx == 0 else None,
                        "gen": gen_tokens, "margins": margins})
        hf_bench = None
        if args.bench and dtype_name == "fp16":
            hf_bench = decode_speed(
                lambda: hf.generate(bench_ids, max_new_tokens=1, min_new_tokens=1, do_sample=False, repetition_penalty=1.0, pad_token_id=pad_id),
                lambda: hf.generate(bench_ids, max_new_tokens=args.bench_tokens, min_new_tokens=args.bench_tokens, do_sample=False, repetition_penalty=1.0, pad_token_id=pad_id),
                args.bench_tokens)
        state_dict = {k: v.detach().cpu() for k, v in hf.state_dict().items()}
        hf = None  # free the GPU copy before building our own model
        torch.cuda.empty_cache()

        print(f"[{dtype_name}] own implementation", flush=True)
        mini = build_mini(cfg, state_dict, dtype, device)
        entry = {}
        for impl in args.impls:
            entry[impl] = correctness(mini, impl, ref, eos_ids, args.new_tokens, device)
            first = entry[impl]["prompts"][0]
            plist = entry[impl]["prompts"]
            print(f"  {impl}: max|diff| {first['max_abs_diff']:.2e} (logits up to {first['ref_abs_max']:.1f}), "
                  f"top-1 agreement {first['top1_agreement']:.3f}, greedy identical "
                  f"{[p['greedy_identical'] for p in plist]}, first divergence {[p['greedy_first_divergence'] for p in plist]}, "
                  f"reference top-2 gap there {[None if p['ref_margin_at_divergence'] is None else round(p['ref_margin_at_divergence'], 3) for p in plist]}, "
                  f"first NaN layer {first.get('first_nan_layer')}", flush=True)
        if args.bench and dtype_name == "fp16":
            bench = {"hf_generate": hf_bench}
            for impl in args.impls:
                mini.set_attention(impl)
                bench[f"own_{impl}"] = decode_speed(
                    lambda: greedy(mini, bench_ids, 1),
                    lambda: greedy(mini, bench_ids, args.bench_tokens),
                    args.bench_tokens)
            entry["bench"] = bench
            print("  bench:", {k: {m: round(x, 3) for m, x in v.items()} for k, v in bench.items()}, flush=True)
        report[dtype_name] = entry
        mini = state_dict = ref = None
        torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"saved {args.out}")


def merge_lora(model, lora_cls):
    """Fold every LoRA layer into its base weight, W <- W + (alpha / r) * B A, and drop the adapter."""
    count = 0
    for _, parent in list(model.named_modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, lora_cls):
                delta = child.scale * (child.lora_B @ child.lora_A)
                child.base.weight.data += delta.to(child.base.weight.dtype)
                setattr(parent, name, child.base)
                count += 1
    return count


def cmd_lora_peft(args):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from peft import LoraConfig, get_peft_model
    from train_lora import TARGETS, LoRALinear, inject_lora
    from transformers import AutoTokenizer
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(PROMPTS[1], return_tensors="pt").input_ids.to(device)
    rank, alpha = args.rank, 2 * args.rank
    report = {"env": env_info(), "model": args.model, "rank": rank, "alpha": alpha, "dtype": "fp32"}

    print("own LoRA", flush=True)
    torch.manual_seed(0)
    model = load_hf(args.model, torch.float32).to(device).eval()
    with torch.inference_mode():
        logits_base = model(ids).logits.float().cpu()
    inject_lora(model, TARGETS["all"], rank, alpha, 0.0)
    for m in model.modules():  # B starts at zero, give it small random values so the adapters matter
        if isinstance(m, LoRALinear):
            nn.init.normal_(m.lora_B, std=0.02)
    weights = {n: (m.lora_A.detach().cpu().clone(), m.lora_B.detach().cpu().clone())
               for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    own_params = sum(a.numel() + b.numel() for a, b in weights.values())
    with torch.inference_mode():
        logits_own = model(ids).logits.float().cpu()
    merged = merge_lora(model, LoRALinear)
    with torch.inference_mode():
        logits_own_merged = model(ids).logits.float().cpu()
    del model
    torch.cuda.empty_cache()

    print("PEFT", flush=True)
    base = load_hf(args.model, torch.float32).to(device).eval()
    peft_cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=TARGETS["all"],
                          bias="none", task_type="CAUSAL_LM")
    try:
        pm = get_peft_model(base, peft_cfg)
    except ImportError as e:
        if "torchao" in str(e):
            raise RuntimeError("PEFT rejects the preinstalled torchao version; run `pip uninstall -y torchao` and retry") from e
        raise
    copied = 0
    for name, m in pm.named_modules():
        if hasattr(m, "lora_A") and hasattr(m, "lora_B") and "default" in m.lora_A:
            a, b = weights[name.replace("base_model.model.", "", 1)]
            m.lora_A["default"].weight.data.copy_(a)
            m.lora_B["default"].weight.data.copy_(b)
            copied += 1
    pm.eval()
    peft_trainable = pm.get_nb_trainable_parameters()[0]
    with torch.inference_mode():
        logits_peft = pm(ids).logits.float().cpu()
    peft_merged_model = pm.merge_and_unload()
    with torch.inference_mode():
        logits_peft_merged = peft_merged_model(ids).logits.float().cpu()

    report.update({
        "adapted_modules_own": len(weights), "adapted_modules_peft": copied,
        "trainable_params_own": own_params, "trainable_params_peft": peft_trainable,
        "merged_modules_own": merged,
        "base_vs_adapted": compare_logits(logits_base, logits_own),
        "own_vs_peft": compare_logits(logits_own, logits_peft),
        "own_unmerged_vs_merged": compare_logits(logits_own, logits_own_merged),
        "peft_unmerged_vs_merged": compare_logits(logits_peft, logits_peft_merged),
        "own_merged_vs_peft_merged": compare_logits(logits_own_merged, logits_peft_merged),
    })
    for key in ["base_vs_adapted", "own_vs_peft", "own_unmerged_vs_merged", "peft_unmerged_vs_merged", "own_merged_vs_peft_merged"]:
        print(f"  {key}: max|diff| {report[key]['max_abs_diff']:.2e}, top-1 agreement {report[key]['top1_agreement']:.3f}")
    print(f"  trainable params: own {own_params:,} vs PEFT {peft_trainable:,}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"saved {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("selftest", help="tiny random model, no downloads")
    st.add_argument("--out", default="results/internals_selftest.json")
    st.set_defaults(func=cmd_selftest)

    qv = sub.add_parser("qwen-verify", help="compare MiniQwen with Hugging Face on the real model")
    qv.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    qv.add_argument("--dtypes", nargs="+", default=["fp32", "fp16"], choices=list(DTYPES))
    qv.add_argument("--impls", nargs="+", default=["manual", "sdpa"], choices=["manual", "sdpa"])
    qv.add_argument("--n-prompts", type=int, default=3)
    qv.add_argument("--new-tokens", type=int, default=48)
    qv.add_argument("--bench", action="store_true", help="decode speed benchmark (fp16 only)")
    qv.add_argument("--bench-tokens", type=int, default=128)
    qv.add_argument("--out", default="results/internals_qwen_verify.json")
    qv.set_defaults(func=cmd_qwen_verify)

    lp = sub.add_parser("lora-peft", help="own LoRA against PEFT, including adapter merging")
    lp.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    lp.add_argument("--rank", type=int, default=8)
    lp.add_argument("--out", default="results/internals_lora_peft.json")
    lp.set_defaults(func=cmd_lora_peft)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
