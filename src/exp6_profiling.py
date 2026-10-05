"""Exp 6: where does single-stream decode time go on a T4, and do CUDA graphs remove the overhead?

Earlier reports (Exp 1 and Exp 5) suggested that single-stream decoding of a 1.5B model is limited by
per-step overhead (kernel launches, Python), not by GPU work, and that vLLM's lead at concurrency 1
comes from removing that overhead. This script tests that with the from-scratch decoder of Exp 5.

Four ways to run the same decode step, all producing the same tokens (checked first):
  dynamic_eager_sync    growing KV cache (torch.cat), eager, one host sync per token (the Exp 5 loop)
  dynamic_eager_nosync  the same without the per-token host sync
  static_eager_nosync   preallocated KV cache and a position tensor, eager
  static_cuda_graph     the static step captured once in a CUDA graph and replayed

For every variant we report the time per decode step and how long the CPU needs just to enqueue the
work (if enqueueing alone takes as long as the whole step, the GPU is waiting for the CPU). An optional
torch.profiler pass counts GPU kernels per step and the time the GPU is actually busy.

Subcommands:
  selftest  tiny random model on the CPU, checks the static-cache step against the normal one
  run       the real model: equivalence check, benchmark and optional profiler pass

Examples:
  python exp6_profiling.py selftest
  python exp6_profiling.py run --check --bench --profile --out results/profiling_exp6.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp5_internals import PROMPTS, apply_rope, build_mini, env_info, greedy, load_hf, rope_tables, rope_theta_of  # noqa: E402


# --------------------------------------------------------------------------- decode variants
def make_static_cache(mini, batch, max_len, dtype, device):
    """One preallocated key and value buffer per layer, shape [batch, kv_heads, max_len, head_dim]."""
    n_kv = mini.model.layers[0].self_attn.n_kv
    return [(torch.zeros(batch, n_kv, max_len, mini.head_dim, dtype=dtype, device=device),
             torch.zeros(batch, n_kv, max_len, mini.head_dim, dtype=dtype, device=device))
            for _ in mini.model.layers]


def static_decode_step(mini, token, pos, caches):
    """One decode step against a preallocated KV cache.

    token is a [B, 1] long tensor and pos a [1] long tensor, both on the device, so the step contains no
    host-side values and can be captured in a CUDA graph. The new key and value are written at index
    pos, and a mask hides all positions after pos.
    """
    cos, sin = rope_tables(mini.head_dim, mini.rope_theta, pos)
    x = mini.model.embed_tokens(token)
    batch = x.shape[0]
    max_len = caches[0][0].shape[2]
    keep = (torch.arange(max_len, device=token.device) <= pos)[None, None, None, :]
    for layer, (k_cache, v_cache) in zip(mini.model.layers, caches):
        attn = layer.self_attn
        h = layer.input_layernorm(x)
        q = attn.q_proj(h).view(batch, 1, attn.n_heads, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(h).view(batch, 1, attn.n_kv, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(h).view(batch, 1, attn.n_kv, attn.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k_cache.index_copy_(2, pos, k)
        v_cache.index_copy_(2, pos, v)
        rep = attn.n_heads // attn.n_kv
        keys = k_cache.repeat_interleave(rep, dim=1)
        values = v_cache.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, keys, values, attn_mask=keep)
        out = out.transpose(1, 2).reshape(batch, 1, attn.n_heads * attn.head_dim)
        x = x + attn.o_proj(out)
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    return mini.lm_head(mini.model.norm(x))


def prefill(mini, ids):
    """Run the prompt once. Returns the first generated token [B, 1] and the dynamic KV cache."""
    logits, past, _ = mini(ids)
    return logits[:, -1].argmax(dim=-1, keepdim=True), past


def fill_static_cache(caches, past):
    prompt_len = past[0][0].shape[2]
    for (k_cache, v_cache), (k, v) in zip(caches, past):
        k_cache[:, :, :prompt_len].copy_(k)
        v_cache[:, :, :prompt_len].copy_(v)


def decode_dynamic(mini, first, past, steps, sync_each):
    """Decode with the growing cache. With sync_each the token is read back on the host at every step."""
    toks, cur = [first], first
    for _ in range(steps):
        logits, past, _ = mini(cur, past)
        cur = logits[:, -1].argmax(dim=-1, keepdim=True)
        if sync_each:
            cur.item()
        toks.append(cur)
    return torch.cat(toks, dim=1)


def decode_static_eager(mini, first, caches, start_pos, steps):
    pos = torch.tensor([start_pos], dtype=torch.long, device=first.device)
    toks, cur = [first], first
    for _ in range(steps):
        logits = static_decode_step(mini, cur, pos, caches)
        cur = logits[:, -1].argmax(dim=-1, keepdim=True)
        toks.append(cur)
        pos += 1
    return torch.cat(toks, dim=1)


class GraphDecoder:
    """The static decode step captured once as a CUDA graph. Input token and position live in fixed buffers."""

    def __init__(self, mini, caches, batch, device):
        self.mini, self.caches = mini, caches
        self.tok = torch.zeros(batch, 1, dtype=torch.long, device=device)
        self.pos = torch.zeros(1, dtype=torch.long, device=device)
        self.graph = None

    def _step(self):
        logits = static_decode_step(self.mini, self.tok, self.pos, self.caches)
        self.tok.copy_(logits[:, -1].argmax(dim=-1, keepdim=True))
        self.pos.add_(1)

    def capture(self, first, start_pos):
        self.tok.copy_(first)
        self.pos.fill_(start_pos)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):  # warm-up outside the graph
                self._step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()

    def run(self, first, start_pos, steps):
        self.tok.copy_(first)
        self.pos.fill_(start_pos)
        out = torch.empty(first.shape[0], steps + 1, dtype=torch.long, device=first.device)
        out[:, :1] = first
        for i in range(steps):
            self.graph.replay()
            out[:, i + 1:i + 2] = self.tok
        return out


# --------------------------------------------------------------------------- measurement
def time_decode(setup, run, steps, repeats):
    """Time `run` (which enqueues `steps` decode steps). Setup is excluded from the timing.

    Returns the time per step including the final synchronization, and the CPU-side time per step
    needed just to enqueue the work. If both are equal, the GPU is waiting for the CPU.
    """
    totals, issues = [], []
    for r in range(repeats + 1):  # the first repetition is a warm-up
        setup()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run(steps)
        issue = time.perf_counter() - t0
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        if r > 0:
            totals.append(total)
            issues.append(issue)
    ms = [t / steps * 1000 for t in totals]
    return {"ms_per_step": statistics.mean(ms),
            "ms_per_step_std": statistics.stdev(ms) if len(ms) > 1 else 0.0,
            "enqueue_ms_per_step": statistics.mean(issues) / steps * 1000,
            "tokens_per_s": 1000 / statistics.mean(ms)}


def profile_decode(setup, run, steps):
    """Count device activities per step and the time the GPU is busy, using torch.profiler."""
    try:
        from torch.profiler import ProfilerActivity, profile
        setup()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run(steps)
            torch.cuda.synchronize()
        device_events = [e for e in prof.events() if "CUDA" in str(e.device_type)]
        by_name = {}
        for e in device_events:
            by_name[e.name] = by_name.get(e.name, 0.0) + e.time_range.elapsed_us()
        top = sorted(by_name.items(), key=lambda kv: -kv[1])[:8]
        busy_us = sum(by_name.values())
        return {"device_events_per_step": len(device_events) / steps,
                "device_busy_ms_per_step": busy_us / 1000 / steps,
                "top_device_ops_ms_per_step": [[name[:90], round(us / 1000 / steps, 4)] for name, us in top]}
    except Exception as exc:  # the profiler API differs between versions, never fail the whole run
        return {"error": repr(exc)}


def measure_bandwidth(device, size_mib=1024, repeats=20):
    """Achievable device memory bandwidth in GB/s, from a large device-to-device copy (reads plus writes)."""
    src = torch.ones(size_mib * 1024 * 1024 // 2, dtype=torch.float16, device=device)
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        dst.copy_(src)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    moved = 2 * src.numel() * src.element_size() * repeats  # every copy reads and writes the buffer once
    return moved / seconds / 1e9


def equivalence(mini, ids, n_new, dtype, device, use_graph):
    """All variants must generate the same tokens as the plain greedy loop."""
    steps = n_new - 1
    prompt_len = ids.shape[1]
    reference = greedy(mini, ids, n_new)
    first, past = prefill(mini, ids)
    result = {"dynamic_nosync": decode_dynamic(mini, first, past, steps, False)[0].tolist() == reference}
    caches = make_static_cache(mini, ids.shape[0], prompt_len + n_new, dtype, device)
    fill_static_cache(caches, past)
    result["static_eager"] = decode_static_eager(mini, first, caches, prompt_len, steps)[0].tolist() == reference
    if use_graph:
        decoder = GraphDecoder(mini, caches, ids.shape[0], device)
        decoder.capture(first, prompt_len)
        result["static_graph"] = decoder.run(first, prompt_len, steps)[0].tolist() == reference
    return result


# --------------------------------------------------------------------------- commands
def cmd_selftest(args):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=200, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      rms_norm_eps=1e-6, tie_word_embeddings=True)
    hf = Qwen2ForCausalLM(cfg).eval()
    for name, p in hf.named_parameters():
        p.data.uniform_(0.5, 1.5) if "norm" in name else p.data.normal_(0, 0.1)
    try:
        theta = rope_theta_of(cfg)
    except ValueError:
        theta = 10000.0
    mini = build_mini(cfg, hf.state_dict(), torch.float32, "cpu", theta=theta)
    mini.set_attention("sdpa")
    ids = torch.randint(0, 200, (1, 12))
    with torch.inference_mode():
        result = equivalence(mini, ids, 10, torch.float32, "cpu", use_graph=False)
        first, past = prefill(mini, ids)
        dynamic_logits = mini(first, past)[0][:, -1]
        caches = make_static_cache(mini, 1, 12 + 10, torch.float32, "cpu")
        fill_static_cache(caches, past)
        static_logits = static_decode_step(mini, first, torch.tensor([12]), caches)[:, -1]
        diff = (dynamic_logits - static_logits).abs().max().item()
    result["first_step_logit_diff"] = diff
    ok = all(v for k, v in result.items() if k != "first_step_logit_diff") and diff < 1e-4
    print(result)
    print("SELFTEST", "PASSED" if ok else "FAILED")
    if not ok:
        sys.exit(1)


def cmd_run(args):
    from transformers import AutoConfig, AutoTokenizer
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    report = {"env": env_info(), "model": args.model, "prompt_len": args.prompt_len,
              "new_tokens": args.new_tokens, "repeats": args.repeats}

    def load(dtype):
        hf = load_hf(args.model, dtype)
        mini = build_mini(cfg, hf.state_dict(), dtype, device)
        mini.set_attention("sdpa")
        return mini

    with torch.inference_mode():
        if args.check:
            print("[check] fp32 equivalence of all variants", flush=True)
            mini = load(torch.float32)
            short = tok(PROMPTS[1], return_tensors="pt").input_ids.to(device)
            report["check_fp32"] = equivalence(mini, short, 12, torch.float32, device, use_graph=True)
            print("  ", report["check_fp32"], flush=True)
            mini = None
            torch.cuda.empty_cache()
            if not all(report["check_fp32"].values()):
                print("EQUIVALENCE FAILED, benchmark would be meaningless")
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps(report, indent=2))
                sys.exit(1)

        if args.bench or args.profile:
            print("[fp16] building the model", flush=True)
            dtype = torch.float16
            mini = load(dtype)
            base = "The quick brown fox jumps over the lazy dog. "
            ids = torch.tensor([tok(base * 110, add_special_tokens=False).input_ids[:args.prompt_len]], device=device)
            prompt_len = ids.shape[1]
            steps = args.new_tokens - 1
            caches = make_static_cache(mini, 1, prompt_len + args.new_tokens, dtype, device)
            decoder = GraphDecoder(mini, caches, 1, device)
            state = {}

            def setup_dynamic():
                state["first"], state["past"] = prefill(mini, ids)

            def setup_static():
                setup_dynamic()
                fill_static_cache(caches, state["past"])

            def setup_graph():
                setup_static()
                if decoder.graph is None:
                    decoder.capture(state["first"], prompt_len)

            variants = {
                "dynamic_eager_sync": (setup_dynamic, lambda n: decode_dynamic(mini, state["first"], state["past"], n, True)),
                "dynamic_eager_nosync": (setup_dynamic, lambda n: decode_dynamic(mini, state["first"], state["past"], n, False)),
                "static_eager_nosync": (setup_static, lambda n: decode_static_eager(mini, state["first"], caches, prompt_len, n)),
                "static_cuda_graph": (setup_graph, lambda n: decoder.run(state["first"], prompt_len, n)),
            }

            if args.bench:
                # Lower bound for one decode step: streaming all weights once at the measured bandwidth
                weights_bytes = sum(p.numel() * p.element_size() for p in mini.parameters())
                bandwidth = measure_bandwidth(device)
                report["memory_floor"] = {"weights_gb": weights_bytes / 1e9, "copy_bandwidth_gb_s": bandwidth,
                                          "weights_floor_ms_per_step": weights_bytes / (bandwidth * 1e9) * 1000}
                print(f"  weights {weights_bytes / 1e9:.2f} GB, measured copy bandwidth {bandwidth:.0f} GB/s, "
                      f"so streaming the weights once takes at least {report['memory_floor']['weights_floor_ms_per_step']:.1f} ms", flush=True)
                bench = {}
                for name, (setup, run) in variants.items():
                    bench[name] = time_decode(setup, run, steps, args.repeats)
                    print(f"  {name}: {bench[name]['ms_per_step']:.2f} ms/step "
                          f"(enqueue alone {bench[name]['enqueue_ms_per_step']:.2f} ms), "
                          f"{bench[name]['tokens_per_s']:.1f} tokens/s", flush=True)
                base_ms = bench["dynamic_eager_sync"]["ms_per_step"]
                for name in bench:
                    bench[name]["speedup_vs_dynamic_eager_sync"] = base_ms / bench[name]["ms_per_step"]
                report["bench_fp16"] = bench

            if args.profile:
                prof_steps = min(steps, 30)
                report["profile_fp16"] = {}
                for name in ["dynamic_eager_nosync", "static_cuda_graph"]:
                    setup, run = variants[name]
                    report["profile_fp16"][name] = profile_decode(setup, run, prof_steps)
                    print(f"  profile {name}: {report['profile_fp16'][name]}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"saved {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("selftest", help="tiny random model on the CPU")
    st.set_defaults(func=cmd_selftest)

    run = sub.add_parser("run", help="equivalence check, benchmark and profiler pass on the real model")
    run.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    run.add_argument("--prompt-len", type=int, default=512)
    run.add_argument("--new-tokens", type=int, default=128)
    run.add_argument("--repeats", type=int, default=5)
    run.add_argument("--check", action="store_true")
    run.add_argument("--bench", action="store_true")
    run.add_argument("--profile", action="store_true")
    run.add_argument("--out", default="results/profiling_exp6.json")
    run.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
