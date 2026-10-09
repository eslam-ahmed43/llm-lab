"""Exp 7: measure batch-1 decode on one model size, for a launch-aware roofline fitted across sizes.

Exp 6 showed that on a T4 the eager decode step of Qwen2.5-1.5B is CPU-bound: the GPU is busy for only
71% of the step and CUDA graphs remove part of the gap. This experiment asks how that changes with the
size of the model. This script measures ONE model; run it once per model, then fit and test the model of
the step time with `exp7_fit.py`.

For the given model it records, with the decode step of Exp 6 (batch of one, fp16, preallocated cache):
  * step time and CPU enqueue time of the eager step and of the CUDA-graph step
  * device operations per step and GPU busy time from torch.profiler
  * parameter count, weight bytes, layer count and the measured memory bandwidth

Examples:
  python exp7_launch_model.py --model Qwen/Qwen2.5-0.5B-Instruct --check --out results/launch_0p5b.json
  python exp7_launch_model.py --model Qwen/Qwen2.5-1.5B-Instruct --out results/launch_1p5b.json
  python exp7_launch_model.py --model Qwen/Qwen2.5-3B-Instruct --out results/launch_3b.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp5_internals import PROMPTS, build_mini, env_info, load_hf  # noqa: E402
from exp6_profiling import (  # noqa: E402
    GraphDecoder, decode_dynamic, decode_static_eager, equivalence, fill_static_cache,
    make_static_cache, measure_bandwidth, prefill, profile_decode, static_decode_step, time_decode)


def build(model_id, cfg, dtype, device):
    hf = load_hf(model_id, dtype)
    n_params = sum(p.numel() for p in hf.parameters())  # a tied output head is counted once
    mini = build_mini(cfg, hf.state_dict(), dtype, device)
    mini.set_attention("sdpa")
    return mini, n_params


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--check", action="store_true", help="fp32 equivalence of the decode variants (use for a small model)")
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoConfig, AutoTokenizer
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    report = {"env": env_info(), "model": args.model, "layers": cfg.num_hidden_layers,
              "hidden_size": cfg.hidden_size, "prompt_len": args.prompt_len,
              "new_tokens": args.new_tokens, "repeats": args.repeats}

    with torch.inference_mode():
        if args.check:
            print("[check] fp32 equivalence of the decode variants", flush=True)
            mini, _ = build(args.model, cfg, torch.float32, device)
            short = tok(PROMPTS[1], return_tensors="pt").input_ids.to(device)
            report["check_fp32"] = equivalence(mini, short, 12, torch.float32, device, use_graph=True)
            print("  ", report["check_fp32"], flush=True)
            mini = None
            torch.cuda.empty_cache()
            if not all(report["check_fp32"].values()):
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps(report, indent=2))
                print("EQUIVALENCE FAILED")
                sys.exit(1)

        print(f"[fp16] building {args.model}", flush=True)
        dtype = torch.float16
        mini, n_params = build(args.model, cfg, dtype, device)
        weights_bytes = sum(p.numel() * p.element_size() for p in mini.parameters())
        bandwidth = measure_bandwidth(device)
        report.update({"params": n_params, "weights_bytes": weights_bytes, "bandwidth_gb_s": bandwidth,
                       "floor_ms": weights_bytes / (bandwidth * 1e9) * 1000})
        print(f"  {n_params / 1e9:.2f}B parameters, {cfg.num_hidden_layers} layers, weights {weights_bytes / 1e9:.2f} GB, "
              f"bandwidth {bandwidth:.0f} GB/s, weight-streaming floor {report['floor_ms']:.1f} ms", flush=True)

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
            "dynamic_eager_nosync": (setup_dynamic, lambda n: decode_dynamic(mini, state["first"], state["past"], n, False)),
            "static_eager_nosync": (setup_static, lambda n: decode_static_eager(mini, state["first"], caches, prompt_len, n)),
            "static_cuda_graph": (setup_graph, lambda n: decoder.run(state["first"], prompt_len, n)),
        }

        # fp16 sanity: the static step and the dynamic step must give almost the same logits
        first, past = prefill(mini, ids)
        dynamic_logits = mini(first, past)[0][:, -1].float()
        fill_static_cache(caches, past)
        pos = torch.tensor([prompt_len], dtype=torch.long, device=device)
        static_logits = static_decode_step(mini, first, pos, caches)[:, -1].float()
        report["fp16_first_step_logit_diff"] = (dynamic_logits - static_logits).abs().max().item()
        report["fp16_first_step_logit_max"] = dynamic_logits.abs().max().item()

        bench = {}
        for name, (setup, run) in variants.items():
            bench[name] = time_decode(setup, run, steps, args.repeats)
            print(f"  {name}: {bench[name]['ms_per_step']:.2f} ms/step "
                  f"(enqueue alone {bench[name]['enqueue_ms_per_step']:.2f} ms)", flush=True)
        report["bench"] = bench

        if not args.no_profile:
            report["profile"] = {}
            for name, (setup, run) in variants.items():
                report["profile"][name] = profile_decode(setup, run, min(steps, 30))
                p = report["profile"][name]
                if "error" in p:
                    print(f"  profile {name}: ERROR {p['error']}", flush=True)
                else:
                    print(f"  profile {name}: {p['device_events_per_step']:.0f} device operations per step, "
                          f"GPU busy {p['device_busy_ms_per_step']:.2f} ms", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
