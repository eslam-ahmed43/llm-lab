"""Plot the single-stream decode time budget (Exp 6) from the result files of Exp 1, 5 and 6.

Bars show milliseconds per decoded token (batch of one, 512-token prompt, T4):
  Hugging Face generate           results/internals_qwen_verify.json   (Exp 5)
  own loop, eager, host sync      results/profiling_exp6.json          (Exp 6)
  own loop, eager, no host sync   results/profiling_exp6.json
  own static cache + CUDA graph   results/profiling_exp6.json
  vLLM, derived from Exp 1        results/vllm_p512_r*.json

For the eager loops the bar is split into GPU-busy time (torch.profiler, no-sync variant) and the rest,
when the GPU waits for the CPU. The dashed line is the time to stream all weights once at the measured
memory bandwidth.

Usage:
  python src/plot_exp6.py --results results --out plots/decode_budget.png
"""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def vllm_step_ms(results_dir, prompt_len=512, new_tokens=128):
    """Decode step of vLLM at concurrency 1, from end-to-end throughput and time to first token."""
    steps = []
    for path in sorted(Path(results_dir).glob(f"vllm_p{prompt_len}_r*.json")):
        row = next(r for r in json.loads(path.read_text())["rows"] if r["concurrency"] == 1)
        total = new_tokens / row["throughput_tok_s"]
        steps.append((total - row["ttft_p50_s"]) / (new_tokens - 1) * 1000)
    return statistics.mean(steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="plots/decode_budget.png")
    args = ap.parse_args()
    res = Path(args.results)

    prof = json.loads((res / "profiling_exp6.json").read_text())
    verify = json.loads((res / "internals_qwen_verify.json").read_text())
    bench = prof["bench_fp16"]
    busy = prof["profile_fp16"]["dynamic_eager_nosync"]["device_busy_ms_per_step"]
    floor = prof["memory_floor"]["weights_floor_ms_per_step"]
    hf_ms = verify["fp16"]["bench"]["hf_generate"]["decode_ms_per_token"]
    vllm_ms = vllm_step_ms(res)

    sync_ms = bench["dynamic_eager_sync"]["ms_per_step"]
    nosync_ms = bench["dynamic_eager_nosync"]["ms_per_step"]
    graph_ms = bench["static_cuda_graph"]["ms_per_step"]

    labels = ["Hugging Face generate\n(Exp 5)", "own loop, eager,\nhost sync per token", "own loop, eager,\nno host sync",
              "own static cache\n+ CUDA graph", "vLLM, derived\nfrom Exp 1"]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    y = list(range(len(labels)))[::-1]
    ax.barh(y[0], hf_ms, color="tab:gray")
    for yy, total in ((y[1], sync_ms), (y[2], nosync_ms)):
        ax.barh(yy, busy, color="tab:blue", label="GPU busy (profiler)" if yy == y[1] else None)
        ax.barh(yy, total - busy, left=busy, color="tab:orange", label="GPU waiting for the CPU" if yy == y[1] else None)
    ax.barh(y[3], graph_ms, color="tab:blue")
    ax.barh(y[4], vllm_ms, color="tab:gray", label="reference, not split")
    for yy, total in zip(y, (hf_ms, sync_ms, nosync_ms, graph_ms, vllm_ms)):
        ax.text(total + 0.4, yy, f"{total:.1f} ms", va="center", fontsize=9)
    ax.axvline(floor, color="black", linestyle="--", linewidth=1)
    ax.text(floor + 0.3, y[0] + 0.45, f"weights streamed once: {floor:.1f} ms", fontsize=8, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("milliseconds per decoded token (batch of one, fp16)")
    ax.set_xlim(0, hf_ms * 1.15)
    ax.set_title("Where the time goes in single-stream decoding on a T4")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
