"""Fit and test a launch-aware model of the batch-1 decode step (Exp 7).

Reads results/launch_*.json, one file per model size written by exp7_launch_model.py. The model is fitted
on ONE reference model and then used to predict the other sizes without looking at their step times.

Model of one decode step (static cache, batch of one):
  GPU time    T_gpu = weight_bytes / bandwidth + layers * c_layer
  CPU time    T_cpu = layers * ops_per_layer * t_launch
  eager step  = max(T_gpu, T_cpu)          (the slower side decides)
  graph step  = T_gpu                      (a CUDA graph removes the launch cost)

Fitted on the reference model: c_layer (GPU time per layer beyond streaming its weights),
ops_per_layer (device operations per layer, from the profiler) and t_launch (CPU time per operation).
The bandwidth is the one measured for the reference run and is used for all sizes.

Two simpler models are scored as baselines, scaling the reference step time by weight bytes only
(a pure bandwidth model) or by layer count only (a pure launch model).

Usage:
  python src/exp7_fit.py --results-dir results --reference 1.5B \\
      --out results/launch_model_fit.json --plot plots/launch_model.png
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_runs(results_dir):
    runs = []
    for path in sorted(Path(results_dir).glob("launch_*.json")):
        if "fit" in path.name or "smoke" in path.name:
            continue
        run = json.loads(path.read_text())
        if "bench" in run:
            runs.append(run)
    return sorted(runs, key=lambda r: r["params"])


def fit_reference(ref):
    prof = ref.get("profile", {}).get("static_eager_nosync", {})
    if "device_events_per_step" not in prof:
        raise SystemExit("the reference run has no profiler data for the static eager step")
    bw = ref["bandwidth_gb_s"] * 1e9
    layers = ref["layers"]
    eager = ref["bench"]["static_eager_nosync"]["ms_per_step"]
    enqueue = ref["bench"]["static_eager_nosync"]["enqueue_ms_per_step"]
    graph = ref["bench"]["static_cuda_graph"]["ms_per_step"]
    ops = prof["device_events_per_step"]
    floor = ref["weights_bytes"] / bw * 1000
    return {
        "bandwidth_bytes_per_s": bw,
        "c_layer_ms": (graph - floor) / layers,
        "ops_per_layer": ops / layers,
        "t_launch_ms": eager / ops,
        "reference_is_cpu_bound": enqueue >= 0.95 * eager,
        "reference_floor_ms": floor,
    }


def predict(run, fit):
    gpu = run["weights_bytes"] / fit["bandwidth_bytes_per_s"] * 1000 + run["layers"] * fit["c_layer_ms"]
    cpu = run["layers"] * fit["ops_per_layer"] * fit["t_launch_ms"]
    eager = max(gpu, cpu)
    return {"gpu_ms": gpu, "cpu_ms": cpu, "eager_ms": eager, "graph_ms": gpu,
            "speedup": eager / gpu, "regime": "CPU-bound" if cpu > gpu else "GPU-bound"}


def measured(run):
    eager = run["bench"]["static_eager_nosync"]
    graph = run["bench"]["static_cuda_graph"]
    out = {"eager_ms": eager["ms_per_step"], "graph_ms": graph["ms_per_step"],
           "speedup": eager["ms_per_step"] / graph["ms_per_step"],
           "eager_cpu_bound": eager["enqueue_ms_per_step"] >= 0.95 * eager["ms_per_step"]}
    prof = run.get("profile", {}).get("static_eager_nosync", {})
    if "device_busy_ms_per_step" in prof:
        out["gpu_busy_fraction_eager"] = min(1.0, prof["device_busy_ms_per_step"] / eager["ms_per_step"])
    out["regime"] = "CPU-bound" if out["eager_cpu_bound"] else "GPU-bound"
    return out


def pct(pred, meas):
    return (pred / meas - 1) * 100


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--reference", default="1.5B", help="substring of the model name used to fit the model")
    ap.add_argument("--out", default="results/launch_model_fit.json")
    ap.add_argument("--plot", default="plots/launch_model.png")
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    refs = [r for r in runs if args.reference in r["model"]]
    if not refs or len(runs) < 2:
        raise SystemExit(f"need the reference ({args.reference}) and at least one more model, found {[r['model'] for r in runs]}")
    ref = refs[0]
    fit = fit_reference(ref)
    if not fit["reference_is_cpu_bound"]:
        print("WARNING: the reference step is not CPU-bound, so t_launch is only an upper bound")
    print(f"fitted on {ref['model']}: c_layer {fit['c_layer_ms'] * 1000:.0f} us, "
          f"{fit['ops_per_layer']:.1f} operations per layer, t_launch {fit['t_launch_ms'] * 1000:.1f} us per operation")

    rows = []
    for run in runs:
        pr, me = predict(run, fit), measured(run)
        w_ratio = run["weights_bytes"] / ref["weights_bytes"]
        l_ratio = run["layers"] / ref["layers"]
        ref_m = measured(ref)
        row = {
            "model": run["model"], "params_b": run["params"] / 1e9, "layers": run["layers"],
            "is_reference": run is ref, "predicted": pr, "measured": me,
            "error_pct": {"eager": pct(pr["eager_ms"], me["eager_ms"]), "graph": pct(pr["graph_ms"], me["graph_ms"])},
            "baselines_error_pct": {
                "bandwidth_only": {"eager": pct(ref_m["eager_ms"] * w_ratio, me["eager_ms"]),
                                   "graph": pct(ref_m["graph_ms"] * w_ratio, me["graph_ms"])},
                "layers_only": {"eager": pct(ref_m["eager_ms"] * l_ratio, me["eager_ms"]),
                                "graph": pct(ref_m["graph_ms"] * l_ratio, me["graph_ms"])},
            },
            "regime_correct": pr["regime"] == me["regime"],
        }
        rows.append(row)

    header = (f"{'model':<28}{'meas eager':>11}{'pred':>8}{'err%':>7}  {'meas graph':>11}{'pred':>8}{'err%':>7}  "
              f"{'speedup meas/pred':>18}  regime meas/pred")
    print("\n" + header)
    for r in rows:
        p, m = r["predicted"], r["measured"]
        tag = "  (fit)" if r["is_reference"] else ""
        print(f"{r['model']:<28}{m['eager_ms']:>11.2f}{p['eager_ms']:>8.2f}{r['error_pct']['eager']:>+7.1f}  "
              f"{m['graph_ms']:>11.2f}{p['graph_ms']:>8.2f}{r['error_pct']['graph']:>+7.1f}  "
              f"{m['speedup']:>8.2f} /{p['speedup']:>6.2f}  {m['regime']} / {p['regime']}{tag}")
    print("\nbaselines, error of the eager step (%): "
          + "; ".join(f"{r['model'].split('/')[-1]} bandwidth-only {r['baselines_error_pct']['bandwidth_only']['eager']:+.0f}, "
                      f"layers-only {r['baselines_error_pct']['layers_only']['eager']:+.0f}" for r in rows if not r["is_reference"]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"reference": ref["model"], "fit": fit, "rows": rows}, indent=2))

    xs = [r["params_b"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    ax.plot(xs, [r["measured"]["eager_ms"] for r in rows], "o-", color="tab:orange", label="measured, eager")
    ax.plot(xs, [r["measured"]["graph_ms"] for r in rows], "s-", color="tab:blue", label="measured, CUDA graph")
    ax.plot(xs, [r["predicted"]["eager_ms"] for r in rows], "x--", color="tab:orange", label="predicted, eager")
    ax.plot(xs, [r["predicted"]["graph_ms"] for r in rows], "x--", color="tab:blue", label="predicted, CUDA graph")
    ax.set_xscale("log")
    ax.set_xlabel("parameters (billions, log scale)")
    ax.set_ylabel("milliseconds per decoded token")
    ax.set_title("Decode step time against model size")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(xs, [r["measured"]["speedup"] for r in rows], "o-", color="tab:green", label="measured")
    ax.plot(xs, [r["predicted"]["speedup"] for r in rows], "x--", color="tab:green", label="predicted")
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("parameters (billions, log scale)")
    ax.set_ylabel("speed-up from a CUDA graph")
    ax.set_title("Where graphs help: eager step time / graph step time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=150)
    print(f"\nsaved {args.out} and {args.plot}")


if __name__ == "__main__":
    main()
