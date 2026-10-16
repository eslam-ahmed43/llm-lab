"""Fit and test a launch-aware model of the batch-1 decode step (Exp 7 and Exp 7b).

Reads results/<prefix>_*.json, one file per model size and precision, written by exp7_launch_model.py.
The model is fitted on ONE reference run and then used to predict the other runs without looking at
their step times.

Model of one decode step (static cache, batch of one):
  GPU time    T_gpu = weight_bytes / bandwidth + layers * c_layer
  CPU time    T_cpu = layers * ops_per_layer * t_launch
  eager step  = max(T_gpu, T_cpu)          (the slower side decides)
  graph step  = T_gpu                      (a CUDA graph removes the launch cost)

Fitted on the reference run: c_layer (GPU time per layer beyond streaming its weights), ops_per_layer
(device operations per layer, from the profiler) and t_launch (CPU time per operation). The bandwidth is
the one measured for the reference run and is used for all runs.

Two simpler models are scored as baselines, scaling the reference step time by weight bytes only
(a pure bandwidth model) or by layer count only (a pure launch model). Note that in the CPU-bound regime
the eager prediction of the model equals the layer-count baseline by construction, so the graph step and
the regime are what separate the models.

The preregistered acceptance criteria are evaluated on all non-reference runs:
  1. every predicted eager and graph step is within 20% of the measurement
  2. the predicted regime (CPU-bound or GPU-bound) equals the measured one
  3. the mean absolute error of the model is lower than that of both baselines

Usage:
  python src/exp7_fit.py --results-dir results --prefix launch --reference 1.5B \\
      --out results/launch_model_fit.json --plot plots/launch_model.png
  python src/exp7_fit.py --results-dir results --prefix launch7b --reference 1.5B --reference-dtype fp16 \\
      --out results/launch7b_fit.json --plot plots/launch7b_model.png
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TOLERANCE_PCT = 20.0


def load_runs(results_dir, prefix):
    runs = []
    for path in sorted(Path(results_dir).glob(f"{prefix}_*.json")):
        if "fit" in path.name or "smoke" in path.name:
            continue
        run = json.loads(path.read_text())
        if "bench" in run:
            run.setdefault("dtype", "fp16")
            run["file"] = path.name
            runs.append(run)
    return sorted(runs, key=lambda r: (r["weights_bytes"], r["dtype"]))


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
    cpu_bound = eager["enqueue_ms_per_step"] >= 0.95 * eager["ms_per_step"]
    out = {"eager_ms": eager["ms_per_step"], "graph_ms": graph["ms_per_step"],
           "speedup": eager["ms_per_step"] / graph["ms_per_step"],
           "regime": "CPU-bound" if cpu_bound else "GPU-bound"}
    prof = run.get("profile", {}).get("static_eager_nosync", {})
    if "device_busy_ms_per_step" in prof:
        out["gpu_busy_fraction_eager"] = min(1.0, prof["device_busy_ms_per_step"] / eager["ms_per_step"])
    return out


def pct(pred, meas):
    return (pred / meas - 1) * 100


def label(run):
    return f"{run['model'].split('/')[-1]} {run['dtype']}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--prefix", default="launch")
    ap.add_argument("--reference", default="1.5B", help="substring of the model name used to fit the model")
    ap.add_argument("--reference-dtype", default="fp16")
    ap.add_argument("--out", default="results/launch_model_fit.json")
    ap.add_argument("--plot", default="plots/launch_model.png")
    args = ap.parse_args()

    runs = load_runs(args.results_dir, args.prefix)
    refs = [r for r in runs if args.reference in r["model"] and r["dtype"] == args.reference_dtype]
    if not refs or len(runs) < 2:
        raise SystemExit(f"need the reference ({args.reference}, {args.reference_dtype}) and at least one more run, "
                         f"found {[label(r) for r in runs]}")
    ref = refs[0]
    fit = fit_reference(ref)
    if not fit["reference_is_cpu_bound"]:
        print("WARNING: the reference step is not CPU-bound, so t_launch is only an upper bound")
    print(f"fitted on {label(ref)}: c_layer {fit['c_layer_ms'] * 1000:.0f} us, "
          f"{fit['ops_per_layer']:.1f} operations per layer, t_launch {fit['t_launch_ms'] * 1000:.1f} us per operation")

    ref_m = measured(ref)
    rows = []
    for run in runs:
        pr, me = predict(run, fit), measured(run)
        w_ratio = run["weights_bytes"] / ref["weights_bytes"]
        l_ratio = run["layers"] / ref["layers"]
        rows.append({
            "model": run["model"], "dtype": run["dtype"], "file": run["file"], "params_b": run["params"] / 1e9,
            "weights_gb": run["weights_bytes"] / 1e9, "layers": run["layers"], "is_reference": run is ref,
            "predicted": pr, "measured": me,
            "error_pct": {"eager": pct(pr["eager_ms"], me["eager_ms"]), "graph": pct(pr["graph_ms"], me["graph_ms"])},
            "baselines_error_pct": {
                "bandwidth_only": {"eager": pct(ref_m["eager_ms"] * w_ratio, me["eager_ms"]),
                                   "graph": pct(ref_m["graph_ms"] * w_ratio, me["graph_ms"])},
                "layers_only": {"eager": pct(ref_m["eager_ms"] * l_ratio, me["eager_ms"]),
                                "graph": pct(ref_m["graph_ms"] * l_ratio, me["graph_ms"])},
            },
            "regime_correct": pr["regime"] == me["regime"],
        })

    print(f"\n{'run':<30}{'meas eager':>11}{'pred':>8}{'err%':>7}  {'meas graph':>11}{'pred':>8}{'err%':>7}  "
          f"{'speedup m/p':>12}  regime meas / pred")
    for r, run in zip(rows, runs):
        p, m = r["predicted"], r["measured"]
        tag = "  (fit)" if r["is_reference"] else ""
        print(f"{label(run):<30}{m['eager_ms']:>11.2f}{p['eager_ms']:>8.2f}{r['error_pct']['eager']:>+7.1f}  "
              f"{m['graph_ms']:>11.2f}{p['graph_ms']:>8.2f}{r['error_pct']['graph']:>+7.1f}  "
              f"{m['speedup']:>5.2f} /{p['speedup']:>5.2f}  {m['regime']} / {p['regime']}{tag}")

    tests = [r for r in rows if not r["is_reference"]]
    errors = {"model": [], "bandwidth_only": [], "layers_only": []}
    for r in tests:
        errors["model"] += [r["error_pct"]["eager"], r["error_pct"]["graph"]]
        for name in ("bandwidth_only", "layers_only"):
            errors[name] += [r["baselines_error_pct"][name]["eager"], r["baselines_error_pct"][name]["graph"]]
    mae = {k: sum(abs(x) for x in v) / len(v) for k, v in errors.items()}
    criteria = {
        "1_errors_within_20_percent": all(abs(x) <= TOLERANCE_PCT for x in errors["model"]),
        "2_regime_correct_for_all": all(r["regime_correct"] for r in tests),
        "3_lower_mean_error_than_both_baselines": mae["model"] < mae["bandwidth_only"] and mae["model"] < mae["layers_only"],
    }
    print("\nmean absolute error over all predicted eager and graph steps of the test runs: "
          + ", ".join(f"{k} {v:.1f}%" for k, v in mae.items()))
    print("regimes measured in the test runs: " + ", ".join(f"{label(run)}: {r['measured']['regime']}"
                                                              for r, run in zip(rows, runs) if not r["is_reference"]))
    for name, ok in criteria.items():
        print(f"  criterion {name}: {'PASS' if ok else 'FAIL'}")
    print("  model accepted by the preregistered criteria" if all(criteria.values()) else "  model NOT accepted by the preregistered criteria")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"reference": label(ref), "fit": fit, "rows": rows,
                                          "mean_abs_error_pct": mae, "criteria": criteria}, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    for dtype, marker in (("fp16", "o"), ("fp32", "s")):
        sel = [r for r in rows if r["dtype"] == dtype]
        if not sel:
            continue
        xs = [r["weights_gb"] for r in sel]
        axes[0].plot(xs, [r["measured"]["eager_ms"] for r in sel], marker + "-", color="tab:orange", label=f"measured eager ({dtype})")
        axes[0].plot(xs, [r["measured"]["graph_ms"] for r in sel], marker + "-", color="tab:blue", label=f"measured graph ({dtype})")
        axes[0].plot(xs, [r["predicted"]["eager_ms"] for r in sel], "x--", color="tab:orange", label=f"predicted eager ({dtype})")
        axes[0].plot(xs, [r["predicted"]["graph_ms"] for r in sel], "x--", color="tab:blue", label=f"predicted graph ({dtype})")
        axes[1].plot(xs, [r["measured"]["speedup"] for r in sel], marker + "-", color="tab:green", label=f"measured ({dtype})")
        axes[1].plot(xs, [r["predicted"]["speedup"] for r in sel], "x--", color="tab:green", label=f"predicted ({dtype})")
    axes[0].set_ylabel("milliseconds per decoded token")
    axes[0].set_title("Decode step time against weight bytes")
    axes[1].axhline(1.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("speed-up from a CUDA graph")
    axes[1].set_title("Where graphs help: eager step / graph step")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("weight bytes read per token (GB, log scale)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=150)
    print(f"\nsaved {args.out} and {args.plot}")


if __name__ == "__main__":
    main()
