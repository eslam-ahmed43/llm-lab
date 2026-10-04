"""Plot the Exp 4 (LoRA / QLoRA) summary table.

Reads results/lora_summary.csv, written by `python src/train_lora.py summarize`, and saves
plots/lora_overview.png with three panels:
  1. test accuracy against the number of trainable parameters (error bars: std over seeds)
  2. peak GPU memory during training against the LoRA rank
  3. training time against the LoRA rank

Usage:
  python src/plot_lora.py --summary results/lora_summary.csv --out plots/lora_overview.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = [
    ("lora", "attn", "LoRA, attention only", "tab:blue"),
    ("lora", "all", "LoRA, all linear layers", "tab:orange"),
    ("qlora", "all", "QLoRA (NF4 base), all linear layers", "tab:green"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/lora_summary.csv")
    ap.add_argument("--out", default="plots/lora_overview.png")
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    zero = df[df["mode"] == "zeroshot"]
    zero_acc = float(zero["acc_mean"].iloc[0]) if not zero.empty else None

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    for mode, targets, label, color in SERIES:
        g = df[(df["mode"] == mode) & (df["targets"] == targets)].sort_values("rank")
        if g.empty:
            continue
        axes[0].errorbar(g["trainable_params"] / 1e6, g["acc_mean"] * 100, yerr=g["acc_std"] * 100,
                         marker="o", capsize=3, label=label, color=color)
        axes[1].plot(g["rank"], g["peak_mem_gib"], marker="o", label=label, color=color)
        axes[2].plot(g["rank"], g["train_seconds"], marker="o", label=label, color=color)

    axes[0].set_xscale("log")
    axes[0].set_xlabel("trainable parameters (millions, log scale)")
    axes[0].set_ylabel("test accuracy (%)")
    title = "Accuracy vs trainable parameters"
    if zero_acc is not None:
        title += f"\n(no fine-tuning: {zero_acc * 100:.1f}%)"
    axes[0].set_title(title)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Peak GPU memory during training")
    axes[1].set_ylabel("GiB")
    axes[2].set_title("Training time (600 steps)")
    axes[2].set_ylabel("seconds")
    for ax in axes[1:]:
        ax.set_xscale("log", base=2)
        ax.set_xticks([4, 8, 16, 32])
        ax.set_xticklabels(["4", "8", "16", "32"])
        ax.set_xlabel("LoRA rank")
    for ax in axes:
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
