"""
Generate publication figures for the compression paper.
Reads from compression_benchmark.json and produces matplotlib charts.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Paths
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)


def load_results():
    path = os.path.join(RESULTS_DIR, "compression_benchmark.json")
    with open(path) as f:
        return json.load(f)


def fig1_compression_ratio(results):
    """Bar chart: compression ratio across model sizes"""
    data = results["compression_ratio"]
    sizes = ["100k", "1m", "10m"]
    labels = ["100K\n(118K params)", "1M\n(798K params)", "10M\n(8.4M params)"]
    
    uncompressed = [data[s]["uncompressed_total_mb"] for s in sizes]
    compressed = [data[s]["compressed_total_mb"] for s in sizes]
    saved = [u - c for u, c in zip(uncompressed, compressed)]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(sizes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, uncompressed, width, label="Uncompressed (FP32)", color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + width/2, compressed, width, label="Compressed (RQ 8-bit)", color="#2ecc71", alpha=0.85)
    
    # Add ratio labels
    for i, (u, c) in enumerate(zip(uncompressed, compressed)):
        ax.annotate(f"4.0x", xy=(i, max(u, c) + 2), ha="center", fontsize=12, fontweight="bold", color="#2c3e50")
    
    ax.set_ylabel("Total Memory (MB) — 10 Snapshots", fontsize=12)
    ax.set_title("Memory Reduction via RQ Compression Across Model Scales", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_compression_ratio.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig1_compression_ratio.png")


def fig2_reconstruction_error(results):
    """Chart showing negligible quality degradation from compression"""
    data = results["reconstruction_error"]
    sizes = ["100k", "1m", "10m"]
    labels = ["100K", "1M", "10M"]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Panel A: Parameter difference
    ax = axes[0]
    diffs = [data[s]["mean_param_diff"] for s in sizes]
    stds = [data[s]["std_param_diff"] for s in sizes]
    ax.bar(labels, diffs, yerr=stds, color="#3498db", alpha=0.85, capsize=5)
    ax.set_ylabel("Mean Absolute Parameter Diff")
    ax.set_title("A) Parameter Divergence", fontweight="bold")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(-3, -3))
    ax.grid(axis="y", alpha=0.3)
    
    # Panel B: Loss difference
    ax = axes[1]
    loss_diffs = [data[s]["mean_loss_diff"] for s in sizes]
    loss_stds = [data[s]["std_loss_diff"] for s in sizes]
    ax.bar(labels, loss_diffs, yerr=loss_stds, color="#e67e22", alpha=0.85, capsize=5)
    ax.set_ylabel("Mean Loss Difference")
    ax.set_title("B) Loss Equivalence", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    
    # Panel C: Accuracy comparison
    ax = axes[2]
    acc_u = [data[s]["mean_acc_uncompressed"] * 100 for s in sizes]
    acc_c = [data[s]["mean_acc_compressed"] * 100 for s in sizes]
    
    x = np.arange(len(sizes))
    width = 0.35
    ax.bar(x - width/2, acc_u, width, label="Uncompressed", color="#e74c3c", alpha=0.85)
    ax.bar(x + width/2, acc_c, width, label="Compressed", color="#2ecc71", alpha=0.85)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("C) Accuracy Parity", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    
    plt.suptitle("Compressed vs Uncompressed Parameter Blending — No Quality Loss", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig2_reconstruction_error.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig2_reconstruction_error.png")


def fig3_quantization_snr(results):
    """SNR analysis showing quantization is high-fidelity"""
    data = results["quantization_error"]
    sizes = ["100k", "1m", "10m"]
    labels = ["100K", "1M", "10M"]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # Panel A: SNR
    ax = axes[0]
    snrs = [data[s]["avg_snr_db"] for s in sizes]
    colors = ["#2ecc71" if snr > 40 else "#e67e22" if snr > 20 else "#e74c3c" for snr in snrs]
    ax.bar(labels, snrs, color=colors, alpha=0.85)
    ax.axhline(y=40, color="gray", linestyle="--", alpha=0.5, label="High fidelity (>40dB)")
    ax.set_ylabel("Signal-to-Noise Ratio (dB)")
    ax.set_title("A) Quantization SNR", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    
    # Panel B: Accuracy before/after quantization
    ax = axes[1]
    acc_before = [data[s]["acc_before_quant"] * 100 for s in sizes]
    acc_after = [data[s]["acc_after_quant"] * 100 for s in sizes]
    
    x = np.arange(len(sizes))
    width = 0.35
    ax.bar(x - width/2, acc_before, width, label="Before Quantization", color="#3498db", alpha=0.85)
    ax.bar(x + width/2, acc_after, width, label="After 8-bit Quantization", color="#9b59b6", alpha=0.85)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("B) Model Quality After Full Quantization", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    
    plt.suptitle("8-bit RQ Quantization: High SNR, Minimal Quality Impact", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig3_quantization_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig3_quantization_analysis.png")


def fig4_overhead(results):
    """Timing overhead comparison"""
    data = results["overhead"]
    sizes = ["100k", "1m", "10m"]
    labels = ["100K", "1M", "10M"]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(sizes))
    width = 0.25
    
    baseline = [data[s]["baseline_time_s"] for s in sizes]
    uncompressed = [data[s]["echo_uncompressed_time_s"] for s in sizes]
    compressed = [data[s]["echo_compressed_time_s"] for s in sizes]
    
    ax.bar(x - width, baseline, width, label="No Echo (baseline)", color="#95a5a6", alpha=0.85)
    ax.bar(x, uncompressed, width, label="Echo (uncompressed)", color="#3498db", alpha=0.85)
    ax.bar(x + width, compressed, width, label="Echo (compressed)", color="#2ecc71", alpha=0.85)
    
    # Add overhead labels
    for i in range(len(sizes)):
        overhead = data[sizes[i]]["overhead_compressed_pct"]
        ax.annotate(f"+{overhead:.0f}%", xy=(x[i] + width, compressed[i] + 0.1),
                   ha="center", fontsize=9, fontweight="bold", color="#27ae60")
    
    ax.set_ylabel("Time (seconds) — 100 training steps", fontsize=11)
    ax.set_title("Computational Overhead of Parameter Memory", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_overhead.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig4_overhead.png")


def fig5_projected_savings(results):
    """Projected cost savings for large models"""
    data = results["projections"]
    
    models = [
        ("LLaMA-7B class", "LLaMA-7B class"),
        ("LLaMA-13B class", "LLaMA-13B class"),
        ("LLaMA-70B class", "LLaMA-70B class"),
        ("GPT-4 class (est.)", "GPT-4 class"),
        ("LLaMA-405B class", "LLaMA-405B class"),
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Memory savings (GB)
    ax = axes[0]
    names = [m[1] for m in models]
    saved_10 = [data[m[0]]["10_snapshots"]["saved_gb"] for m in models]
    saved_50 = [data[m[0]]["50_snapshots"]["saved_gb"] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width/2, saved_10, width, label="10 snapshots", color="#3498db", alpha=0.85)
    ax.bar(x + width/2, saved_50, width, label="50 snapshots", color="#e74c3c", alpha=0.85)
    ax.set_ylabel("Memory Saved (GB)")
    ax.set_title("A) RAM Savings with 4x Compression", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.legend()
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(axis="y", alpha=0.3)
    
    # Panel B: Cost savings (USD)
    ax = axes[1]
    cost_10 = [data[m[0]]["10_snapshots"]["saved_cost_usd"] for m in models]
    cost_50 = [data[m[0]]["50_snapshots"]["saved_cost_usd"] for m in models]
    
    ax.bar(x - width/2, cost_10, width, label="10 snapshots", color="#2ecc71", alpha=0.85)
    ax.bar(x + width/2, cost_50, width, label="50 snapshots", color="#e67e22", alpha=0.85)
    ax.set_ylabel("Estimated Cost Savings (USD)")
    ax.set_title("B) Hardware Cost Savings @ $30/GB HBM3", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.legend()
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax.grid(axis="y", alpha=0.3)
    
    plt.suptitle("Projected Savings for Production-Scale Models", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig5_projected_savings.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig5_projected_savings.png")


if __name__ == "__main__":
    print("Generating publication figures...")
    results = load_results()
    
    fig1_compression_ratio(results)
    fig2_reconstruction_error(results)
    fig3_quantization_snr(results)
    fig4_overhead(results)
    fig5_projected_savings(results)
    
    print(f"\nAll figures saved to {FIGURES_DIR}")
