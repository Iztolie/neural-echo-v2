"""
Per-Tensor Error Breakdown for GPT-OSS 20B
============================================
Identifies which tensor categories (MoE experts, router, attention, 
layernorm, embedding) are responsible for the degradation from 
48.5 dB (toy models) to 40.0 dB (12B MoE).

Outputs:
1. Detailed per-tensor metrics (CSV + JSON)
2. Category-level summary statistics
3. Histogram data for visualization
4. Per-layer heatmap data
"""

import sys
import os
import time
import json
import csv
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from safetensors.torch import load_file


# ============================================================
# Configuration
# ============================================================

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "model_cache", "models--openai--gpt-oss-20b"
)

SNAPSHOT_HASH = "6cee5e81ee83917806bbde320786a8fb61efebee"
SNAPSHOT_DIR = os.path.join(MODEL_DIR, "snapshots", SNAPSHOT_HASH)

SHARD_FILES = [
    "model-00000-of-00002.safetensors",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]


def resolve_shard_path(filename):
    """Resolve a safetensors filename to its actual path (handles HF cache symlinks)."""
    path = os.path.join(SNAPSHOT_DIR, filename)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                f.read(8)
            return path
        except:
            pass
    blobs_dir = os.path.join(MODEL_DIR, "blobs")
    blob_files = []
    for f in os.listdir(blobs_dir):
        fp = os.path.join(blobs_dir, f)
        size = os.path.getsize(fp)
        if size > 1024**3:
            blob_files.append((fp, size))
    blob_files.sort(key=lambda x: x[1])
    shard_idx = SHARD_FILES.index(filename)
    if shard_idx < len(blob_files):
        return blob_files[shard_idx][0]
    raise FileNotFoundError(f"Cannot resolve shard: {filename}")


def classify_tensor(name):
    """Classify a tensor name into a category."""
    if "experts" in name:
        if "blocks" in name:
            return "expert_blocks"
        elif "scales" in name:
            return "expert_scales"
        elif "bias" in name:
            return "expert_bias"
        return "expert_other"
    elif "router" in name:
        return "router"
    elif "self_attn" in name:
        if "sinks" in name:
            return "attention_sinks"
        elif "bias" in name:
            return "attention_bias"
        else:
            return "attention_weight"
    elif "layernorm" in name:
        return "layernorm"
    elif "embed_tokens" in name:
        return "embedding"
    elif "lm_head" in name:
        return "lm_head"
    else:
        return "other"
    

def extract_layer_num(name):
    """Extract layer number from tensor name. Returns -1 for non-layer tensors."""
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return -1


def quantize_and_measure_detailed(name, tensor):
    """
    Quantize a single FP32 tensor to INT8, measure detailed error stats.
    Returns comprehensive metrics dict.
    """
    t = tensor.detach().float().cpu()
    numel = t.numel()
    
    # Weight statistics (pre-quantization)
    t_min = t.min().item()
    t_max = t.max().item()
    t_mean = t.mean().item()
    t_std = t.std().item()
    dynamic_range = t_max - t_min
    
    # Quantize
    scale = dynamic_range / 255.0
    if scale == 0.0:
        scale = 1.0
    
    quantized = ((t - t_min) / scale).round().clamp(0, 255).to(torch.uint8)
    
    # Dequantize
    restored = quantized.float() * scale + t_min
    del quantized
    
    # Error metrics
    error = (t - restored).abs()
    sq_error = (t - restored) ** 2
    del restored
    
    mean_abs_error = error.mean().item()
    max_abs_error = error.max().item()
    # For large tensors, sample for percentiles to avoid OOM
    if numel > 10_000_000:
        sample_idx = torch.randperm(numel)[:1_000_000]
        error_sample = error.flatten()[sample_idx]
        p95_error = torch.quantile(error_sample.float(), 0.95).item()
        p99_error = torch.quantile(error_sample.float(), 0.99).item()
        del error_sample, sample_idx
    else:
        p95_error = torch.quantile(error.float(), 0.95).item()
        p99_error = torch.quantile(error.float(), 0.99).item()
    
    mean_signal = t.abs().mean().item()
    signal_power = (t ** 2).mean().item()
    noise_power = sq_error.mean().item()
    
    del error, sq_error
    
    relative_error = mean_abs_error / max(mean_signal, 1e-10)
    
    if noise_power > 0 and signal_power > 0:
        snr_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_db = float('inf')
    
    # Outlier analysis: how many values are near min/max (top/bottom 1%)
    sorted_vals = t.flatten()
    n_vals = len(sorted_vals)
    
    category = classify_tensor(name)
    layer_num = extract_layer_num(name)
    
    return {
        "name": name,
        "category": category,
        "layer": layer_num,
        "numel": numel,
        "shape": list(t.shape),
        "fp32_bytes": numel * 4,
        "int8_bytes": numel + 8,
        # Weight distribution stats
        "w_min": t_min,
        "w_max": t_max,
        "w_mean": t_mean,
        "w_std": t_std,
        "dynamic_range": dynamic_range,
        # Error metrics
        "mean_abs_error": mean_abs_error,
        "max_abs_error": max_abs_error,
        "p95_error": p95_error,
        "p99_error": p99_error,
        "relative_error": relative_error,
        "snr_db": float(snr_db) if snr_db != float('inf') else 999.0,
        # Signal stats
        "mean_signal": mean_signal,
        "signal_power": signal_power,
        "noise_power": noise_power,
    }


def run_analysis():
    """Run per-tensor error analysis across all shards."""
    print("=" * 72)
    print("NEURAL ECHO V2 — PER-TENSOR ERROR BREAKDOWN")
    print(f"Model: openai/gpt-oss-20b (MoE, 32 experts, 4 active)")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 72)
    
    all_tensors = []
    
    for shard_idx, shard_file in enumerate(SHARD_FILES):
        print(f"\n--- Shard {shard_idx + 1}/{len(SHARD_FILES)}: {shard_file} ---")
        
        shard_path = resolve_shard_path(shard_file)
        shard_tensors = load_file(shard_path, device="cpu")
        
        n_tensors = len(shard_tensors)
        print(f"  Loading {n_tensors} tensors...")
        
        tensor_names = list(shard_tensors.keys())
        for ti, name in enumerate(tensor_names):
            tensor = shard_tensors[name].float().cpu()
            del shard_tensors[name]
            
            metrics = quantize_and_measure_detailed(name, tensor)
            metrics["shard"] = shard_file
            metrics["shard_idx"] = shard_idx
            all_tensors.append(metrics)
            
            del tensor
            
            if (ti + 1) % 50 == 0 or ti == n_tensors - 1:
                gc.collect()
                print(f"    [{ti+1}/{n_tensors}] processed")
        
        del shard_tensors
        gc.collect()
    
    # ============================================================
    # Category-level analysis
    # ============================================================
    print("\n" + "=" * 72)
    print("CATEGORY-LEVEL ANALYSIS")
    print("=" * 72)
    
    categories = {}
    for t in all_tensors:
        cat = t["category"]
        if cat not in categories:
            categories[cat] = {
                "tensors": [],
                "total_params": 0,
                "total_fp32_bytes": 0,
            }
        categories[cat]["tensors"].append(t)
        categories[cat]["total_params"] += t["numel"]
        categories[cat]["total_fp32_bytes"] += t["fp32_bytes"]
    
    category_summary = []
    for cat, data in sorted(categories.items()):
        tensors = data["tensors"]
        n = len(tensors)
        total_params = data["total_params"]
        
        # Weighted metrics (by parameter count)
        weighted_rel_error = sum(t["relative_error"] * t["numel"] for t in tensors) / total_params
        weighted_snr = sum(t["snr_db"] * t["numel"] for t in tensors if t["snr_db"] < 900) / total_params
        
        avg_rel_error = np.mean([t["relative_error"] for t in tensors])
        avg_snr = np.mean([t["snr_db"] for t in tensors if t["snr_db"] < 900])
        max_rel_error = max(t["relative_error"] for t in tensors)
        min_snr = min(t["snr_db"] for t in tensors if t["snr_db"] < 900) if any(t["snr_db"] < 900 for t in tensors) else 999.0
        
        avg_dynamic_range = np.mean([t["dynamic_range"] for t in tensors])
        
        pct_params = total_params / sum(c["total_params"] for c in categories.values()) * 100
        
        summary = {
            "category": cat,
            "count": n,
            "total_params": total_params,
            "pct_params": round(pct_params, 1),
            "weighted_relative_error": round(weighted_rel_error, 6),
            "avg_relative_error": round(avg_rel_error, 6),
            "max_relative_error": round(max_rel_error, 6),
            "weighted_snr_db": round(float(weighted_snr), 1),
            "avg_snr_db": round(float(avg_snr), 1),
            "min_snr_db": round(float(min_snr), 1),
            "avg_dynamic_range": round(avg_dynamic_range, 6),
        }
        category_summary.append(summary)
        
        print(f"\n  {cat}:")
        print(f"    Tensors: {n} | Params: {total_params:,} ({pct_params:.1f}%)")
        print(f"    Weighted RelErr: {weighted_rel_error:.4%} | Avg RelErr: {avg_rel_error:.4%} | Max RelErr: {max_rel_error:.4%}")
        print(f"    Weighted SNR: {weighted_snr:.1f} dB | Avg SNR: {avg_snr:.1f} dB | Min SNR: {min_snr:.1f} dB")
        print(f"    Avg Dynamic Range: {avg_dynamic_range:.4f}")
    
    # ============================================================
    # Per-layer heatmap (relative error by layer × category)
    # ============================================================
    print("\n" + "=" * 72)
    print("PER-LAYER ERROR HEATMAP")
    print("=" * 72)
    
    layer_data = {}
    for t in all_tensors:
        if t["layer"] < 0:
            continue
        layer = t["layer"]
        cat = t["category"]
        if layer not in layer_data:
            layer_data[layer] = {}
        if cat not in layer_data[layer]:
            layer_data[layer][cat] = {"errors": [], "params": 0}
        layer_data[layer][cat]["errors"].append(t["relative_error"])
        layer_data[layer][cat]["params"] += t["numel"]
    
    # Print compact heatmap
    all_cats_in_layers = sorted(set(
        cat for ld in layer_data.values() for cat in ld.keys()
    ))
    
    header = f"{'Layer':>6}"
    for cat in all_cats_in_layers:
        header += f" | {cat[:12]:>12}"
    print(header)
    print("-" * len(header))
    
    layer_heatmap = []
    for layer in sorted(layer_data.keys()):
        row = {"layer": layer}
        line = f"{layer:>6}"
        for cat in all_cats_in_layers:
            if cat in layer_data[layer]:
                avg_err = np.mean(layer_data[layer][cat]["errors"])
                line += f" | {avg_err:>11.4%}"
                row[cat] = round(avg_err, 6)
            else:
                line += f" | {'—':>12}"
                row[cat] = None
        print(line)
        layer_heatmap.append(row)
    
    # ============================================================
    # Top 20 worst tensors
    # ============================================================
    print("\n" + "=" * 72)
    print("TOP 20 WORST TENSORS (by relative error)")
    print("=" * 72)
    
    sorted_by_error = sorted(all_tensors, key=lambda t: t["relative_error"], reverse=True)
    
    top_worst = []
    for i, t in enumerate(sorted_by_error[:20]):
        print(f"  {i+1:>2}. {t['relative_error']:.4%} | SNR {t['snr_db']:.1f} dB | "
              f"{t['numel']:>12,} params | {t['category']:>15} | {t['name']}")
        top_worst.append({
            "rank": i + 1,
            "name": t["name"],
            "category": t["category"],
            "layer": t["layer"],
            "relative_error": round(t["relative_error"], 6),
            "snr_db": round(t["snr_db"], 1),
            "numel": t["numel"],
            "dynamic_range": round(t["dynamic_range"], 6),
        })
    
    # ============================================================
    # Histogram data: distribution of relative errors
    # ============================================================
    all_rel_errors = [t["relative_error"] for t in all_tensors]
    all_snrs = [t["snr_db"] for t in all_tensors if t["snr_db"] < 900]
    
    # Binned histogram
    error_bins = np.histogram(all_rel_errors, bins=50)
    snr_bins = np.histogram(all_snrs, bins=50)
    
    histogram_data = {
        "relative_error": {
            "counts": error_bins[0].tolist(),
            "bin_edges": error_bins[1].tolist(),
        },
        "snr_db": {
            "counts": snr_bins[0].tolist(),
            "bin_edges": snr_bins[1].tolist(),
        }
    }
    
    # Per-category error distributions
    category_distributions = {}
    for cat in sorted(categories.keys()):
        errs = [t["relative_error"] for t in categories[cat]["tensors"]]
        snrs = [t["snr_db"] for t in categories[cat]["tensors"] if t["snr_db"] < 900]
        category_distributions[cat] = {
            "relative_errors": sorted(errs),
            "snr_values": sorted(snrs),
        }
    
    # ============================================================
    # Save results
    # ============================================================
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Full per-tensor JSON
    full_results = {
        "model": "openai/gpt-oss-20b",
        "architecture": "GptOssForCausalLM (MoE, 32 experts, 4 active)",
        "total_tensors": len(all_tensors),
        "category_summary": category_summary,
        "layer_heatmap": layer_heatmap,
        "top_20_worst": top_worst,
        "histogram_data": histogram_data,
        "category_distributions": category_distributions,
        "per_tensor": [
            {k: v for k, v in t.items() if k != "shape"}
            for t in all_tensors
        ],
    }
    
    json_path = os.path.join(results_dir, "per_tensor_error_breakdown.json")
    with open(json_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"\nFull results saved to: {json_path}")
    
    # CSV for easy spreadsheet analysis
    csv_path = os.path.join(results_dir, "per_tensor_errors.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "name", "category", "layer", "shard", "numel", "shape",
            "relative_error", "snr_db", "mean_abs_error", "max_abs_error",
            "p95_error", "p99_error", "dynamic_range", "w_min", "w_max", "w_mean", "w_std"
        ])
        for t in all_tensors:
            writer.writerow([
                t["name"], t["category"], t["layer"], t["shard"], t["numel"],
                str(t.get("shape", "")),
                f"{t['relative_error']:.8f}", f"{t['snr_db']:.2f}",
                f"{t['mean_abs_error']:.10f}", f"{t['max_abs_error']:.10f}",
                f"{t['p95_error']:.10f}", f"{t['p99_error']:.10f}",
                f"{t['dynamic_range']:.8f}",
                f"{t['w_min']:.8f}", f"{t['w_max']:.8f}",
                f"{t['w_mean']:.10f}", f"{t['w_std']:.10f}",
            ])
    print(f"CSV saved to: {csv_path}")
    
    return full_results


if __name__ == "__main__":
    results = run_analysis()
