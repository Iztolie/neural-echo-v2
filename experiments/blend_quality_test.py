"""
Blend Quality Test for GPT-OSS 20B
====================================
Tests whether INT8 quantization→dequantization of model weights
actually affects the model's output quality.

Approach:
1. Load model weights (the tensors we can access)
2. Simulate a forward pass through key components
3. Compare original vs quantized-then-dequantized outputs
4. Measure: KL divergence, top-k agreement, cosine similarity
"""

import sys
import os
import time
import json
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np


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

# Model config
HIDDEN_SIZE = 2880
NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 64
VOCAB_SIZE = 201088
INTERMEDIATE_SIZE = 2880
NUM_EXPERTS = 32
EXPERTS_PER_TOKEN = 4


def get_all_blob_paths():
    """Get all safetensors blob paths sorted by size."""
    blobs_dir = os.path.join(MODEL_DIR, "blobs")
    blob_files = []
    for f in os.listdir(blobs_dir):
        fp = os.path.join(blobs_dir, f)
        size = os.path.getsize(fp)
        if size > 1024**3:
            blob_files.append(fp)
    return blob_files


def load_tensors_by_keys(needed_keys):
    """Load specific tensor keys from whichever blob contains them."""
    from safetensors import safe_open
    blobs = get_all_blob_paths()
    result = {}
    remaining = set(needed_keys)
    
    for blob_path in blobs:
        if not remaining:
            break
        with safe_open(blob_path, framework="pt") as f:
            available = set(f.keys())
            found = remaining & available
            if found:
                for key in found:
                    result[key] = f.get_tensor(key)
                remaining -= found
    
    if remaining:
        print(f"  WARNING: Could not find keys: {remaining}")
    return result


def load_shard_by_layer(target_layers):
    """Load all tensors for specific layers from the correct blobs."""
    from safetensors import safe_open
    blobs = get_all_blob_paths()
    result = {}
    
    # Build prefix patterns for target layers
    prefixes = [f"model.layers.{li}." for li in target_layers]
    
    for blob_path in blobs:
        with safe_open(blob_path, framework="pt") as f:
            for key in f.keys():
                if any(key.startswith(p) for p in prefixes):
                    result[key] = f.get_tensor(key)
    
    return result


def quantize_dequant(tensor):
    """INT8 quantize then dequantize a tensor (round-trip)."""
    t = tensor.detach().float()
    min_val = t.min().item()
    max_val = t.max().item()
    scale = (max_val - min_val) / 255.0
    if scale == 0.0:
        return t.clone()
    quantized = ((t - min_val) / scale).round().clamp(0, 255).to(torch.uint8)
    restored = quantized.float() * scale + min_val
    return restored


def kl_divergence(logits_orig, logits_quant, temperature=1.0):
    """Compute KL divergence between two logit distributions."""
    p = torch.softmax(logits_orig / temperature, dim=-1)
    q = torch.softmax(logits_quant / temperature, dim=-1)
    # KL(P || Q) = sum(P * log(P/Q))
    kl = (p * (torch.log(p + 1e-10) - torch.log(q + 1e-10))).sum(dim=-1)
    return kl


def top_k_agreement(logits_orig, logits_quant, k=10):
    """Compute top-k token agreement between original and quantized logits."""
    top_orig = torch.topk(logits_orig, k, dim=-1).indices
    top_quant = torch.topk(logits_quant, k, dim=-1).indices
    
    # For each position, count how many of the top-k tokens match
    agreements = []
    for i in range(logits_orig.shape[0]):
        orig_set = set(top_orig[i].tolist())
        quant_set = set(top_quant[i].tolist())
        agreements.append(len(orig_set & quant_set) / k)
    return np.mean(agreements)


def cosine_sim(a, b):
    """Cosine similarity between two tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return torch.nn.functional.cosine_similarity(
        a_flat.unsqueeze(0), b_flat.unsqueeze(0)
    ).item()


def run_blend_quality_test():
    """
    Blend quality test: simulate forward pass through key layers,
    comparing original vs quantized-then-dequantized weights.
    """
    print("=" * 72)
    print("NEURAL ECHO V2 — BLEND QUALITY TEST")
    print(f"Model: openai/gpt-oss-20b (MoE, 32 experts, 4 active)")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 72)
    
    results = {
        "model": "openai/gpt-oss-20b",
        "tests": [],
    }
    
    # ============================================================
    # Test 1: lm_head logit degradation
    # ============================================================
    print("\n--- Test 1: lm_head Logit Degradation ---")
    print("  Simulating: hidden_state → lm_head → logits")
    print("  This directly measures output probability changes.")
    
    # Load lm_head and embed_tokens from wherever they are
    needed = ["lm_head.weight", "model.embed_tokens.weight"]
    shard = load_tensors_by_keys(needed)
    
    lm_head_weight = shard["lm_head.weight"].float()
    embed_weight = shard["model.embed_tokens.weight"].float()
    print(f"  lm_head shape: {list(lm_head_weight.shape)}")
    print(f"  embed_tokens shape: {list(embed_weight.shape)}")
    
    # Create realistic hidden states by sampling random embeddings
    # (simulates the output of the last transformer layer)
    torch.manual_seed(42)
    seq_len = 128
    # Use actual embedding vectors as "hidden states" — more realistic than random
    random_token_ids = torch.randint(0, VOCAB_SIZE, (seq_len,))
    hidden_states = embed_weight[random_token_ids].detach()  # [seq_len, hidden_size]
    
    # Add some noise to simulate transformer processing
    hidden_states = hidden_states + torch.randn_like(hidden_states) * hidden_states.std() * 0.1
    
    # Original logits
    logits_orig = hidden_states @ lm_head_weight.T  # [seq_len, vocab_size]
    
    # Quantized-then-dequantized lm_head
    lm_head_quant = quantize_dequant(lm_head_weight)
    logits_quant = hidden_states @ lm_head_quant.T
    
    # Also test with quantized embeddings (simulating a stored snapshot)
    embed_quant = quantize_dequant(embed_weight)
    hidden_states_quant = embed_quant[random_token_ids].detach()
    hidden_states_quant = hidden_states_quant + torch.randn_like(hidden_states_quant) * hidden_states_quant.std() * 0.1
    torch.manual_seed(42)  # Reset seed for consistent noise
    logits_both_quant = hidden_states_quant @ lm_head_quant.T
    
    # Metrics
    kl_lm_head = kl_divergence(logits_orig, logits_quant).mean().item()
    kl_both = kl_divergence(logits_orig, logits_both_quant).mean().item()
    
    topk_1_lm = top_k_agreement(logits_orig, logits_quant, k=1)
    topk_5_lm = top_k_agreement(logits_orig, logits_quant, k=5)
    topk_10_lm = top_k_agreement(logits_orig, logits_quant, k=10)
    
    topk_1_both = top_k_agreement(logits_orig, logits_both_quant, k=1)
    topk_5_both = top_k_agreement(logits_orig, logits_both_quant, k=5)
    topk_10_both = top_k_agreement(logits_orig, logits_both_quant, k=10)
    
    cos_logits_lm = cosine_sim(logits_orig, logits_quant)
    cos_logits_both = cosine_sim(logits_orig, logits_both_quant)
    
    # Argmax agreement (greedy decoding)
    argmax_orig = logits_orig.argmax(dim=-1)
    argmax_lm = logits_quant.argmax(dim=-1)
    argmax_both = logits_both_quant.argmax(dim=-1)
    greedy_agreement_lm = (argmax_orig == argmax_lm).float().mean().item()
    greedy_agreement_both = (argmax_orig == argmax_both).float().mean().item()
    
    test1 = {
        "test": "lm_head_logit_degradation",
        "description": "Output logit quality after quantizing lm_head and/or embeddings",
        "lm_head_only": {
            "kl_divergence": round(kl_lm_head, 6),
            "top1_agreement": round(topk_1_lm, 4),
            "top5_agreement": round(topk_5_lm, 4),
            "top10_agreement": round(topk_10_lm, 4),
            "greedy_agreement": round(greedy_agreement_lm, 4),
            "cosine_similarity": round(cos_logits_lm, 6),
        },
        "lm_head_and_embedding": {
            "kl_divergence": round(kl_both, 6),
            "top1_agreement": round(topk_1_both, 4),
            "top5_agreement": round(topk_5_both, 4),
            "top10_agreement": round(topk_10_both, 4),
            "greedy_agreement": round(greedy_agreement_both, 4),
            "cosine_similarity": round(cos_logits_both, 6),
        }
    }
    results["tests"].append(test1)
    
    print(f"\n  lm_head quantized only:")
    print(f"    KL divergence:     {kl_lm_head:.6f}")
    print(f"    Top-1 agreement:   {topk_1_lm:.1%}")
    print(f"    Top-5 agreement:   {topk_5_lm:.1%}")
    print(f"    Top-10 agreement:  {topk_10_lm:.1%}")
    print(f"    Greedy agreement:  {greedy_agreement_lm:.1%}")
    print(f"    Cosine similarity: {cos_logits_lm:.6f}")
    
    print(f"\n  lm_head + embedding quantized:")
    print(f"    KL divergence:     {kl_both:.6f}")
    print(f"    Top-1 agreement:   {topk_1_both:.1%}")
    print(f"    Top-5 agreement:   {topk_5_both:.1%}")
    print(f"    Top-10 agreement:  {topk_10_both:.1%}")
    print(f"    Greedy agreement:  {greedy_agreement_both:.1%}")
    print(f"    Cosine similarity: {cos_logits_both:.6f}")
    
    del lm_head_weight, lm_head_quant, embed_weight, embed_quant
    del logits_orig, logits_quant, logits_both_quant
    del hidden_states, hidden_states_quant
    
    # ============================================================
    # Test 2: Attention layer output degradation
    # ============================================================
    print("\n--- Test 2: Attention Layer Output Degradation ---")
    print("  Simulating: hidden_state → Q/K/V proj → attention → O proj")
    print("  Tests the worst-offending layers (o_proj at 91% rel error)")
    
    # Load attention weights for test layers
    del shard
    gc.collect()
    
    layer_tests = []
    test_layers = [0, 3, 7, 12, 17]  # Sample across early/mid/late layers
    shard = load_shard_by_layer(test_layers)
    
    for li in test_layers:
        prefix = f"model.layers.{li}.self_attn"
        q_w = shard[f"{prefix}.q_proj.weight"].float()
        k_w = shard[f"{prefix}.k_proj.weight"].float()
        v_w = shard[f"{prefix}.v_proj.weight"].float()
        o_w = shard[f"{prefix}.o_proj.weight"].float()
        q_b = shard[f"{prefix}.q_proj.bias"].float()
        k_b = shard[f"{prefix}.k_proj.bias"].float()
        v_b = shard[f"{prefix}.v_proj.bias"].float()
        o_b = shard[f"{prefix}.o_proj.bias"].float()
        
        # Simulate a batch of hidden states
        torch.manual_seed(42 + li)
        batch_size = 8
        seq_len = 64
        x = torch.randn(batch_size, seq_len, HIDDEN_SIZE) * 0.02  # typical hidden state scale
        
        # Original attention output  
        q = x @ q_w.T + q_b
        k = x @ k_w.T + k_b
        v = x @ v_w.T + v_b
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)  # [B, H, S, D]
        k = k.view(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        
        # GQA: repeat K/V for grouped query attention
        n_rep = NUM_HEADS // NUM_KV_HEADS
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (HEAD_DIM ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output_orig = attn_output @ o_w.T + o_b
        
        # Now with quantized-then-dequantized weights
        q_w_q = quantize_dequant(q_w)
        k_w_q = quantize_dequant(k_w)
        v_w_q = quantize_dequant(v_w)
        o_w_q = quantize_dequant(o_w)
        q_b_q = quantize_dequant(q_b)
        k_b_q = quantize_dequant(k_b)
        v_b_q = quantize_dequant(v_b)
        o_b_q = quantize_dequant(o_b)
        
        q2 = x @ q_w_q.T + q_b_q
        k2 = x @ k_w_q.T + k_b_q
        v2 = x @ v_w_q.T + v_b_q
        
        q2 = q2.view(batch_size, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k2 = k2.view(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v2 = v2.view(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        
        k2 = k2.repeat_interleave(n_rep, dim=1)
        v2 = v2.repeat_interleave(n_rep, dim=1)
        
        attn_weights2 = torch.matmul(q2, k2.transpose(-2, -1)) / (HEAD_DIM ** 0.5)
        attn_weights2 = torch.softmax(attn_weights2, dim=-1)
        attn_output2 = torch.matmul(attn_weights2, v2)
        attn_output2 = attn_output2.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output_quant = attn_output2 @ o_w_q.T + o_b_q
        
        # Compute residual: in a real transformer, output = x + attn_output
        residual_orig = x + output_orig
        residual_quant = x + output_quant
        
        # Metrics
        cos = cosine_sim(output_orig, output_quant)
        cos_residual = cosine_sim(residual_orig, residual_quant)
        rel_err_output = (output_orig - output_quant).abs().mean().item() / output_orig.abs().mean().item()
        rel_err_residual = (residual_orig - residual_quant).abs().mean().item() / residual_orig.abs().mean().item()
        
        # Attention pattern agreement
        attn_cos = cosine_sim(attn_weights, attn_weights2)
        
        layer_result = {
            "layer": li,
            "output_cosine_sim": round(cos, 6),
            "output_relative_error": round(rel_err_output, 6),
            "residual_cosine_sim": round(cos_residual, 6),
            "residual_relative_error": round(rel_err_residual, 6),
            "attention_pattern_cosine": round(attn_cos, 6),
        }
        layer_tests.append(layer_result)
        
        print(f"\n  Layer {li}:")
        print(f"    Attn output cosine sim:     {cos:.6f} | rel error: {rel_err_output:.4%}")
        print(f"    Residual cosine sim:         {cos_residual:.6f} | rel error: {rel_err_residual:.4%}")
        print(f"    Attention pattern cosine:    {attn_cos:.6f}")
        
        del q_w, k_w, v_w, o_w, q_b, k_b, v_b, o_b
        del q_w_q, k_w_q, v_w_q, o_w_q, q_b_q, k_b_q, v_b_q, o_b_q
        del q, k, v, q2, k2, v2
        del attn_weights, attn_weights2, attn_output, attn_output2
        del output_orig, output_quant, residual_orig, residual_quant
        gc.collect()
    
    test2 = {
        "test": "attention_layer_degradation",
        "description": "Attention output quality comparing original vs quantized Q/K/V/O projections",
        "layers": layer_tests,
    }
    results["tests"].append(test2)
    
    del shard
    gc.collect()
    
    # ============================================================
    # Test 3: Router sensitivity test
    # ============================================================
    print("\n--- Test 3: Router Sensitivity ---")
    print("  Tests whether quantized router weights change expert selection.")
    
    router_layer_ids = [0, 5, 10, 15, 17]
    router_keys = []
    for li in router_layer_ids:
        router_keys.append(f"model.layers.{li}.mlp.router.weight")
        router_keys.append(f"model.layers.{li}.mlp.router.bias")
    shard = load_tensors_by_keys(router_keys)
    
    router_tests = []
    for li in router_layer_ids:
        prefix = f"model.layers.{li}.mlp.router"
        if f"{prefix}.weight" not in shard:
            continue
            
        r_w = shard[f"{prefix}.weight"].float()
        r_b = shard[f"{prefix}.bias"].float()
        
        # Simulate hidden states going to router
        torch.manual_seed(42 + li)
        x = torch.randn(32, HIDDEN_SIZE) * 0.02  # 32 token positions
        
        # Original routing
        logits_orig = x @ r_w.T + r_b
        top_orig = torch.topk(logits_orig, EXPERTS_PER_TOKEN, dim=-1).indices
        
        # Quantized routing
        r_w_q = quantize_dequant(r_w)
        r_b_q = quantize_dequant(r_b)
        logits_quant = x @ r_w_q.T + r_b_q
        top_quant = torch.topk(logits_quant, EXPERTS_PER_TOKEN, dim=-1).indices
        
        # Expert selection agreement
        agreement = 0
        for i in range(32):
            orig_set = set(top_orig[i].tolist())
            quant_set = set(top_quant[i].tolist())
            agreement += len(orig_set & quant_set) / EXPERTS_PER_TOKEN
        agreement /= 32
        
        # Logit cosine similarity
        router_cos = cosine_sim(logits_orig, logits_quant)
        
        # Top-1 expert agreement 
        top1_agree = (top_orig[:, 0] == top_quant[:, 0]).float().mean().item()
        
        router_result = {
            "layer": li,
            "expert_selection_agreement": round(agreement, 4),
            "top1_expert_agreement": round(top1_agree, 4),
            "router_logit_cosine_sim": round(router_cos, 6),
        }
        router_tests.append(router_result)
        
        print(f"\n  Layer {li} Router:")
        print(f"    Expert selection agreement (top-4): {agreement:.1%}")
        print(f"    Top-1 expert agreement:             {top1_agree:.1%}")
        print(f"    Router logit cosine sim:            {router_cos:.6f}")
        
        del r_w, r_b, r_w_q, r_b_q
        gc.collect()
    
    test3 = {
        "test": "router_sensitivity",
        "description": "MoE router: does quantization change which experts are selected?",
        "layers": router_tests,
    }
    results["tests"].append(test3)
    
    del shard
    gc.collect()
    
    # ============================================================
    # Test 4: Cumulative error propagation (multi-layer simulation)
    # ============================================================
    print("\n--- Test 4: Cumulative Error Propagation ---")
    print("  Simulating error accumulation across multiple attention layers")
    print("  (residual connections + quantized weights)")
    
    # Load all attention layers 0-23
    all_layer_ids = list(range(24))
    shard = load_shard_by_layer(all_layer_ids)
    
    torch.manual_seed(42)
    batch_size = 4
    seq_len = 32
    x_orig = torch.randn(batch_size, seq_len, HIDDEN_SIZE) * 0.02
    x_quant = x_orig.clone()
    
    propagation_data = []
    
    for li in range(24):  # All 24 layers
        prefix = f"model.layers.{li}.self_attn"
        ln_key = f"model.layers.{li}.input_layernorm.weight"
        
        if f"{prefix}.q_proj.weight" not in shard:
            break
        
        # Get weights
        q_w = shard[f"{prefix}.q_proj.weight"].float()
        k_w = shard[f"{prefix}.k_proj.weight"].float()
        v_w = shard[f"{prefix}.v_proj.weight"].float()
        o_w = shard[f"{prefix}.o_proj.weight"].float()
        q_b = shard[f"{prefix}.q_proj.bias"].float()
        k_b = shard[f"{prefix}.k_proj.bias"].float()
        v_b = shard[f"{prefix}.v_proj.bias"].float()
        o_b = shard[f"{prefix}.o_proj.bias"].float()
        
        # Quantized weights
        q_w_q = quantize_dequant(q_w)
        k_w_q = quantize_dequant(k_w)
        v_w_q = quantize_dequant(v_w)
        o_w_q = quantize_dequant(o_w)
        q_b_q = quantize_dequant(q_b)
        k_b_q = quantize_dequant(k_b)
        v_b_q = quantize_dequant(v_b)
        o_b_q = quantize_dequant(o_b)
        
        # Simple RMSNorm simulation
        def rms_norm(x):
            return x / (x.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-5)
        
        # Original path
        h = rms_norm(x_orig)
        q = (h @ q_w.T + q_b).view(batch_size, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = (h @ k_w.T + k_b).view(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = (h @ v_w.T + v_b).view(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        n_rep = NUM_HEADS // NUM_KV_HEADS
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
        aw = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (HEAD_DIM ** 0.5), dim=-1)
        ao = torch.matmul(aw, v).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        x_orig = x_orig + (ao @ o_w.T + o_b)
        
        # Quantized path
        h2 = rms_norm(x_quant)
        q2 = (h2 @ q_w_q.T + q_b_q).view(batch_size, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k2 = (h2 @ k_w_q.T + k_b_q).view(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v2 = (h2 @ v_w_q.T + v_b_q).view(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        k2 = k2.repeat_interleave(n_rep, dim=1)
        v2 = v2.repeat_interleave(n_rep, dim=1)
        aw2 = torch.softmax(torch.matmul(q2, k2.transpose(-2, -1)) / (HEAD_DIM ** 0.5), dim=-1)
        ao2 = torch.matmul(aw2, v2).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        x_quant = x_quant + (ao2 @ o_w_q.T + o_b_q)
        
        # Measure divergence
        cos = cosine_sim(x_orig, x_quant)
        rel_err = (x_orig - x_quant).abs().mean().item() / x_orig.abs().mean().item()
        
        propagation_data.append({
            "layer": li,
            "cumulative_cosine_sim": round(cos, 6),
            "cumulative_relative_error": round(rel_err, 6),
        })
        
        print(f"  After layer {li:>2}: cosine={cos:.6f} | rel_error={rel_err:.4%}")
        
        del q_w, k_w, v_w, o_w, q_b, k_b, v_b, o_b
        del q_w_q, k_w_q, v_w_q, o_w_q, q_b_q, k_b_q, v_b_q, o_b_q
        del q, k, v, q2, k2, v2, aw, aw2, ao, ao2, h, h2
        gc.collect()
    
    test4 = {
        "test": "cumulative_error_propagation",
        "description": "How quantization error accumulates through 18 attention layers with residual connections",
        "layers": propagation_data,
    }
    results["tests"].append(test4)
    
    del shard
    gc.collect()
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 72)
    print("BLEND QUALITY SUMMARY")
    print("=" * 72)
    
    final_layer = propagation_data[-1] if propagation_data else {}
    
    print(f"\n  lm_head → logit degradation:")
    t1_lm = test1["lm_head_only"]
    t1_both = test1["lm_head_and_embedding"]
    print(f"    Greedy decode agreement (lm_head only):  {t1_lm['greedy_agreement']:.1%}")
    print(f"    Greedy decode agreement (+ embedding):   {t1_both['greedy_agreement']:.1%}")
    print(f"    KL divergence (lm_head only):            {t1_lm['kl_divergence']:.6f}")
    print(f"    KL divergence (+ embedding):             {t1_both['kl_divergence']:.6f}")
    
    print(f"\n  Attention layers (average across tested):")
    avg_residual_cos = np.mean([l["residual_cosine_sim"] for l in layer_tests])
    avg_residual_err = np.mean([l["residual_relative_error"] for l in layer_tests])
    print(f"    Avg residual cosine sim:    {avg_residual_cos:.6f}")
    print(f"    Avg residual relative err:  {avg_residual_err:.4%}")
    
    print(f"\n  Router sensitivity (average):")
    avg_expert_agree = np.mean([r["expert_selection_agreement"] for r in router_tests])
    avg_top1_agree = np.mean([r["top1_expert_agreement"] for r in router_tests])
    print(f"    Expert selection agreement:  {avg_expert_agree:.1%}")
    print(f"    Top-1 expert agreement:      {avg_top1_agree:.1%}")
    
    print(f"\n  Cumulative propagation (after 24 layers):")
    if final_layer:
        print(f"    Final cosine sim:           {final_layer['cumulative_cosine_sim']:.6f}")
        print(f"    Final relative error:       {final_layer['cumulative_relative_error']:.4%}")
    
    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    results_path = os.path.join(results_dir, "blend_quality_test.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    results = run_blend_quality_test()
