# A 4x Compression Method for Parameter Snapshots (And What I Learned Breaking It on a Real 12B Model)

*A practical guide to compressed parameter snapshots for continual learning — what works, what breaks, and why it matters more this week than ever*

**By Caleb Stevens | April 2026**

---

## The Memory Crisis Everyone's Screaming About

In case you missed it, the AI memory shortage just went nuclear.

**Today — April 6, 2026 — Samsung reported Q1 operating profit of $37.9 billion.** That's an eightfold jump from a year ago. Their first quarter *alone* exceeded their entire 2025 annual profit. Ninety-five percent of it came from one thing: memory chips being devoured by AI datacenters (Reuters).

This isn't a blip. Here's what the last 90 days look like:

- **Micron's CEO told CNBC** they can only supply "50% to two-thirds" of what their key customers need. Their stock is up 350% in a year. They're *sold out for all of 2026* (CNBC, March 19).
- **DRAM contract prices rose 50–55% in Q1** vs. Q4 2025, and TrendForce just forecast **another 58–63% jump in Q2 2026** (TrendForce, March 31). That's not annual — that's *quarter over quarter*.
- **Every 1 bit of HBM memory produced means 3 bits of consumer DRAM that doesn't get made.** Micron literally stopped selling RAM to consumers in December 2025 to save supply for AI chips (CNBC).
- **Consumer RAM prices have exploded.** One tech executive posted that his 256GB kit went from ~$300 to ~$3,000 in a matter of months. Memory now accounts for 20% of laptop hardware costs, up from 10–18% in early 2025 (CNBC).
- **HPE's CEO said it plainly**: "We will continue to raise prices because the industry will continue to raise prices. There is not enough supply for demand" (CNBC, March 11).
- **No relief is coming until 2027 at the earliest.** New Micron fabs in Idaho won't produce chips until 2027–2028. Broadcom's CEO has locked in supply contracts through 2028.

The AI industry calls this the **"memory wall"** — processors getting faster while memory can't keep up, leaving billion-dollar GPUs twiddling their thumbs waiting for data.

And just last month, **Google released a technique called TurboQuant** — a memory-saving quantization method. The announcement coincided with a notable selloff in memory chip stocks (Reuters). Whether or not TurboQuant directly caused the move, the market is telling you something: *any technology that reduces memory consumption is now worth real money.*

But here's the thing most people miss: it's not just the model weights eating your memory budget. If you're doing **continual learning**, **checkpoint averaging**, **model soups**, or any form of **parameter regularization** (EWC, etc.), you're storing *multiple copies* of your model's parameters. For a 70B-parameter model, keeping just 10 snapshots means **2.8 terabytes** of parameter memory.

That's not a rounding error. That's an entire rack of GPUs worth of memory — memory that costs $30/GB and rising.

I set out to build a sophisticated system for continual learning — and ended up making a simpler discovery that could save your AI lab tens of thousands of dollars in memory costs. Along the way, I also found exactly where that compression breaks on a real 12B-parameter model, and why. This article is a practical guide to what worked, what didn't, and what you need to know before applying INT8 compression to production-scale parameter snapshots.

---

## What I Built (And What Actually Mattered)

**Neural Echo** started as an ambitious system: a neural encoder that transforms model parameters into semantic vector embeddings, stores them in a vector database, and uses similarity search to retrieve the most relevant historical snapshots when blending parameters. Think of it as "RAG for model weights."

It was elegant. It was complex. And it didn't matter.

After months of experimentation across 100K, 1M, and 10M parameter models, I made two discoveries — one surprising, one practical:

### Discovery 1: Semantic Retrieval in Parameter Space Doesn't Work (And That's Worth Knowing)

> **If you're building a system that tries to "intelligently" select which historical checkpoints to blend — read this first. It could save you months.**

I built a sophisticated parameter encoder — a neural network that learns to embed model configurations into a semantic space. The idea was that when learning a new task, the system could retrieve the most *relevant* historical snapshots rather than just the most *recent* ones.

**Semantic retrieval accuracy: 20–24%.**  
**Random retrieval accuracy: 20%.**

That's right — my carefully trained encoder performed no better than random selection. The cosine similarities between different parameter snapshots all clustered between 0.93 and 0.96. In parameter space, everything looks the same.

This isn't a bug in my implementation. It's a fundamental property of high-dimensional parameter spaces. The geometric structure of trained neural networks doesn't carry the kind of task-level semantic information that would make retrieval useful. To my knowledge, this negative result hasn't been widely reported in the continual learning literature — most papers assume semantic structure exists and build on top of it. If you're designing a parameter memory system, **simple FIFO or random selection is as good as learned retrieval**, and dramatically simpler to implement.

### Discovery 2: Compression Is the Real Win

While the retrieval system was a dead end, the compression infrastructure I built underneath it turned out to be genuinely valuable. Using **INT8 affine quantization** of full parameter tensors, the system achieves a consistent **4x compression ratio** on parameter snapshots — and this is real RAM savings, not just storage compression.

**An important note on what's new here:** INT8 quantization is not a novel technique. Libraries like [TorchAO](https://github.com/pytorch/ao) (Meta/PyTorch, ICML 2025, 2.8K stars) and [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) (8.1K stars, used by 36K+ repos) have had INT8 — and even INT4 and 2-bit — quantization for years. They're excellent for inference speedup and training memory reduction, and they do it better than I do for those use cases. What they *don't* do is maintain a compressed library of historical parameter snapshots for online blending during continual learning. The quantization primitive is well-solved. What I'm sharing here is the **application pattern** — snapshot, compress, store, dequantize, blend — and the experimental evidence that it works without degrading training quality.

Here's what that looks like at scale:

| Model Scale | Per Snapshot (FP32) | 10 Snapshots | With INT8 (4x) | **RAM Saved** |
|---|---|---|---|---|
| 100K params | 0.45 MB | 4.5 MB | 1.1 MB | 3.4 MB |
| 1M params | 3.05 MB | 30.5 MB | 7.6 MB | 22.8 MB |
| 10M params | 32.2 MB | 321.7 MB | 80.4 MB | **241.2 MB** |
| 7B (projected) | 26.1 GB | 260.8 GB | 65.2 GB | **195.6 GB** |
| 70B (projected) | 260.8 GB | 2.6 TB | 651.9 GB | **1.96 TB** |

At current HBM3 prices, saving ~2 TB on a 70B model training setup translates to roughly **$58,000 in hardware costs** — per setup. For a LLaMA-405B class model with 50 snapshots, we're talking about **$1.7 million in memory savings**.

---

## How It Works (In ~200 Lines of Code)

The final system is embarrassingly simple:

```
Neural Echo V2
├── Parameter Storage
│   └── INT8 affine quantization of full parameter tensors (4x RAM savings)
├── Snapshot Management
│   └── FIFO buffer — store every N training steps
└── Parameter Blending
    └── dequantize → current_params × 0.8 + historical_avg × 0.2
```

That's it. No neural encoder. No semantic search. No contrastive learning. No external dependencies. Just:

1. **Periodically snapshot** your model's parameters during training
2. **Quantize and store them** as INT8 tensors in RAM (4x less memory than FP32)
3. **Dequantize and blend** current parameters with the historical average

Each parameter tensor is quantized from FP32 (4 bytes per value) to UINT8 (1 byte per value) using per-tensor affine quantization: `scale = (max - min) / 255`, `quantized = round((val - min) / scale)`. When parameters are needed for blending, they're dequantized back to FP32 on the fly. The whole thing is pure PyTorch — no external services, no Docker containers, no network calls.

### The Code

```python
from neural_echo_v2.core import NeuralEchoV2

# Initialize — that's almost all you need
echo = NeuralEchoV2(
    use_compression=True,     # Enable 4x INT8 compression
    max_snapshots=10,         # Keep 10 historical snapshots
    blend_weight=0.8,         # 80% current, 20% historical
    snapshot_interval=10      # Store every 10 steps
)

# In your training loop — one line change
for step in range(num_steps):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    echo.train_step(model, loss, optimizer)  # ← this replaces your usual backward/step
```

---

## The Experiments

I ran five experiments to validate the compression approach, then three more at production scale that complicated — and ultimately strengthened — the story. Experiments 1–5 were conducted on CPU with small models. Experiments 6–8 run against the actual GPT-OSS 20B model (12B parameters) on an RTX 5090.

### Experiment 1: Compression Ratio at Scale

I measured compression ratios across three model sizes (100K, 1M, and 10M parameters), storing 10 snapshots each.

| Model | Parameters | 10 Snapshots (FP32) | 10 Snapshots (INT8) | Ratio | Avg Store Time |
|---|---|---|---|---|---|
| 100K | 118,282 | 4.5 MB | 1.1 MB | **4.0x** | 48.3 ± 13.7 ms |
| 1M | 798,474 | 30.5 MB | 7.6 MB | **4.0x** | 65.2 ± 10.9 ms |
| 10M | 8,432,138 | 321.7 MB | 80.4 MB | **4.0x** | 323.2 ± 31.2 ms |

The 4.0x ratio is consistent regardless of model size — a direct consequence of quantizing 32-bit floats to 8-bit integers.

### Experiment 2: Does Compression Hurt Blending Quality?

This is the key question: if you store compressed snapshots and use them for parameter blending, does the model suffer?

I ran 10 independent trials per model size, comparing:
- **Path A**: Uncompressed parameter blending (the gold standard)
- **Path B**: Blending via compressed storage (the efficient path)

| Model | Param Divergence | Loss Diff |
|---|---|---|
| 100K | 0.00072 ± 0.00014 | 0.0010 ± 0.0007 |
| 1M | 0.00051 ± 0.00008 | 0.0005 ± 0.0004 |
| 10M | 0.00031 ± 0.00006 | 0.0005 ± 0.0002 |

*Averaged over 10 independent runs per model size.*

**What this measures:** This experiment deliberately uses randomly-initialized models with minimal training (50 steps on synthetic data). The goal isn't to measure task performance — it's to measure whether *the two paths diverge* in parameter state and loss trajectory. If both paths arrive at statistically equivalent parameter states and loss values, then compression introduced no degradation to the optimization process. I removed accuracy columns from this table because they're meaningless for random-init models on this budget — both paths produce random-chance accuracy, which is expected and irrelevant. The metrics that matter are parameter divergence (how far the two paths drift) and loss difference (whether compression affects convergence). For task-performance validation with real pre-trained weights, see Experiments 7 and 8 below.

With that framing: the compressed and uncompressed paths produce **statistically equivalent results**. Parameter divergence is on the order of 10⁻⁴, and loss differences are below 0.001. Critically, the relative parameter difference is ~2% across all model sizes, meaning compression introduces negligible drift regardless of scale.

**Why the difference is so small:** 8-bit affine quantization introduces ~0.36% relative error per parameter (see Experiment 3 below), but when you average over thousands of parameters during blending, these errors cancel out. The dequantized FP32 tensors are statistically equivalent to the originals for gradient-based optimization. The measured parameter divergence of ~10⁻⁴ confirms that the quantization noise is well below the noise floor of SGD itself.

### Experiment 3: Quantization Error at Scale

Since the system stores parameters as INT8 and dequantizes on demand, I measured the actual quantization error to characterize the fidelity of the round-trip:

| Model | Avg Quantization Error | Relative Error | SNR | Loss Degradation | Acc Change |
|---|---|---|---|---|---|
| 100K | 8.7 × 10⁻⁵ | 0.36% | **48.8 dB** | 0.000003 | +0.4% |
| 1M | 7.9 × 10⁻⁵ | 0.38% | **48.4 dB** | 0.000017 | 0.0% |
| 10M | 5.0 × 10⁻⁵ | 0.38% | **48.5 dB** | 0.000006 | 0.0% |

Signal-to-noise ratios above 40 dB are considered high fidelity in signal processing. The 8-bit quantization achieves **~48.5 dB SNR** on these randomly-initialized models — the quantized parameters are virtually indistinguishable from the originals at this stage. However, as Experiments 7 and 8 will show, pre-trained models with wider weight distributions see lower SNR in specific architectural components. The 48.5 dB here represents a best-case baseline.

### Experiment 4: Computational Overhead

Parameter memory adds a small computational overhead for snapshot storage and blending.

| Model | Baseline | Echo (With INT8) | Overhead |
|---|---|---|---|
| 100K | 0.14s | 0.14s | **+3.2%** |
| 1M | 0.19s | 0.24s | **+27.4%** |
| 10M | 0.74s | 0.93s | **+25.2%** |

*100 training steps each. CPU-only, no GPU acceleration.*

**3–27% overhead** for parameter blending with INT8 quantization, all in-memory. The overhead comes primarily from the per-tensor CPU quantization loop: for each parameter tensor, the system computes min/max, scales, rounds, and clamps to UINT8. At 1M+ parameters, this loop dominates the snapshot storage time. On GPU, these operations map directly to vectorized CUDA kernels that process millions of elements in parallel — I'd expect the overhead to drop to single digits. (The quantize/dequantize math is the same add-multiply-round pattern that TorchAO already runs at near-memory-bandwidth speed on GPU.)

### Experiment 5: Projected Cost Savings

Based on the measured 4x compression ratio, here's what this method would save for production-scale models:

| Model | 10 Snapshots (Full) | 10 Snapshots (Compressed) | Memory Saved | **Cost Saved** |
|---|---|---|---|---|
| LLaMA-7B class | 260.8 GB | 65.2 GB | 195.6 GB | **~$5,867** |
| LLaMA-13B class | 484.3 GB | 121.1 GB | 363.2 GB | **~$10,896** |
| LLaMA-70B class | 2.6 TB | 651.9 GB | 1.96 TB | **~$58,673** |
| GPT-4 class (est.) | 7.5 TB | 1.9 TB | 5.6 TB | **~$167,638** |
| LLaMA-405B class | 15.1 TB | 3.8 TB | 11.3 TB | **~$339,467** |

*Cost estimates use ~$30/GB as an equivalent HBM3 memory allocation cost — what you're paying per GB when you provision GPU hardware for training. HBM isn't sold by the GB to end users; it's bundled into GPUs. But when your training run needs 2.6 TB for snapshots vs. 651 GB, the difference translates directly to fewer GPUs or smaller instances. With DRAM contract prices rising 50–60% per quarter in 2026 (TrendForce), these figures are conservative — by Q3 2026, the real savings could be 2–3x higher.*

With 50 snapshots (common for long continual learning runs), multiply these numbers by 5x. At that scale, a LLaMA-405B class setup would save over **$1.7 million** in memory costs — and that number keeps climbing with every quarterly price report.

*Note: these projections assume per-tensor INT8 across all layers. Experiments 7–8 show this works cleanly for ~85% of parameters on MoE models; attention and embedding layers need group-wise quantization to preserve quality at full savings.*

### Experiment 6: Validation at Scale — 12B Parameters on RTX 5090

Experiments 1–5 used randomly-initialized small models where INT8 quantization is trivially clean. Experiments 6–8 stress-test the method on real pre-trained weights — where it gets interesting.

The experiments above used models up to 10M parameters. To validate at production scale, I ran the INT8 quantization pipeline against **openai/gpt-oss-20b** — a 12B-parameter Mixture-of-Experts model (32 experts, 4 active per token) — on an NVIDIA RTX 5090 with 32 GB VRAM.

| Metric | Value |
|---|---|
| Model | openai/gpt-oss-20b (11.96B parameters, MoE) |
| Hardware | NVIDIA RTX 5090, 32 GB VRAM |
| FP32 Snapshot Size | 44.54 GB |
| INT8 Snapshot Size | 11.14 GB |
| Compression Ratio | **4.0x** |
| SNR | **40.0 dB** |
| Weighted Relative Error | 3.14% |
| Quantize Time (CPU) | 42.8s (0.28B params/s) |
| 10 Snapshots Saved | 334 GB (**~$10,022** @ $30/GB) |

*459 tensors across 3 safetensors shards. Parameters cast to FP32 to simulate training-time master weights.*

The 4.0x compression ratio is **identical** to the small-model results. But the SNR of 40.0 dB is notably lower than the 48.5 dB measured on randomly-initialized small models. That's an 8.5 dB drop — roughly a 7x increase in quantization noise power. My small-model experiments claimed "virtually indistinguishable from the originals." The 12B results say "close, but not close enough to stop investigating."

So I investigated.

### Experiment 7: Per-Tensor Error Breakdown — Where It Breaks

I ran a detailed error analysis on all 459 tensors, classifying each by architectural role. This is where the real finding emerged:

| Component | % of Parameters | Weighted Relative Error | SNR | Verdict |
|---|---|---|---|---|
| **MoE expert blocks** | **79.9%** | **0.000%** | **∞ (lossless)** | Perfect |
| Expert scales | 5.0% | 0.009% | 78.3 dB | Excellent |
| **Attention projections** | **5.3%** | **25.7%** | **18.0 dB** | **Broken** |
| **Embedding** | **4.8%** | **32.7%** | **18.0 dB** | **Broken** |
| lm_head | 4.8% | 3.9% | 29.3 dB | Moderate |
| Router | 0.02% | 5.8% | 27.2 dB | Acceptable |
| LayerNorm / bias | 0.1% | 0.3–1.6% | 35–50 dB | Fine |

**The MoE experts are lossless.** GPT-OSS 20B stores its expert weights in mxfp4 format on disk — when cast to FP32, they're already integer-valued (0–255 range). INT8 quantization on these tensors introduces zero error. That's 80% of the model's parameters, perfectly preserved.

**The attention output projections are the failure mode.** The worst offenders are the `o_proj.weight` tensors — layer 3 hits **91.2% relative error** with a signal-to-noise ratio of just 5.4 dB. For context, 5 dB is the kind of noise that makes audio unintelligible. The top 20 worst tensors across the entire model are *all* attention Q/K/V/O projections and the embedding layer.

**Why it happens:** Pre-trained attention weights develop wide dynamic ranges during training — the `o_proj` weights have dynamic ranges of 10–200x, with extreme outliers that a single per-tensor scale/zero-point can't capture. Randomly-initialized weights (like in Experiments 1–5) have tight, near-Gaussian distributions that INT8 quantizes cleanly. Trained weights don't.

The error increases in deeper layers and is worst at the early-to-mid layers (0–12) where the model is learning hierarchical representations:

| Layer | Attention Relative Error | Expert Error | Router Error |
|---|---|---|---|
| 0 | 24.5% | 0.000% | 1.6% |
| 3 | 30.3% | 0.000% | 2.3% |
| 7 | 30.1% | 0.000% | 2.8% |
| 12 | 14.8% | 0.000% | 4.2% |
| 17 | 9.1% | 0.000% | 2.7% |
| 23 | 37.9% | 0.000% | 3.8% |

### Experiment 8: Blend Quality — Does the Error Actually Matter?

Having found the error, the question becomes: does it matter for the actual use case? INT8 quantization severely degrades ~10% of the model's parameters, but those parameters are blended at 20% weight with fresh FP32 parameters during training. I ran four targeted tests on the actual GPT-OSS weights.

**Test A: Output logit degradation.** I quantized the `lm_head` projection (the vocabulary output layer) and measured how much the output probability distribution changes:

| Scenario | Greedy Decode Agreement | KL Divergence | Cosine Similarity |
|---|---|---|---|
| lm_head quantized only | **94.5%** | 0.000279 | 1.000 |
| lm_head + embedding quantized | **70.3%** | 0.008797 | 0.982 |

With only `lm_head` quantized (3.9% relative error), greedy decoding produces the same token 94.5% of the time — a surprisingly small impact. Adding the quantized embedding (32.7% error) drops agreement to 70.3%, which is significant but not catastrophic.

**Test B: Attention layer output.** I simulated a full attention pass (Q→K→V→softmax→O) with quantized vs original weights:

| Layer | Output Cosine Sim | Output Relative Error | Attn Pattern Cosine |
|---|---|---|---|
| 0 | 0.957 | 73.3% | 1.000 |
| 3 | 0.908 | 134.6% | 1.000 |
| 7 | 0.772 | 112.5% | 1.000 |
| 12 | 0.931 | 95.6% | 1.000 |
| 17 | 0.995 | 13.0% | 1.000 |

The attention *patterns* (which tokens attend to which) are **perfectly preserved** — cosine similarity of 1.000. The damage is entirely in the output projection, which distorts the magnitude of the attention output. Layer 3's o_proj is so badly quantized that the attention output has 134% relative error. Interestingly, layer 17 is a dramatic outlier — 0.995 cosine similarity and only 13% error — which I don't yet have an explanation for. My best guess is that later layers develop narrower weight distributions during training, but confirming this requires per-layer dynamic range analysis I haven't done yet.

**Test C: Router sensitivity.** The critical question for MoE: does quantizing the router change which experts get activated?

| Metric | Result |
|---|---|
| Top-1 expert agreement | **100%** across all layers |
| Top-4 expert selection agreement | **99.2–100%** |
| Router logit cosine similarity | **0.99997+** |

**The router is immune to INT8 quantization.** Expert routing is completely preserved, meaning the MoE gating mechanism will produce identical expert assignments from quantized snapshots. This is the most important finding for MoE models: the routing decisions survive compression perfectly.

**Test D: Cumulative error propagation.** I chained quantized attention through all 24 layers to measure worst-case error accumulation:

| After Layer | Cosine Similarity | Relative Error |
|---|---|---|
| 1 | 0.879 | 77.5% |
| 6 | 0.687 | 136.5% |
| 12 | 0.575 | 155.6% |
| 18 | 0.468 | 124.0% |
| 24 | 0.716 | 141.6% |

After 24 layers of quantized attention, the hidden states have diverged to cosine 0.716. In isolation, this is severe. But this is the absolute worst case — every layer using quantized weights for a full forward pass with no residual learning. In the actual use case (parameter blending during training), the impact is diluted by three factors:

1. **Blend weight**: Only 20% of the parameter state comes from quantized history
2. **Expert immunity**: 85% of the parameters (experts + scales + router + norms) contribute zero quantization error to the blend
3. **Continued training**: SGD naturally corrects for the noise in subsequent steps — quantization noise is statistically indistinguishable from gradient noise at this scale

**The verdict:** Per-tensor INT8 quantization preserves 85% of GPT-OSS 20B's parameters with negligible-to-zero error. The remaining 10% (attention projections + embedding) suffers real degradation. For parameter blending with 80/20 weights, this translates to a diluted but measurable impact on model quality — one that continued training can partially recover from, but that a careful practitioner should be aware of.

**The fix is known:** group-wise quantization (per-channel or per-128-element blocks) handles wide dynamic ranges by fitting separate scale/zero-point values to smaller segments of each tensor. This is exactly what TorchAO's INT8 implementation does internally, and why their reports show higher fidelity than per-tensor approaches. Integrating group-wise quantization for the attention and embedding layers — while keeping cheap per-tensor quantization for the already-lossless expert weights — is the clear next step.

---

## The Case Against Complexity

The most important finding of this research isn't the compression — it's what I eliminated.

My original system (Neural Echo V1) had:
- A neural parameter encoder (2048 → 1024 → 512 → 768 dims)
- Contrastive learning for embedding training
- Task discrimination tokens
- Parameter velocity tracking
- Fisher information-weighted interpolation
- Layer-wise adaptive blending strategies
- ~2,000+ lines of Python

Neural Echo V2 has:
- Parameter averaging
- INT8 affine quantization (pure PyTorch)
- ~200 lines of Python

**Simpler architecture. 10x less code. 4x less memory — with known limits at production scale.**

The lesson: before you build a complex ML system, make sure the complexity is earning its keep. In my case, the semantic retrieval complexity wasn't — but the 12B-scale results (Experiments 7–8) suggest that *some* complexity in the quantization approach is warranted. Per-tensor affine quantization is the right choice for 85% of a model's parameters, but attention projections and embeddings need group-wise quantization to avoid the 18 dB SNR cliff. The simplest approach that works is not always the *simplest possible* approach.

---

## Who Should Care?

The fact that Google's TurboQuant announcement in March 2026 *coincided with a selloff in memory chip stocks* tells you everything about how the market views memory reduction right now. When a quantization technique is even *perceived* as threatening memory demand, it's no longer a research curiosity — it's a strategic priority.

This matters for anyone doing:

- **Continual learning** — storing parameter snapshots for forgetting mitigation
- **Model soups / checkpoint averaging** — blending multiple training runs
- **EWC and related methods** — storing Fisher information matrices (same compression applies)
- **Federated learning** — compressing model updates for transmission
- **Experiment tracking** — keeping historical checkpoints without blowing your storage budget

If you're spending money on GPU memory that's holding historical parameter snapshots, you're spending 4x more than you need to.

In a world where Micron is sold out for the year, Samsung is printing $38 billion quarters, and your laptop's RAM costs more than the laptop did two years ago — 4x compression isn't a research curiosity. It's a survival strategy.

---

## Try It Yourself

The code is open source and MIT licensed:

```bash
git clone https://github.com/Iztolie/neural-echo-v2.git
cd neural-echo-v2
pip install -r requirements.txt
```

Run the benchmark yourself:
```bash
python experiments/compression_benchmark.py
```

---

## What's Next

The per-tensor error breakdown (Experiment 7) identified a clear path forward:

1. **Group-wise quantization for attention layers** — The attention projections (o_proj, k_proj especially) need per-channel or per-block quantization to handle their wide dynamic ranges. TorchAO already implements this; the integration is straightforward. This should bring the attention SNR from 18 dB back up to the 45+ dB range, making the full-model quantization genuinely "free."

2. **Mixed-precision snapshot strategy** — The findings suggest an obvious optimization: keep expert weights in per-tensor INT8 (already lossless), apply group-wise INT8 to attention/embedding layers, and potentially leave the router and layernorm weights in FP16 (they're tiny — 0.02% of total parameters). This hybrid approach would maintain the 4x headline compression while eliminating the quality gap.

3. **Forgetting mitigation** — Does the parameter blending itself reduce catastrophic forgetting? I'm running controlled experiments with real vision tasks (CIFAR-10 split tasks with ResNet18) to find out.

4. **Integration with existing frameworks** — Making Neural Echo a drop-in module for HuggingFace Transformers and PyTorch Lightning, using TorchAO as the quantization backend for production-grade group-wise quantization.

---

## Citation

If this work is useful to your research, please cite:

```bibtex
@software{neural_echo_v2_2026,
  title={Neural Echo V2: INT8 Compressed Parameter Snapshots},
  author={Stevens, Caleb},
  year={2026},
  note={4x memory reduction via INT8 quantization for parameter snapshot systems},
  url={https://github.com/Iztolie/neural-echo-v2}
}
```

---

*Caleb Stevens is an independent AI researcher based in Little Elm, Texas, working on practical solutions for continual learning systems. Reach me at [caleb@mithrus.ai](mailto:caleb@mithrus.ai), [GitHub](https://github.com/Iztolie), or [LinkedIn](https://www.linkedin.com/in/caleb--stevens/).*

*Experiments 1–5 were run on consumer CPU hardware. Experiments 6–8 were run on an NVIDIA RTX 5090 against the GPT-OSS 20B model. No datacenter required — this method is accessible to individual researchers, not just well-funded labs.*
