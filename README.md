# Neural Echo V2: INT8 Compressed Parameter Snapshots

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status: Beta](https://img.shields.io/badge/Status-Beta-yellow.svg)](#)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**Author:** Caleb Stevens  
**Status:** Beta — Experimental Research  
**Article:** [A 4x Compression Method for Parameter Snapshots (And What I Learned Breaking It on a Real 12B Model)](https://medium.com/@calebstevens)

> **Beta Notice:** This is pre-release research code published alongside the article above. The core compression and blending work as described. GPU experiments (Experiments 6–8) require an RTX 5090 and GPT-OSS 20B weights. Expect rough edges.

---

## What Is This?

Neural Echo V2 compresses neural network parameter snapshots to **25% of their original size** using per-tensor INT8 affine quantization. Pure PyTorch, no external services, ~200 lines of core code.

During training, it periodically stores compressed snapshots and blends historical parameters with current ones — a form of temporal regularization for continual learning.

### Key Results

| Claim | Evidence |
|-------|----------|
| **4.0x compression** (FP32 → UINT8) | Consistent across 100K–11.96B parameters |
| **~48.5 dB SNR** reconstruction quality | Measured across 5 independent runs at 100K–10M scale |
| **MoE experts compress losslessly** | 79.9% of 12B model parameters, <0.001% relative error |
| **Attention projections break at INT8** | 5.3% of parameters, 25.7% relative error |

### The Honest Version

This works great for storing snapshots. The compression is real and the quality impact is negligible for most layers. But at 12B scale, we discovered that attention projection matrices are pathologically sensitive to quantization — a finding that matters for anyone doing checkpoint compression on large models.

---

## Quick Start

```python
from core import NeuralEchoV2

import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

echo = NeuralEchoV2(
    use_compression=True,    # Enable INT8 compression
    max_snapshots=10,        # FIFO buffer of 10 snapshots
    blend_weight=0.8,        # 80% current, 20% historical
    snapshot_interval=10     # Store every 10 steps
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(1000):
    x = torch.randn(32, 784)
    y = torch.randint(0, 10, (32,))
    loss = nn.functional.cross_entropy(model(x), y)

    echo.train_step(model, loss, optimizer)

    if step % 100 == 0:
        stats = echo.get_stats()
        print(f"Step {step}: {stats['num_snapshots']} snapshots, "
              f"saved {stats['memory_saved_mb']:.1f}MB")
```

---

## Installation

```bash
git clone https://github.com/Iztolie/neural-echo-v2.git
cd neural-echo-v2
pip install -r requirements.txt
```

No Docker, no external services, no vector databases. Just PyTorch.

---

## Project Structure

```
neural-echo-v2/
├── core/
│   ├── __init__.py                # Exports NeuralEchoV2, QuantizedSnapshot
│   ├── echo_memory.py             # Main class — snapshot storage + blending
│   └── quantization.py            # INT8 per-tensor affine quantization
├── experiments/
│   ├── compression_benchmark.py   # Experiments 1–5: compression, SNR, scaling
│   ├── gpu_benchmark_20b.py       # Experiment 6: GPT-OSS 20B on RTX 5090
│   ├── per_tensor_breakdown.py    # Experiment 7: per-layer error analysis
│   ├── blend_quality_test.py      # Experiment 8: blend quality at 12B scale
│   ├── scaling_test.py            # 100K / 1M / 10M parameter sweep
│   ├── validate_100k.py           # 100K parameter validation suite (5 tests)
│   └── test_real_models.py        # ResNet18 on CIFAR-10/100
├── analysis/
│   └── plot_results.py            # Generate matplotlib figures from results
├── paper/
│   ├── medium_article.md          # Full article text
│   ├── article_visuals.html       # Interactive 8-chart visual report
│   ├── results/                   # Raw experiment outputs (JSON)
│   └── figures/                   # Generated plots (PNG)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## How It Works

### Compression — QuantizedSnapshot

Per-tensor affine quantization: for each parameter tensor, compute min/max, scale to \[0, 255\], store as `uint8` + two float scalars (scale, zero_point).

```
compressed_size  = n_elements × 1 byte  + 2 floats
uncompressed     = n_elements × 4 bytes
ratio            ≈ 4.0x
```

### Memory — NeuralEchoV2

A FIFO `deque` of compressed snapshots. Every `snapshot_interval` training steps:

1. Clone current model parameters
2. Compress to INT8 via `QuantizedSnapshot` (if `use_compression=True`)
3. Push to deque (oldest snapshot evicted when buffer is full)
4. Blend current parameters with historical mean:  
   `params = blend_weight × current + (1 - blend_weight) × mean(historical)`

---

## Experiments

### CPU experiments (any machine)

```bash
# Core benchmarks — Experiments 1–5: compression ratio, SNR, scaling
python experiments/compression_benchmark.py

# Validation suite — 5 tests at 100K parameters
python experiments/validate_100k.py

# Scaling sweep — 100K / 1M / 10M
python experiments/scaling_test.py

# Real model — ResNet18 on CIFAR-10/100 (downloads ~500MB)
python experiments/test_real_models.py
```

### GPU experiments (RTX 5090 + GPT-OSS 20B weights required)

```bash
# Experiment 6: Full 12B model compression benchmark
python experiments/gpu_benchmark_20b.py

# Experiment 7: Per-tensor layer-by-layer error breakdown
python experiments/per_tensor_breakdown.py

# Experiment 8: Blend quality — does averaging destroy expert structure?
python experiments/blend_quality_test.py
```

### Results Summary

| Experiment | Scale | Compression | SNR (dB) | Key Finding |
|-----------|-------|-------------|----------|-------------|
| 1–3 | 100K–10M | 4.00x | 48.5 | Consistent ratio and quality across scales |
| 4–5 | 100K–10M | 4.00x | — | 3–5% training overhead, blending preserves accuracy |
| 6 | 11.96B | 4.00x | — | Works on GPT-OSS 20B (32 MoE experts) |
| 7 | 11.96B | — | Varies | MoE experts: lossless. Attention: 25.7% error |
| 8 | 11.96B | — | — | Blending preserves expert gate structure |

**The finding that matters:** At 12B scale, 79.9% of parameters (MoE experts) compress with <0.001% error, but 5.3% (attention projections) hit 25.7% error. Layer-selective quantization is the clear path forward.

---

## API Reference

### NeuralEchoV2

```python
echo = NeuralEchoV2(
    use_compression=True,   # False to store raw FP32 snapshots
    max_snapshots=10,       # FIFO buffer size
    blend_weight=0.8,       # Weight for current params vs. historical
    snapshot_interval=10    # Training steps between snapshots
)

# Integrated training step — backward, snapshot, blend, optimizer step
echo.train_step(model, loss, optimizer)

# Or use components individually:
echo.store_snapshot(model)              # Store a compressed snapshot now
echo.interpolate_parameters(model)      # Blend historical into current params
stats = echo.get_stats()                # Dict: num_snapshots, compression_ratio,
                                        #   memory_saved_mb, actual_ram_bytes, etc.
```

### QuantizedSnapshot

```python
from core.quantization import QuantizedSnapshot

# Compress
params = {n: p.data.clone() for n, p in model.named_parameters()}
snap = QuantizedSnapshot(params)

print(f"Compression: {snap.compression_ratio:.1f}x")   # ~4.0x
print(f"RAM used:    {snap.memory_bytes() / 1e6:.1f}MB")

# Decompress
restored = snap.dequantize()   # Dict[str, torch.Tensor]
```

---

## Roadmap

- [ ] Group-wise INT8 for attention layers (fix the 25.7% error)
- [ ] Mixed-precision snapshot strategy (per-tensor for experts, group-wise for attention)
- [ ] Controlled forgetting experiments (CIFAR-10 split tasks)
- [ ] HuggingFace Transformers / PyTorch Lightning integration
- [ ] TorchAO backend for production-grade quantization

---

## Citation

```bibtex
@software{neural_echo_v2_2026,
  title={Neural Echo V2: INT8 Compressed Parameter Snapshots},
  author={Stevens, Caleb},
  year={2026},
  note={4x memory reduction via INT8 quantization for parameter snapshot systems},
  url={https://github.com/Iztolie/neural-echo-v2}
}
```

## License

MIT — see [LICENSE](LICENSE).