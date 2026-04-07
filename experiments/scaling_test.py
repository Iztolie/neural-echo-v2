"""
Scaling tests for Neural Echo V2
Test with 100K, 1M, and 10M parameter models
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import time
import json
from core.echo_memory import NeuralEchoV2

def create_model(size: str) -> nn.Module:
    """Create models of different sizes"""
    if size == "100k":
        return nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    elif size == "1m":
        return nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    elif size == "10m":
        return nn.Sequential(
            nn.Linear(784, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

def run_scaling_test(model_size: str, use_compression: bool):
    """Run scaling test for a specific model size"""
    print(f"\nTesting {model_size} model {'with' if use_compression else 'without'} compression")
    print("-" * 60)
    
    model = create_model(model_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    echo = NeuralEchoV2(
        use_compression=use_compression,
        max_snapshots=10,
        blend_weight=0.8,
        snapshot_interval=10
    )
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Training metrics
    times = []
    memory_stats = []
    
    for step in range(100):
        start = time.time()
        
        # Dummy data
        x = torch.randn(32, 784)
        y = torch.randint(0, 10, (32,))
        
        # Forward pass
        out = model(x)
        loss = nn.functional.cross_entropy(out, y)
        
        # Neural Echo training step
        echo.train_step(model, loss, optimizer)
        
        times.append(time.time() - start)
        
        if step % 20 == 0:
            stats = echo.get_stats()
            memory_stats.append(stats)
            print(f"Step {step}: Loss={loss.item():.4f}, "
                  f"Memory saved={stats.get('memory_saved_mb', 0):.2f}MB")
    
    # Final statistics
    final_stats = echo.get_stats()
    final_stats['avg_step_time'] = sum(times) / len(times)
    final_stats['model_size'] = model_size
    final_stats['total_parameters'] = total_params
    
    echo.close()
    
    return final_stats

# Run all tests
if __name__ == "__main__":
    results = {}
    
    for size in ["100k", "1m", "10m"]:
        # With compression
        results[f"{size}_compressed"] = run_scaling_test(size, use_compression=True)
        
        # Without compression (baseline)
        results[f"{size}_baseline"] = run_scaling_test(size, use_compression=False)
    
    # Save results
    with open("neural_echo_v2/paper/scaling_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("SCALING TEST COMPLETE")
    print("Results saved to scaling_results.json")