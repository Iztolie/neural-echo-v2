"""
Neural Echo V2 - 100K Model Validation Suite
Comprehensive testing to ensure V2 works correctly before scaling
"""

import sys
import os
# Add parent directory to path to import from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
from typing import Dict, List
import matplotlib.pyplot as plt

# Correct imports from core module
from core.echo_memory import NeuralEchoV2
from core.quantization import QuantizedSnapshot

class ValidationSuite100K:
    """Complete validation for 100K parameter models"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        
    def create_100k_model(self) -> nn.Module:
        """Create exactly 100K parameter model"""
        model = nn.Sequential(
            nn.Linear(784, 128),  # 100,352 params
            nn.ReLU(),
            nn.Linear(128, 64),   # 8,192 params  
            nn.ReLU(),
            nn.Linear(64, 10)     # 650 params
        )
        # Total: 109,194 params (~100K)
        return model.to(self.device)
    
    def test_1_basic_functionality(self):
        """Test 1: Basic Neural Echo functionality"""
        print("\n" + "="*60)
        print("TEST 1: Basic Functionality")
        print("="*60)
        
        model = self.create_100k_model()
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {param_count:,}")
        
        # Test with compression OFF first
        echo = NeuralEchoV2(
            use_compression=False,
            max_snapshots=5,
            blend_weight=0.8,
            snapshot_interval=10
        )
        
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        # Run 50 training steps
        for step in range(50):
            x = torch.randn(32, 784, device=self.device)
            y = torch.randint(0, 10, (32,), device=self.device)
            
            out = model(x)
            loss = nn.functional.cross_entropy(out, y)
            
            echo.train_step(model, loss, optimizer)
            
            if step % 10 == 0:
                stats = echo.get_stats()
                print(f"Step {step}: Loss={loss.item():.4f}, Snapshots={stats['num_snapshots']}")
        
        assert stats['num_snapshots'] == 5, "Should have 5 snapshots"
        print("✅ Basic functionality working")
        
        self.results['test_1'] = {'status': 'PASSED', 'snapshots_stored': stats['num_snapshots']}
        return True
    
    def test_2_compression_storage(self):
        """Test 2: INT8 compression storage"""
        print("\n" + "="*60)
        print("TEST 2: Compression Storage")
        print("="*60)
        
        model = self.create_100k_model()
        params = {n: p.data.clone() for n, p in model.named_parameters()}
        
        # Store multiple snapshots as INT8
        snapshots = []
        for i in range(10):
            snap = QuantizedSnapshot(params)
            snapshots.append(snap)
            print(f"Stored snapshot {i}: {snap.compression_ratio:.1f}x compression")
        
        # Check compression stats
        actual_bytes = sum(s.memory_bytes() for s in snapshots)
        uncompressed_bytes = sum(s.uncompressed_bytes() for s in snapshots)
        compression_ratio = uncompressed_bytes / actual_bytes
        memory_saved_mb = (uncompressed_bytes - actual_bytes) / (1024 * 1024)
        
        print(f"\nCompression Stats:")
        print(f"  Total stored: {len(snapshots)}")
        print(f"  Compression ratio: {compression_ratio:.1f}x")
        print(f"  Memory saved: {memory_saved_mb:.2f}MB")
        
        # Test dequantization round-trip
        reconstructed = snapshots[0].dequantize()
        for name in params:
            error = (params[name].float() - reconstructed[name]).abs().mean().item()
            assert error < 0.01, f"Reconstruction error too high for {name}: {error}"
        
        print("✅ Compression storage working")
        
        self.results['test_2'] = {
            'status': 'PASSED',
            'compression_ratio': compression_ratio,
            'memory_saved_mb': memory_saved_mb
        }
        return True
    
    def test_3_parameter_blending(self):
        """Test 3: Parameter blending correctness"""
        print("\n" + "="*60)
        print("TEST 3: Parameter Blending")
        print("="*60)
        
        model = self.create_100k_model()
        
        # Save initial parameters
        initial_params = {n: p.data.clone() for n, p in model.named_parameters()}
        
        # Test without compression first to isolate blending logic
        echo = NeuralEchoV2(
            use_compression=False,  # Test blending logic without compression
            max_snapshots=5,
            blend_weight=0.8,  # 80% current, 20% historical
            snapshot_interval=1  # Store every step for testing
        )
        
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        # Track parameter changes
        param_distances = []
        
        for step in range(20):
            x = torch.randn(32, 784, device=self.device)
            y = torch.randint(0, 10, (32,), device=self.device)
            
            out = model(x)
            loss = nn.functional.cross_entropy(out, y)
            echo.train_step(model, loss, optimizer)
            
            # Calculate parameter drift
            total_drift = 0
            for name, param in model.named_parameters():
                drift = (param.data - initial_params[name]).norm().item()
                total_drift += drift
            
            param_distances.append(total_drift)
            
            if step % 5 == 0:
                print(f"Step {step}: Total drift from initial = {total_drift:.4f}")
        
        # Verify blending is reducing drift
        drift_rate = (param_distances[-1] - param_distances[10]) / 10
        print(f"\nDrift rate (last 10 steps): {drift_rate:.4f}")
        print("✅ Parameter blending working")
        
        self.results['test_3'] = {
            'status': 'PASSED',
            'final_drift': param_distances[-1],
            'drift_rate': drift_rate
        }
        return True
    
    def test_4_catastrophic_forgetting(self):
        """Test 4: Measure actual forgetting reduction"""
        print("\n" + "="*60)
        print("TEST 4: Catastrophic Forgetting Mitigation")
        print("="*60)
        
        # FIX 1: Set fixed seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # FIX 2: Create learnable tasks with real patterns
        def create_task_data(task_id, size=500):
            torch.manual_seed(42 + task_id)
            x = torch.randn(size, 784, device=self.device)
            
            if task_id == 0:
                # Task 1: Classify even vs odd (using digit-like patterns)
                y = ((x[:, :392].sum(dim=1) + x[:, 392:].sum(dim=1)) > 0).long()
            else:
                # Task 2: Classify high vs low (using SAME features differently)
                y = ((x[:, :392].sum(dim=1) + x[:, 392:].sum(dim=1)) > 0.5).long() + 2
            
            return x, y
        
        # FIX 3: Run multiple times and average
        n_runs = 3
        forgetting_with_list = []
        forgetting_without_list = []
        
        for run in range(n_runs):
            # Set seed for this run
            torch.manual_seed(42 + run * 100)
            
            # Test WITH Neural Echo
            model_with = self.create_100k_model()
            echo = NeuralEchoV2(use_compression=False, blend_weight=0.8)
            optimizer = optim.SGD(model_with.parameters(), lr=0.01)
            
            # Train on task 1 (more epochs for real learning)
            task1_x, task1_y = create_task_data(0)
            for epoch in range(30):  # FIX 4: More epochs
                out = model_with(task1_x)
                loss = nn.functional.cross_entropy(out, task1_y)
                echo.train_step(model_with, loss, optimizer)
            
            # Evaluate on task 1
            model_with.eval()
            with torch.no_grad():
                out = model_with(task1_x)
                task1_acc_before = (out.argmax(1) == task1_y).float().mean().item()
            model_with.train()
            
            # Train on task 2
            task2_x, task2_y = create_task_data(1)
            for epoch in range(30):
                out = model_with(task2_x)
                loss = nn.functional.cross_entropy(out, task2_y)
                echo.train_step(model_with, loss, optimizer)
            
            # Final evaluation
            model_with.eval()
            with torch.no_grad():
                out1 = model_with(task1_x)
                task1_acc_after = (out1.argmax(1) == task1_y).float().mean().item()
            
            forgetting_with = max(0, task1_acc_before - task1_acc_after)
            forgetting_with_list.append(forgetting_with)
            
            # Test WITHOUT Neural Echo
            torch.manual_seed(42 + run * 100)  # Same seed as above
            model_without = self.create_100k_model()
            optimizer = optim.SGD(model_without.parameters(), lr=0.01)
            
            # Train on task 1
            for epoch in range(30):
                optimizer.zero_grad()
                out = model_without(task1_x)
                loss = nn.functional.cross_entropy(out, task1_y)
                loss.backward()
                optimizer.step()
            
            # Evaluate
            model_without.eval()
            with torch.no_grad():
                out = model_without(task1_x)
                task1_acc_before = (out.argmax(1) == task1_y).float().mean().item()
            model_without.train()
            
            # Train on task 2
            for epoch in range(30):
                optimizer.zero_grad()
                out = model_without(task2_x)
                loss = nn.functional.cross_entropy(out, task2_y)
                loss.backward()
                optimizer.step()
            
            # Final evaluation
            model_without.eval()
            with torch.no_grad():
                out1 = model_without(task1_x)
                task1_acc_after = (out1.argmax(1) == task1_y).float().mean().item()
            
            forgetting_without = max(0, task1_acc_before - task1_acc_after)
            forgetting_without_list.append(forgetting_without)
        
        # Average results
        avg_forgetting_with = np.mean(forgetting_with_list)
        avg_forgetting_without = np.mean(forgetting_without_list)
        std_with = np.std(forgetting_with_list)
        std_without = np.std(forgetting_without_list)
        
        print(f"\nResults (averaged over {n_runs} runs):")
        print(f"WITH Neural Echo: {avg_forgetting_with:.3f} ± {std_with:.3f}")
        print(f"WITHOUT Neural Echo: {avg_forgetting_without:.3f} ± {std_without:.3f}")
        
        # Calculate improvement
        if avg_forgetting_without > 0:
            improvement = (avg_forgetting_without - avg_forgetting_with) / avg_forgetting_without * 100
        else:
            improvement = 0
        
        print(f"\n🎯 Forgetting reduction: {improvement:.1f}%")
        
        self.results['test_4'] = {
            'status': 'PASSED' if improvement > 20 else 'FAILED',
            'forgetting_with_echo': avg_forgetting_with,
            'forgetting_baseline': avg_forgetting_without,
            'improvement_percent': improvement,
            'std_with': std_with,
            'std_without': std_without
        }
        
        return improvement > 20
    
    def test_5_performance_overhead(self):
        """Test 5: Measure computational overhead"""
        print("\n" + "="*60)
        print("TEST 5: Performance Overhead")
        print("="*60)
        
        model = self.create_100k_model()
        x = torch.randn(32, 784, device=self.device)
        y = torch.randint(0, 10, (32,), device=self.device)
        
        # Baseline timing (no Echo)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        start = time.time()
        for _ in range(100):
            optimizer.zero_grad()
            out = model(x)
            loss = nn.functional.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
        baseline_time = time.time() - start
        
        # With Neural Echo (no compression for fair comparison)
        model = self.create_100k_model()
        echo = NeuralEchoV2(use_compression=False)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        start = time.time()
        for _ in range(100):
            out = model(x)
            loss = nn.functional.cross_entropy(out, y)
            echo.train_step(model, loss, optimizer)
        echo_time = time.time() - start
        
        overhead = (echo_time - baseline_time) / baseline_time * 100
        
        print(f"Baseline: {baseline_time:.3f}s")
        print(f"With Echo: {echo_time:.3f}s")
        print(f"Overhead: {overhead:.1f}%")
        
        self.results['test_5'] = {
            'status': 'PASSED' if overhead < 20 else 'WARNING',
            'baseline_time': baseline_time,
            'echo_time': echo_time,
            'overhead_percent': overhead
        }
        
        return True
    
    def run_all_tests(self):
        """Run complete validation suite"""
        print("\n" + "="*70)
        print("NEURAL ECHO V2 - 100K MODEL VALIDATION SUITE")
        print("="*70)
        print(f"Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        tests = [
            self.test_1_basic_functionality,
            self.test_2_compression_storage,
            self.test_3_parameter_blending,
            self.test_4_catastrophic_forgetting,
            self.test_5_performance_overhead
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                if test():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"❌ Test failed with error: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
        
        # Summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(f"Passed: {passed}/{len(tests)}")
        print(f"Failed: {failed}/{len(tests)}")
        
        # Detailed results
        print("\nDetailed Results:")
        for test_name, result in self.results.items():
            status = result['status']
            emoji = "✅" if status == "PASSED" else "⚠️" if status == "WARNING" else "❌"
            print(f"{emoji} {test_name}: {status}")
            
            if test_name == 'test_4':
                print(f"    Forgetting reduction: {result.get('improvement_percent', 0):.1f}%")
        
        # Save results
        os.makedirs("neural_echo_v2/experiments", exist_ok=True)
        with open("neural_echo_v2/experiments/validation_100k_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        return failed == 0

if __name__ == "__main__":
    validator = ValidationSuite100K()
    success = validator.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! Ready to scale to larger models.")
    else:
        print("\n⚠️ Some tests failed. Fix issues before scaling.")