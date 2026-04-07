"""
Test Neural Echo V2 with real pretrained models
Using actual computer vision tasks that share features
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, Subset
import numpy as np
from core.echo_memory import NeuralEchoV2
import time

class RealModelTest:
    def __init__(self, model_name="resnet18", device=None):
        # Automatically detect best available device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("Using Apple Metal Performance Shaders (MPS)")
            else:
                self.device = torch.device("cpu")
                print("Using CPU (this will be slower)")
        else:
            self.device = torch.device(device)
            
        self.model_name = model_name
        print(f"Device set to: {self.device}")
        
    def load_pretrained_model(self):
        """Load pretrained model and adapt for CIFAR"""
        if self.model_name == "resnet18":
            # Use weights parameter instead of pretrained
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # Adapt for CIFAR (32x32 images, 10 classes)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()  # Remove maxpool for small images
            model.fc = nn.Linear(512, 10)
        elif self.model_name == "mobilenet_v2":
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            model.classifier[1] = nn.Linear(1280, 10)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        return model.to(self.device)
    
    def prepare_data(self):
        """Prepare CIFAR-10 split into tasks"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Download CIFAR-10
        train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
        
        # Split into 5 tasks (2 classes each)
        # Task 0: airplane, automobile
        # Task 1: bird, cat  
        # Task 2: deer, dog
        # Task 3: frog, horse
        # Task 4: ship, truck
        
        tasks = []
        for task_id in range(5):
            classes = [task_id * 2, task_id * 2 + 1]
            
            # Get train indices for these classes
            train_indices = [i for i, (_, label) in enumerate(train_dataset) 
                           if label in classes]
            test_indices = [i for i, (_, label) in enumerate(test_dataset)
                          if label in classes]
            
            # Create subset
            train_subset = Subset(train_dataset, train_indices[:1000])  # 1000 samples per task
            test_subset = Subset(test_dataset, test_indices[:200])
            
            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
            
            tasks.append({
                'train': train_loader,
                'test': test_loader,
                'classes': classes,
                'name': f"CIFAR-10 Classes {classes}"
            })
        
        return tasks
    
    def train_epoch(self, model, loader, optimizer, echo=None):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            outputs = model(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)
            
            if echo:
                echo.train_step(model, loss, optimizer)
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(loader), 100. * correct / total
    
    def evaluate(self, model, loader):
        """Evaluate model accuracy"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return 100. * correct / total
    
    def run_continual_learning(self, use_echo=True):
        """Run continual learning experiment"""
        model = self.load_pretrained_model()
        tasks = self.prepare_data()
        
        # Setup Neural Echo if requested
        echo = None
        if use_echo:
            echo = NeuralEchoV2(
                use_compression=True,
                max_snapshots=10,
                blend_weight=0.85,  # Higher weight for current task
                snapshot_interval=20
            )
        
        # Track accuracy on all tasks
        task_accuracies = {i: [] for i in range(len(tasks))}
        
        print(f"\n{'='*60}")
        print(f"Testing {self.model_name} {'WITH' if use_echo else 'WITHOUT'} Neural Echo")
        print('='*60)
        
        # Train on each task sequentially
        for current_task_id, task in enumerate(tasks):
            print(f"\nTraining on {task['name']}...")
            
            # Use lower learning rate for pretrained model
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            
            # Train for a few epochs
            for epoch in range(3):
                train_loss, train_acc = self.train_epoch(
                    model, task['train'], optimizer, echo
                )
                print(f"  Epoch {epoch+1}: Loss={train_loss:.3f}, Acc={train_acc:.1f}%")
            
            # Evaluate on all previous tasks
            print("\nEvaluating on all tasks:")
            for eval_task_id in range(current_task_id + 1):
                acc = self.evaluate(model, tasks[eval_task_id]['test'])
                task_accuracies[eval_task_id].append(acc)
                print(f"  Task {eval_task_id}: {acc:.1f}%")
        
        # Calculate average forgetting
        total_forgetting = 0
        count = 0
        
        for task_id, accs in task_accuracies.items():
            if len(accs) > 1:
                max_acc = max(accs)
                final_acc = accs[-1]
                forgetting = max_acc - final_acc
                total_forgetting += forgetting
                count += 1
                print(f"\nTask {task_id} forgetting: {forgetting:.1f}%")
        
        avg_forgetting = total_forgetting / count if count > 0 else 0
        
        if echo:
            stats = echo.get_stats()
            print(f"\nNeural Echo Stats:")
            print(f"  Snapshots stored: {stats['num_snapshots']}")
            print(f"  Memory saved: {stats['memory_saved_mb']:.1f}MB")
        
        return avg_forgetting, task_accuracies

def main():
    print("="*70)
    print("NEURAL ECHO V2 - REAL MODEL TEST")
    print("="*70)
    
    # Check for available device
    device = None  # Will auto-detect
    
    # Test with ResNet18
    tester = RealModelTest(model_name="resnet18", device=device)
    
    # Run WITH Neural Echo
    print("\n" + "="*70)
    print("EXPERIMENT 1: WITH NEURAL ECHO")
    print("="*70)
    forgetting_with, accs_with = tester.run_continual_learning(use_echo=True)
    
    # Run WITHOUT Neural Echo
    print("\n" + "="*70)
    print("EXPERIMENT 2: WITHOUT NEURAL ECHO (Baseline)")
    print("="*70)
    tester = RealModelTest(model_name="resnet18")  # Reset model
    forgetting_without, accs_without = tester.run_continual_learning(use_echo=False)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Average Forgetting WITHOUT Neural Echo: {forgetting_without:.1f}%")
    print(f"Average Forgetting WITH Neural Echo: {forgetting_with:.1f}%")
    
    if forgetting_without > 0:
        improvement = (forgetting_without - forgetting_with) / forgetting_without * 100
        print(f"\nForgetting Reduction: {improvement:.1f}%")
        
        if improvement > 0:
            print("✅ Neural Echo reduced catastrophic forgetting!")
        else:
            print("❌ Neural Echo did not help in this scenario")
    
    # Test with different model
    print("\n" + "="*70)
    print("TESTING WITH MOBILENET V2")
    print("="*70)
    
    tester_mobile = RealModelTest(model_name="mobilenet_v2", device=device)
    forgetting_mobile_with, _ = tester_mobile.run_continual_learning(use_echo=True)
    
    tester_mobile = RealModelTest(model_name="mobilenet_v2", device=device)
    forgetting_mobile_without, _ = tester_mobile.run_continual_learning(use_echo=False)
    
    print(f"\nMobileNet Forgetting WITHOUT Echo: {forgetting_mobile_without:.1f}%")
    print(f"MobileNet Forgetting WITH Echo: {forgetting_mobile_with:.1f}%")

if __name__ == "__main__":
    main()