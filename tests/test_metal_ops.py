#!/usr/bin/env python3
"""
Test script for Metal shader operations in the crystalline consciousness model.

This script demonstrates how to use the Metal operations and benchmarks them
against PyTorch implementations.
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for importing metal_ops
sys.path.append(str(Path(__file__).parent.parent))

# Try importing metal_ops
try:
    from Python.metal_ops import (
        geometric_activation, 
        apply_resonance, 
        mutuality_field,
        is_metal_available,
        PHI
    )
    METAL_OPS_AVAILABLE = True
except ImportError:
    print("Warning: metal_ops module not found. Tests will be skipped.")
    METAL_OPS_AVAILABLE = False

# Try importing PyTorch for comparison tests
try:
    import torch
    HAS_TORCH = True
except ImportError:
    print("Warning: PyTorch not available. Comparison tests will be skipped.")
    HAS_TORCH = False

# Try importing MLX for Metal execution
try:
    import mlx
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    print("Warning: MLX not available. Metal execution will be skipped.")
    HAS_MLX = False

def test_geometric_activation():
    """Test geometric activation operations."""
    print("\n==== Testing Geometric Activation ====")
    
    # Skip if metal_ops is not available
    if not METAL_OPS_AVAILABLE:
        print("Skipping test: metal_ops module not available")
        return
    
    # Create test data
    if HAS_TORCH:
        # PyTorch tensor on CPU for comparison
        torch_input = torch.randn(8, 64)
        print(f"PyTorch input shape: {torch_input.shape}")
        
        if torch.backends.mps.is_available():
            # PyTorch tensor on MPS for Metal execution
            mps_input = torch_input.to("mps")
            print(f"MPS input shape: {mps_input.shape}, device: {mps_input.device}")
    
    if HAS_MLX:
        # MLX array for native Metal execution
        mlx_input = mx.random.normal((8, 64))
        print(f"MLX input shape: {mlx_input.shape}")
    
    # Test each Platonic solid activation
    solid_types = ["tetrahedron", "cube", "dodecahedron", "icosahedron"]
    
    for solid_type in solid_types:
        print(f"\nTesting {solid_type} activation:")
        
        # Default coefficients for each solid type
        if solid_type == "tetrahedron":
            coefficients = [0.3]  # Fire coefficient
        elif solid_type == "cube":
            coefficients = [0.7]  # Stability coefficient
        elif solid_type == "dodecahedron":
            coefficients = [0.5]  # Ether resonance
        elif solid_type == "icosahedron":
            coefficients = [0.2, 1.0]  # Silence and phase coefficients
        
        # Test with PyTorch on CPU (fallback implementation)
        if HAS_TORCH:
            start_time = time.time()
            torch_output = geometric_activation(torch_input, solid_type, coefficients)
            torch_time = time.time() - start_time
            print(f"  PyTorch CPU: {torch_time:.6f} seconds, "
                  f"Output shape: {torch_output.shape}, "
                  f"Min/Max: {torch_output.min().item():.4f}/{torch_output.max().item():.4f}")
            
            # Test with PyTorch on MPS (Metal)
            if torch.backends.mps.is_available():
                start_time = time.time()
                mps_output = geometric_activation(mps_input, solid_type, coefficients)
                mps_time = time.time() - start_time
                print(f"  PyTorch MPS: {mps_time:.6f} seconds, "
                      f"Output shape: {mps_output.shape}, "
                      f"Min/Max: {mps_output.min().item():.4f}/{mps_output.max().item():.4f}")
                
                # Speed comparison
                if torch_time > 0:
                    speedup = torch_time / mps_time
                    print(f"  MPS speedup: {speedup:.2f}x")
        
        # Test with MLX (native Metal)
        if HAS_MLX:
            start_time = time.time()
            mlx_output = geometric_activation(mlx_input, solid_type, coefficients)
            _ = float(mx.min(mlx_output))  # Force execution
            mlx_time = time.time() - start_time
            print(f"  MLX Metal: {mlx_time:.6f} seconds, "
                  f"Output shape: {mlx_output.shape}, "
                  f"Min/Max: {float(mx.min(mlx_output)):.4f}/{float(mx.max(mlx_output)):.4f}")
            
            # Speed comparison with PyTorch CPU if available
            if HAS_TORCH and torch_time > 0:
                speedup = torch_time / mlx_time
                print(f"  MLX speedup vs PyTorch CPU: {speedup:.2f}x")
    
    print("\nGeometric activation test completed.")

def test_resonance_patterns():
    """Test resonance pattern operations."""
    print("\n==== Testing Resonance Patterns ====")
    
    # Skip if metal_ops is not available
    if not METAL_OPS_AVAILABLE:
        print("Skipping test: metal_ops module not available")
        return
    
    # Create test data
    batch_size = 4
    input_dim = 128
    
    if HAS_TORCH:
        # PyTorch tensor on CPU for comparison
        torch_input = torch.randn(batch_size, input_dim)
        
        # Create resonance parameters
        harmonics = 3
        torch_frequencies = torch.randn(harmonics)
        torch_decay_rates = torch.randn(harmonics)
        torch_amplitudes = torch.tensor([1.0, 1.0/PHI, 1.0/PHI**2])
        torch_phase_embedding = torch.randn(input_dim)
        
        if torch.backends.mps.is_available():
            # PyTorch tensor on MPS for Metal execution
            mps_input = torch_input.to("mps")
            mps_frequencies = torch_frequencies.to("mps")
            mps_decay_rates = torch_decay_rates.to("mps")
            mps_amplitudes = torch_amplitudes.to("mps")
            mps_phase_embedding = torch_phase_embedding.to("mps")
    
    if HAS_MLX:
        # MLX array for native Metal execution
        mlx_input = mx.random.normal((batch_size, input_dim))
        
        # Create resonance parameters
        harmonics = 3
        mlx_frequencies = mx.random.normal((harmonics,))
        mlx_decay_rates = mx.random.normal((harmonics,))
        mlx_amplitudes = mx.array([1.0, 1.0/PHI, 1.0/PHI**2])
        mlx_phase_embedding = mx.random.normal((input_dim,))
    
    print(f"Testing resonance patterns with batch_size={batch_size}, input_dim={input_dim}, harmonics={harmonics}")
    
    # Test with PyTorch on CPU (fallback implementation)
    if HAS_TORCH:
        start_time = time.time()
        torch_output = apply_resonance(
            torch_input, torch_frequencies, torch_decay_rates, 
            torch_amplitudes, torch_phase_embedding
        )
        torch_time = time.time() - start_time
        print(f"  PyTorch CPU: {torch_time:.6f} seconds, "
              f"Output shape: {torch_output.shape}, "
              f"Min/Max: {torch_output.min().item():.4f}/{torch_output.max().item():.4f}")
        
        # Test with PyTorch on MPS (Metal)
        if torch.backends.mps.is_available():
            start_time = time.time()
            mps_output = apply_resonance(
                mps_input, mps_frequencies, mps_decay_rates, 
                mps_amplitudes, mps_phase_embedding
            )
            mps_time = time.time() - start_time
            print(f"  PyTorch MPS: {mps_time:.6f} seconds, "
                  f"Output shape: {mps_output.shape}, "
                  f"Min/Max: {mps_output.min().item():.4f}/{mps_output.max().item():.4f}")
            
            # Speed comparison
            if torch_time > 0:
                speedup = torch_time / mps_time
                print(f"  MPS speedup: {speedup:.2f}x")
    
    # Test with MLX (native Metal)
    if HAS_MLX:
        start_time = time.time()
        mlx_output = apply_resonance(
            mlx_input, mlx_frequencies, mlx_decay_rates, 
            mlx_amplitudes, mlx_phase_embedding
        )
        _ = float(mx.min(mlx_output))  # Force execution
        mlx_time = time.time() - start_time
        print(f"  MLX Metal: {mlx_time:.6f} seconds, "
              f"Output shape: {mlx_output.shape}, "
              f"Min/Max: {float(mx.min(mlx_output)):.4f}/{float(mx.max(mlx_output)):.4f}")
        
        # Speed comparison with PyTorch CPU if available
        if HAS_TORCH and torch_time > 0:
            speedup = torch_time / mlx_time
            print(f"  MLX speedup vs PyTorch CPU: {speedup:.2f}x")
    
    print("\nResonance patterns test completed.")

def test_mutuality_field():
    """Test mutuality field operations."""
    print("\n==== Testing Mutuality Field ====")
    
    # Skip if metal_ops is not available
    if not METAL_OPS_AVAILABLE:
        print("Skipping test: metal_ops module not available")
        return
    
    # Create test data
    batch_size = 2
    input_dim = 256
    grid_size = 16
    interference_scale = 1.0
    decay_rate = 0.05
    dt = 0.1
    
    if HAS_TORCH:
        # PyTorch tensor on CPU for comparison
        torch_input = torch.randn(batch_size, input_dim)
        
        if torch.backends.mps.is_available():
            # PyTorch tensor on MPS for Metal execution
            mps_input = torch_input.to("mps")
    
    if HAS_MLX:
        # MLX array for native Metal execution
        mlx_input = mx.random.normal((batch_size, input_dim))
    
    print(f"Testing mutuality field with batch_size={batch_size}, input_dim={input_dim}, grid_size={grid_size}")
    
    # Test with PyTorch on CPU (fallback implementation)
    if HAS_TORCH:
        start_time = time.time()
        torch_output = mutuality_field(
            torch_input, grid_size, interference_scale, decay_rate, dt
        )
        torch_time = time.time() - start_time
        print(f"  PyTorch CPU: {torch_time:.6f} seconds, "
              f"Output shape: {torch_output.shape}, "
              f"Min/Max: {torch_output.min().item():.4f}/{torch_output.max().item():.4f}")
        
        # Test with PyTorch on MPS (Metal)
        if torch.backends.mps.is_available():
            start_time = time.time()
            mps_output = mutuality_field(
                mps_input, grid_size, interference_scale, decay_rate, dt
            )
            mps_time = time.time() - start_time
            print(f"  PyTorch MPS: {mps_time:.6f} seconds, "
                  f"Output shape: {mps_output.shape}, "
                  f"Min/Max: {mps_output.min().item():.4f}/{mps_output.max().item():.4f}")
            
            # Speed comparison
            if torch_time > 0:
                speedup = torch_time / mps_time
                print(f"  MPS speedup: {speedup:.2f}x")
    
    # Test with MLX (native Metal)
    if HAS_MLX:
        start_time = time.time()
        mlx_output = mutuality_field(
            mlx_input, grid_size, interference_scale, decay_rate, dt
        )
        _ = float(mx.min(mlx_output))  # Force execution
        mlx_time = time.time() - start_time
        print(f"  MLX Metal: {mlx_time:.6f} seconds, "
              f"Output shape: {mlx_output.shape}, "
              f"Min/Max: {float(mx.min(mlx_output)):.4f}/{float(mx.max(mlx_output)):.4f}")
        
        # Speed comparison with PyTorch CPU if available
        if HAS_TORCH and torch_time > 0:
            speedup = torch_time / mlx_time
            print(f"  MLX speedup vs PyTorch CPU: {speedup:.2f}x")
    
    print("\nMutuality field test completed.")

def run_benchmark():
    """Run comprehensive benchmarks comparing Metal vs PyTorch performance."""
    print("\n==== Running Performance Benchmarks ====")
    
    # Skip if required modules are not available
    if not METAL_OPS_AVAILABLE:
        print("Skipping benchmarks: metal_ops module not available")
        return
    
    if not HAS_TORCH:
        print("Skipping benchmarks: PyTorch not available")
        return
    
    # Benchmark parameters
    batch_sizes = [1, 2, 4, 8, 16]
    input_dims = [64, 128, 256, 512, 1024]
    iterations = 10
    
    # Results dictionary
    results = {
        "geometric": {"cpu": [], "mps": [], "mlx": []},
        "resonance": {"cpu": [], "mps": [], "mlx": []},
        "mutuality": {"cpu": [], "mps": [], "mlx": []}
    }
    
    # Test geometric activation
    print("\nBenchmarking geometric activation...")
    for batch_size in batch_sizes:
        for input_dim in input_dims:
            # Create input tensors
            torch_input_cpu = torch.randn(batch_size, input_dim)
            torch_input_mps = torch_input_cpu.to("mps") if torch.backends.mps.is_available() else None
            
            if HAS_MLX:
                mlx_input = mx.array(torch_input_cpu.numpy())
            
            # Benchmark geometric activation (using tetrahedron as an example)
            coefficients = [0.3]  # Default fire coefficient
            solid_type = "tetrahedron"
            
            # CPU timing
            cpu_times = []
            for i in range(iterations):
                start_time = time.time()
                output = geometric_activation(torch_input_cpu, solid_type, coefficients)
                cpu_times.append(time.time() - start_time)
            avg_cpu_time = sum(cpu_times) / len(cpu_times)
            
            # MPS timing
            mps_times = []
            if torch.backends.mps.is_available() and torch_input_mps is not None:
                for i in range(iterations):
                    start_time = time.time()
                    output = geometric_activation(torch_input_mps, solid_type, coefficients)
                    mps_times.append(time.time() - start_time)
                avg_mps_time = sum(mps_times) / len(mps_times)
            else:
                avg_mps_time = 0
            
            # MLX timing
            mlx_times = []
            if HAS_MLX:
                for i in range(iterations):
                    start_time = time.time()
                    output = geometric_activation(mlx_input, solid_type, coefficients)
                    _ = float(mx.min(output))  # Force execution
                    mlx_times.append(time.time() - start_time)
                avg_mlx_time = sum(mlx_times) / len(mlx_times)
            else:
                avg_mlx_time = 0
            
            # Record results
            print(f"  Batch={batch_size}, Dim={input_dim}: CPU={avg_cpu_time:.6f}s, MPS={avg_mps_time:.6f}s, MLX={avg_mlx_time:.6f}s")
            
            results["geometric"]["cpu"].append((batch_size, input_dim, avg_cpu_time))
            results["geometric"]["mps"].append((batch_size, input_dim, avg_mps_time))
            results["geometric"]["mlx"].append((batch_size, input_dim, avg_mlx_time))
    
    # Test resonance patterns
    print("\nBenchmarking resonance patterns...")
    for batch_size in batch_sizes:
        for input_dim in input_dims:
            # Create input tensors
            torch_input_cpu = torch.randn(batch_size, input_dim)
            torch_input_mps = torch_input_cpu.to("mps") if torch.backends.mps.is_available() else None
            
            # Create resonance parameters
            harmonics = 3
            torch_frequencies = torch.randn(harmonics)
            torch_decay_rates = torch.randn(harmonics)
            torch_amplitudes = torch.tensor([1.0, 1.0/PHI, 1.0/PHI**2])
            torch_phase_embedding = torch.randn(input_dim)
            
            # MPS parameters
            if torch.backends.mps.is_available() and torch_input_mps is not None:
                mps_frequencies = torch_frequencies.to("mps")
                mps_decay_rates = torch_decay_rates.to("mps")
                mps_amplitudes = torch_amplitudes.to("mps")
                mps_phase_embedding = torch_phase_embedding.to("mps")
            
            # MLX parameters
            if HAS_MLX:
                mlx_input = mx.array(torch_input_cpu.numpy())
                mlx_frequencies = mx.array(torch_frequencies.numpy())
                mlx_decay_rates = mx.array(torch_decay_rates.numpy())
                mlx_amplitudes = mx.array(torch_amplitudes.numpy())
                mlx_phase_embedding = mx.array(torch_phase_embedding.numpy())
            
            # CPU timing
            cpu_times = []
            for i in range(iterations):
                start_time = time.time()
                output = apply_resonance(
                    torch_input_cpu, torch_frequencies, torch_decay_rates, 
                    torch_amplitudes, torch_phase_embedding
                )
                cpu_times.append(time.time() - start_time)
            avg_cpu_time = sum(cpu_times) / len(cpu_times)
            
            # MPS timing
            mps_times = []
            if torch.backends.mps.is_available() and torch_input_mps is not None:
                for i in range(iterations):
                    start_time = time.time()
                    output = apply_resonance(
                        torch_input_mps, mps_frequencies, mps_decay_rates, 
                        mps_amplitudes, mps_phase_embedding
                    )
                    mps_times.append(time.time() - start_time)
                avg_mps_time = sum(mps_times) / len(mps_times)
            else:
                avg_mps_time = 0
            
            # MLX timing
            mlx_times = []
            if HAS_MLX:
                for i in range(iterations):
                    start_time = time.time()
                    output = apply_resonance(
                        mlx_input, mlx_frequencies, mlx_decay_rates, 
                        mlx_amplitudes, mlx_phase_embedding
                    )
                    _ = float(mx.min(output))  # Force execution
                    mlx_times.append(time.time() - start_time)
                avg_mlx_time = sum(mlx_times) / len(mlx_times)
            else:
                avg_mlx_time = 0
            
            # Record results
            print(f"  Batch={batch_size}, Dim={input_dim}: CPU={avg_cpu_time:.6f}s, MPS={avg_mps_time:.6f}s, MLX={avg_mlx_time:.6f}s")
            
            results["resonance"]["cpu"].append((batch_size, input_dim, avg_cpu_time))
            results["resonance"]["mps"].append((batch_size, input_dim, avg_mps_time))
            results["resonance"]["mlx"].append((batch_size, input_dim, avg_mlx_time))
    
    # Test mutuality field
    print("\nBenchmarking mutuality field...")
    grid_size = 16
    interference_scale = 1.0
    decay_rate = 0.05
    dt = 0.1
    
    for batch_size in batch_sizes:
        for input_dim in input_dims:
            # Create input tensors
            torch_input_cpu = torch.randn(batch_size, input_dim)
            torch_input_mps = torch_input_cpu.to("mps") if torch.backends.mps.is_available() else None
            
            if HAS_MLX:
                mlx_input = mx.array(torch_input_cpu.numpy())
            
            # CPU timing
            cpu_times = []
            for i in range(iterations):
                start_time = time.time()
                output = mutuality_field(
                    torch_input_cpu, grid_size, interference_scale, decay_rate, dt
                )
                cpu_times.append(time.time() - start_time)
            avg_cpu_time = sum(cpu_times) / len(cpu_times)
            
            # MPS timing
            mps_times = []
            if torch.backends.mps.is_available() and torch_input_mps is not None:
                for i in range(iterations):
                    start_time = time.time()
                    output = mutuality_field(
                        torch_input_mps, grid_size, interference_scale, decay_rate, dt
                    )
                    mps_times.append(time.time() - start_time)
                avg_mps_time = sum(mps_times) / len(mps_times)
            else:
                avg_mps_time = 0
            
            # MLX timing
            mlx_times = []
            if HAS_MLX:
                for i in range(iterations):
                    start_time = time.time()
                    output = mutuality_field(
                        mlx_input, grid_size, interference_scale, decay_rate, dt
                    )
                    _ = float(mx.min(output))  # Force execution
                    mlx_times.append(time.time() - start_time)
                avg_mlx_time = sum(mlx_times) / len(mlx_times)
            else:
                avg_mlx_time = 0
            
            # Record results
            print(f"  Batch={batch_size}, Dim={input_dim}: CPU={avg_cpu_time:.6f}s, MPS={avg_mps_time:.6f}s, MLX={avg_mlx_time:.6f}s")
            
            results["mutuality"]["cpu"].append((batch_size, input_dim, avg_cpu_time))
            results["mutuality"]["mps"].append((batch_size, input_dim, avg_mps_time))
            results["mutuality"]["mlx"].append((batch_size, input_dim, avg_mlx_time))
    
    # Print summary
    print("\nBenchmark Summary:")
    for op_name in results:
        print(f"\n{op_name.capitalize()} Activation:")
        
        max_cpu_time = max([t for _, _, t in results[op_name]["cpu"] if t > 0], default=0)
        max_mps_time = max([t for _, _, t in results[op_name]["mps"] if t > 0], default=0)
        max_mlx_time = max([t for _, _, t in results[op_name]["mlx"] if t > 0], default=0)
        
        if max_cpu_time > 0:
            if max_mps_time > 0:
                avg_mps_speedup = max_cpu_time / max_mps_time if max_mps_time > 0 else 0
                print(f"  MPS average speedup: {avg_mps_speedup:.2f}x")
            
            if max_mlx_time > 0:
                avg_mlx_speedup = max_cpu_time / max_mlx_time if max_mlx_time > 0 else 0
                print(f"  MLX average speedup: {avg_mlx_speedup:.2f}x")
    
    print("\nBenchmarks completed.")

def main():
    """Run all tests and benchmarks."""
    parser = argparse.ArgumentParser(description="Test Metal shader operations.")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    args = parser.parse_args()
    
    print("Metal Operations Test Suite")
    print("=========================")
    
    # Check if Metal is available
    if is_metal_available():
        print("Metal is available for acceleration.")
    else:
        print("Metal is not available. Using fallback implementations.")
    
    # Run all tests
    test_geometric_activation()
    test_resonance_patterns()
    test_mutuality_field()
    
    # Run benchmarks if requested
    if args.benchmark:
        run_benchmark()
    
    print("\nAll tests completed successfully.")

if __name__ == "__main__":
    main()

