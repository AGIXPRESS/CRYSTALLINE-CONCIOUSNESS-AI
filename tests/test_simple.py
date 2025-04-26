#!/usr/bin/env python3
"""
Simple demonstration of Metal operations for crystalline consciousness model.

This script provides basic examples of how to use the Metal shader operations.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for importing metal_ops
sys.path.append(str(Path(__file__).parent.parent))

# Try importing required modules
try:
    from Python.metal_ops import (
        geometric_activation, 
        apply_resonance, 
        mutuality_field,
        is_metal_available,
        PHI
    )
    print("Metal operations module loaded successfully.")
except ImportError:
    print("Error: metal_ops module not found.")
    sys.exit(1)

try:
    import torch
    HAS_TORCH = True
    print("PyTorch available for testing.")
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available.")

try:
    import mlx
    import mlx.core as mx
    HAS_MLX = True
    print("MLX available for Metal execution.")
except ImportError:
    HAS_MLX = False
    print("MLX not available.")

def example_geometric_activation():
    """Demonstrate geometric activation operations."""
    print("\n=== Geometric Activation Example ===")
    
    # Create sample data
    if HAS_TORCH:
        # Create a simple PyTorch tensor
        input_tensor = torch.randn(4, 32)
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Try each Platonic solid activation
        for solid_type in ["tetrahedron", "cube", "dodecahedron", "icosahedron"]:
            print(f"\nApplying {solid_type} activation:")
            output = geometric_activation(input_tensor, solid_type)
            print(f"  Output shape: {output.shape}")
            print(f"  Min/Max values: {output.min().item():.4f}/{output.max().item():.4f}")
            
            # Try with MPS if available
            if torch.backends.mps.is_available():
                mps_input = input_tensor.to("mps")
                mps_output = geometric_activation(mps_input, solid_type)
                print(f"  MPS output shape: {mps_output.shape}")
                print(f"  MPS Min/Max values: {mps_output.min().item():.4f}/{mps_output.max().item():.4f}")
    
    # MLX example
    if HAS_MLX:
        # Create a simple MLX array
        input_array = mx.random.normal((4, 32))
        print(f"\nMLX input array shape: {input_array.shape}")
        
        # Try tetrahedron activation
        solid_type = "tetrahedron"
        print(f"Applying {solid_type} activation with MLX:")
        output = geometric_activation(input_array, solid_type)
        print(f"  Output shape: {output.shape}")
        print(f"  Min/Max values: {float(mx.min(output)):.4f}/{float(mx.max(output)):.4f}")

def example_resonance_patterns():
    """Demonstrate resonance pattern operations."""
    print("\n=== Resonance Patterns Example ===")
    
    if HAS_TORCH:
        # Create sample data
        batch_size = 2
        dim = 64
        harmonics = 3
        
        # Input tensor
        input_tensor = torch.randn(batch_size, dim)
        
        # Resonance parameters
        frequencies = torch.randn(harmonics)  # Random frequencies
        decay_rates = torch.ones(harmonics)   # Unity decay rates
        amplitudes = torch.tensor([1.0, 1.0/PHI, 1.0/PHI**2])  # Golden ratio amplitudes
        phase_embedding = torch.randn(dim)    # Random phase embedding
        
        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Resonance parameters: {harmonics} harmonics")
        
        # Apply resonance
        output = apply_resonance(
            input_tensor, frequencies, decay_rates, amplitudes, phase_embedding
        )
        
        print(f"Output tensor shape: {output.shape}")
        print(f"Min/Max values: {output.min().item():.4f}/{output.max().item():.4f}")
        
        # Try with explicit time values
        time_values = torch.ones(batch_size, 1) * 0.5  # Half time
        output_timed = apply_resonance(
            input_tensor, frequencies, decay_rates, amplitudes, phase_embedding, time_values
        )
        
        print(f"Output with time values shape: {output_timed.shape}")
        print(f"Min/Max values: {output_timed.min().item():.4f}/{output_timed.max().item():.4f}")
    
    if HAS_MLX:
        # Create sample data with MLX
        batch_size = 2
        dim = 64
        harmonics = 3
        
        # Input array
        input_array = mx.random.normal((batch_size, dim))
        
        # Resonance parameters
        frequencies = mx.random.normal((harmonics,))
        decay_rates = mx.ones((harmonics,))
        amplitudes = mx.array([1.0, 1.0/PHI, 1.0/PHI**2])
        phase_embedding = mx.random.normal((dim,))
        
        print(f"\nMLX input array shape: {input_array.shape}")
        
        # Apply resonance
        output = apply_resonance(
            input_array, frequencies, decay_rates, amplitudes, phase_embedding
        )
        
        print(f"MLX output array shape: {output.shape}")
        print(f"Min/Max values: {float(mx.min(output)):.4f}/{float(mx.max(output)):.4f}")

def example_mutuality_field():
    """Demonstrate mutuality field operations."""
    print("\n=== Mutuality Field Example ===")
    
    if HAS_TORCH:
        # Create sample data
        batch_size = 2
        input_dim = 256
        grid_size = 16
        
        # Input tensor
        input_tensor = torch.randn(batch_size, input_dim)
        
        # Field parameters
        interference_scale = 1.0
        decay_rate = 0.05
        dt = 0.1
        
        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Grid size: {grid_size}x{grid_size}")
        
        # Apply mutuality field
        output = mutuality_field(
            input_tensor, grid_size, interference_scale, decay_rate, dt
        )
        
        print(f"Output tensor shape: {output.shape}")
        print(f"Min/Max values: {output.min().item():.4f}/{output.max().item():.4f}")
        
        # Apply again to demonstrate persistence effect
        output2 = mutuality_field(
            input_tensor, grid_size, interference_scale, decay_rate, dt
        )
        
        print(f"Second output tensor shape: {output2.shape}")
        print(f"Min/Max values: {output2.min().item():.4f}/{output2.max().item():.4f}")
        print("Note: Values should reflect persistence effect from previous call")
    
    if HAS_MLX:
        # Create sample data with MLX
        batch_size = 2
        input_dim = 256
        grid_size = 16
        
        # Input array
        input_array = mx.random.normal((batch_size, input_dim))
        
        # Field parameters
        interference_scale = 1.0
        decay_rate = 0.05
        dt = 0.1
        
        print(f"\nMLX input array shape: {input_array.shape}")
        
        # Apply mutuality field
        output = mutuality_field(
            input_array, grid_size, interference_scale, decay_rate, dt
        )
        
        print(f"MLX output array shape: {output.shape}")
        print(f"Min/Max values: {float(mx.min(output)):.4f}/{float(mx.max(output)):.4f}")

def main():
    """Run all examples."""
    print("Metal Operations Simple Test")
    print("---------------------------")
    
    # Check Metal availability
    if is_metal_available():
        print("Metal is available for acceleration.")
    else:
        print("Metal is not available. Using fallback implementations.")
    
    # Run examples
    example_geometric_activation()
    example_resonance_patterns()
    example_mutuality_field()
    
    print("\nAll examples completed successfully.")

if __name__ == "__main__":
    main()

