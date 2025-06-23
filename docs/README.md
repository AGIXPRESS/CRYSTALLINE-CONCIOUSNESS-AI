# Crystalline MLX - Metal Shader Implementation

## Introduction

Crystalline MLX provides Metal shader implementations for the crystalline consciousness neural network model. These Metal shaders are designed to accelerate the computationally intensive geometric operations on Apple Silicon hardware, leveraging the Metal Performance Shaders (MPS) and Apple's MLX framework.

The implementation focuses on three core operations:
- Geometric activations based on Platonic solids (tetrahedron, cube, dodecahedron, icosahedron)
- Resonance patterns with golden ratio harmonics
- Mutuality field interference patterns and persistence

## Directory Structure

```
crystalline_mlx/
├── Shaders/                  # Metal shader implementations
│   ├── GeometricActivation.metal   # Platonic solid activations
│   ├── ResonancePatterns.metal     # Resonance pattern calculations
│   └── MutualityField.metal        # Field interference patterns
├── Python/                   # Python interface to Metal shaders
│   └── metal_ops.py          # Python API for Metal operations
├── Tests/                    # Test and example scripts
│   ├── test_metal_ops.py     # Comprehensive tests and benchmarks
│   └── test_simple.py        # Simple usage examples
└── README.md                 # This file
```

## Usage

To use the Metal shader implementations in your code:

```python
# Import the Metal operations
from crystalline_mlx.Python.metal_ops import (
    geometric_activation,
    apply_resonance,
    mutuality_field
)

# 1. Geometric Activation
# Apply tetrahedron activation to your tensor
output = geometric_activation(input_tensor, "tetrahedron")

# 2. Resonance Patterns
# Apply resonance patterns with golden ratio harmonics
output = apply_resonance(
    input_tensor,
    frequencies,
    decay_rates,
    amplitudes,
    phase_embedding
)

# 3. Mutuality Field
# Apply mutuality field with interference patterns
output = mutuality_field(
    input_tensor,
    grid_size=16,
    interference_scale=1.0,
    decay_rate=0.05,
    dt=0.1
)
```

The Metal operations work with both PyTorch tensors (with MPS device) and MLX arrays. If Metal is not available, the operations will fall back to CPU implementations.

## Requirements

- macOS with Apple Silicon hardware (M1/M2/M3)
- Python 3.9+
- PyTorch 2.0+ with MPS support or MLX framework
- Metal compatible macOS version (macOS 12+)

For MLX support:
```bash
pip install mlx
```

For PyTorch with MPS support:
```bash
pip install torch
```

## Testing

Run the simple test examples to verify your setup:

```bash
python crystalline_mlx/Tests/test_simple.py
```

This will demonstrate the basic usage of all three Metal operations with both PyTorch and MLX (if available).

