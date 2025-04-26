# Crystalline Consciousness AI

## Overview

Crystalline Consciousness AI is a neural network architecture inspired by crystalline structures and geometric principles found in nature. This project implements a unique computational model that leverages geometric operations, resonance patterns, and mutuality field interference for advanced machine learning tasks. The implementation features Metal shader acceleration for Apple Silicon hardware.

## Key Features

- **Geometric Activations**: Neural activations based on Platonic solids (tetrahedron, cube, dodecahedron, icosahedron)
- **Resonance Patterns**: Signal processing using golden ratio harmonics and phase interactions
- **Mutuality Field**: Emergent pattern formation through grid-based interference
- **Metal Acceleration**: Hardware acceleration on Apple Silicon using Metal shaders

## Directory Structure

The project is organized into the following directories:

```
crystalineconciousnessai/
├── src/                    # Source code
│   ├── python/             # Python implementation of core algorithms
│   ├── geometry/           # Geometric primitive definitions
│   ├── layers/             # Neural network layer implementations
│   ├── model/              # Model architecture definitions
│   ├── metal/              # Metal-specific implementations
│   └── utils/              # Utility functions and helpers
├── shaders/                # Metal shader implementations
│   ├── GeometricActivation.metal   # Platonic solid activations
│   ├── ResonancePatterns.metal     # Resonance pattern calculations
│   └── MutualityField.metal        # Field interference patterns
├── tests/                  # Test suites and benchmarks
│   ├── test_metal_ops.py   # Comprehensive tests for Metal operations
│   ├── test_simple.py      # Simple usage examples
│   └── test_results/       # Performance benchmarks and visualizations
├── docs/                   # Documentation
│   ├── README.md           # General documentation
│   ├── INTEGRATION.md      # Integration guidelines
│   └── SUMMARY.md          # Project summary
├── examples/               # Example applications
└── utils/                  # Additional utilities
```

## Core Components

### Metal Shaders (`shaders/`)

The Metal shader implementations provide hardware acceleration for the computationally intensive geometric operations:

- **GeometricActivation.metal**: Implements activation functions based on Platonic solids
- **ResonancePatterns.metal**: Computes resonance patterns with golden ratio harmonics
- **MutualityField.metal**: Implements field interference patterns and persistence

### Python Interface (`src/python/`)

Python interfaces to the Metal shaders, providing a simple API for use in machine learning models:

- **metal_ops.py**: Main API for interacting with Metal operations
- **metal_manager.py**: Manages Metal resources and shader compilation

## Usage

To use the Metal shader implementations in your code:

```python
# Import the Metal operations
from crystallineconciousnessai.src.python.metal_ops import (
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

## Installation

Clone the repository and add it to your Python path:

```bash
git clone https://github.com/yourusername/crystalineconciousnessai.git
cd crystalineconciousnessai
pip install -e .
```

## Testing

Run the test suite to verify your setup:

```bash
python tests/test_simple.py
```

For comprehensive tests and benchmarks:

```bash
python tests/test_metal_ops.py
```

## Contributing

Contributions are welcome! Please see the documentation in the `docs/` directory for guidelines on contributing, code standards, and the development roadmap.

