# Crystalline Consciousness AI - Environment Specification

This document outlines the technical environment requirements and configurations for the Crystalline Consciousness AI framework, with special focus on the Quantum Geometric Nexus implementation.

## System Requirements

### Hardware
- **Recommended**: Apple Silicon Mac (M1/M2/M3 or newer) for Metal GPU acceleration
- **Minimum**: Any macOS, Linux, or Windows system capable of running Python 3.9+
- **Memory**: 16GB RAM recommended (8GB minimum)
- **Storage**: 20GB free space (including 4GB for training data)

### Operating System
- **Recommended**: macOS 15.4.1 or newer (optimized for Metal Performance Shaders)
- **Supported**: macOS 12+, Linux, Windows 10/11 (with reduced GPU acceleration capabilities)

## Software Environment

### Python
- **Version**: Python 3.9.19 
- **Virtual Environment**: Recommended for isolation and dependency management

### Key Dependencies
- **MLX**: version 0.25.1
  - Core tensor computation library with Metal acceleration for Apple Silicon
  - Enables GPU-accelerated math operations for quantum geometric processing
  
- **NumPy**: version 2.0.2
  - Provides fallback CPU computation when Metal acceleration is unavailable
  - Used for numeric tensor operations and mathematical functions
  
- **Matplotlib**: For visualization of resonance patterns and quantum geometric fields
- **Pillow**: For image processing and data augmentation
- **PyPDF2**: For PDF parsing in the unified data loader
- **Pandas**: For structured data handling and CSV processing
- **svglib**: For SVG processing in the unified data loader
- **markdown**: For Markdown processing in the unified data loader

## GPU Acceleration

### Metal Performance Shaders (MPS)
The framework heavily utilizes Apple's Metal Performance Shaders (MPS) through the MLX library for GPU acceleration on Apple Silicon devices.

Key acceleration features:
- **Tensor Operations**: All core tensor operations accelerated by Metal
- **φ-Resonant Filters**: Custom Metal shaders for phi-resonant geometric processing
- **Quantum-Like Wave Functions**: Optimized wave function operations through MPS
- **Platonic Solid Activations**: GPU-accelerated geometric activation functions

### Acceleration Detection
The framework automatically detects Metal availability and configures itself accordingly:
```python
# MLX import and configuration for Metal acceleration
try:
    import mlx
    import mlx.core as mx
    
    # Test MLX GPU availability
    def test_mlx_gpu():
        try:
            # Simple test operation to verify GPU access
            x = mx.ones((10, 10))
            y = mx.sum(x)
            y.item()  # Force computation
            print("MLX GPU acceleration is available")
            return True
        except Exception as e:
            print(f"MLX GPU acceleration test failed: {e}")
            return False

    HAS_MLX = test_mlx_gpu()
except ImportError:
    print("MLX not available. Falling back to NumPy only.")
    HAS_MLX = False
```

## Environment Setup

### Using venv (Recommended)
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Using pip directly
```bash
pip3 install mlx==0.25.1 numpy==2.0.2 matplotlib pillow pypdf2 pandas svglib markdown
```

### Validation
Run the environment validation script to confirm proper setup:
```bash
python3 environment_check.py
```

The validation script checks:
1. Python version compatibility
2. Required package availability and versions
3. MLX installation and Metal acceleration detection
4. Basic tensor operation benchmarks
5. Memory availability for training

## Troubleshooting

### Common Metal/MLX Issues
- **Metal acceleration unavailable**: Ensure you're using an Apple Silicon Mac
- **MLX installation fails**: Try installing with `pip install -U --pre mlx`
- **Memory errors during processing**: Reduce batch size in config.json

### Python Environment Issues
- **ImportError**: Ensure all dependencies are installed
- **Version conflicts**: Use a fresh virtual environment
- **Module not found**: Check PYTHONPATH includes the project root

## Performance Notes

- **Metal Acceleration**: Provides 5-10x speedup on supported hardware
- **Memory Usage**: Phi-resonant operations require approximately 2.5x the memory of standard operations
- **Batch Processing**: For optimal performance, use batch sizes that are multiples of 8 (e.g., 8, 16, 32)
- **Cache Benefits**: Using the resonance field cache can reduce processing time by up to 60%

