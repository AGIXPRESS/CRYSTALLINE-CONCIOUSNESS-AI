# Crystalline Consciousness MLX Testing: Runtime Configuration Guide

This guide serves as an entry point for configuring the runtime environment for MLX testing in the Crystalline Consciousness project. It provides an overview of environment-specific configurations, resource management, performance monitoring, and testing customization.

## Quick Start

For basic configuration:

```bash
# Development environment
export MLX_DEBUG=1
export MLX_LOG_LEVEL=DEBUG
export MLX_VISUALIZE=1

# CI environment
export MLX_CI_MODE=1
export MLX_PERF_METRICS=1

# Production environment
export MLX_PRODUCTION=1
export MLX_MAX_MEMORY_GB=4
```

## Documentation Structure

For comprehensive documentation, please refer to the following resources:

### Environment Configuration

- [MLX Test Runtime Configuration](testing/mlx_runtime_config.md) - Detailed environment-specific configurations in YAML format
  - Development environment settings
  - CI environment settings
  - Production environment settings

### Resource Management

- [Memory Configuration](testing/mlx_runtime_config.md#1-memory-configuration) - Memory allocation strategies
- [Processor Allocation](testing/mlx_runtime_config.md#2-processor-allocation) - CPU and thread management

### Test Execution

- [MLX Test Execution Guide](testing/mlx_test_execution_guide.md) - Detailed test execution instructions
- [MLX Test Quick Reference](testing/mlx_test_quickref.md) - Common commands and troubleshooting

### Configuration Tools

Several tools are available in the `tools/` directory to help with environment setup and validation:

- `tools/environment_setup.py` - Detects and configures the test environment
- `tools/validate_environment.py` - Validates environment setup before testing
- `tools/verify_resources.py` - Verifies resource availability
- `tools/runtime_adjustment.py` - Adjusts test parameters based on environment
- `tools/resource_scaling.py` - Scales resource allocation based on environment

## Environment Detection

The testing framework automatically detects the current environment based on environment variables and system characteristics. You can explicitly set the environment:

```bash
# Set environment type
export MLX_ENVIRONMENT=development  # Or "ci" or "production"
```

## Test Configuration Files

Environment-specific configurations are stored in YAML files:

- `config/environments/dev.yaml` - Development environment configuration
- `config/environments/ci.yaml` - CI environment configuration
- `config/environments/prod.yaml` - Production environment configuration

## Common Issues and Troubleshooting

For common issues and troubleshooting, refer to:

- [MLX Test Quick Reference: Common Issues](testing/mlx_test_quickref.md#common-issues)
- [Troubleshooting Guide](troubleshooting_guide.md)

---

For detailed implementation examples and advanced configuration options, please refer to the comprehensive documentation in the [testing/](testing/) directory.

---

## 1. Environment-Specific Configurations

### 1.1 Development Environment

In the development environment, detailed logging and visualization tools are essential for debugging and understanding the behavior of Metal operations.

#### Debug Logging Configuration

Enable debug logging in Python by setting environment variables:

```bash
# Enable debug logging for Python tests
export MLX_DEBUG=1
export MLX_LOG_LEVEL=DEBUG

# Run tests with debug output
python tests/test_metal_ops.py
```

For Swift-based tests, edit the scheme settings to include environment variables:

```swift
// In MetalTest.swift, check for debug environment
let debugMode = ProcessInfo.processInfo.environment["MLX_DEBUG"] == "1"
if debugMode {
    print("Running in debug mode with extended logging")
}
```

#### Visualization Support

Install additional libraries for visualization:

```bash
# Install visualization libraries
pip install matplotlib jupyter

# For real-time visualization during test runs
export MLX_VISUALIZE=1
```

Example of visualization code (already in test_metal_ops.py):

```python
if os.environ.get('MLX_VISUALIZE', '0') == '1':
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(result[0].reshape(grid_size, grid_size))
    plt.title('Mutual Field Output')
    plt.colorbar()
    plt.savefig('test_results/mutuality_visualization.png')
```

#### Command-line Options

Add custom parameters to test scripts for development-specific settings:

```bash
# Run with specific development options
python tests/test_metal_ops.py --debug --visualize --iterations 3
```

Implement these flags in your test scripts:

```python
parser = argparse.ArgumentParser(description="Test Metal shader operations.")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
parser.add_argument("--iterations", type=int, default=10, help="Number of iterations for benchmarks")
args = parser.parse_args()

if args.debug:
    logging.basicConfig(level=logging.DEBUG)
    print("Debug logging enabled")
```

### 1.2 Continuous Integration (CI) Environment

CI environments prioritize performance metrics and minimal but essential logging.

#### Minimal Logging Configuration

```bash
# CI environment settings
export MLX_LOG_LEVEL=INFO
export MLX_CI_MODE=1
export MLX_PERF_METRICS=1
```

#### Performance Metrics Collection

Create a metrics collection script:

```python
# performance_metrics.py

import json
import time
import os
from pathlib import Path

class PerformanceTracker:
    def __init__(self, test_name, output_dir="test_results"):
        self.test_name = test_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.metrics = {
            "timestamp": time.time(),
            "test_name": test_name,
            "operations": {}
        }
    
    def record_operation(self, op_name, batch_size, input_dim, 
                        cpu_time=0, mps_time=0, mlx_time=0, 
                        cpu_memory=0, mps_memory=0, mlx_memory=0):
        op_key = f"{op_name}_{batch_size}_{input_dim}"
        self.metrics["operations"][op_key] = {
            "batch_size": batch_size,
            "input_dim": input_dim,
            "cpu_time": cpu_time,
            "mps_time": mps_time,
            "mlx_time": mlx_time,
            "speedup_mps": cpu_time / mps_time if mps_time > 0 else 0,
            "speedup_mlx": cpu_time / mlx_time if mlx_time > 0 else 0,
            "cpu_memory": cpu_memory,
            "mps_memory": mps_memory,
            "mlx_memory": mlx_memory
        }
    
    def save(self):
        output_file = self.output_dir / f"{self.test_name}_{int(time.time())}.json"
        with open(output_file, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Performance metrics saved to {output_file}")
```

Usage in CI scripts:

```python
# In your test script
tracker = PerformanceTracker("geometric_activation_test")

# After running benchmarks
tracker.record_operation(
    "geometric", batch_size, input_dim, 
    cpu_time=avg_cpu_time, 
    mps_time=avg_mps_time, 
    mlx_time=avg_mlx_time
)

# Save results
tracker.save()
```

#### Historical Performance Comparison

Set up a historical performance comparison script:

```python
# compare_performance.py

import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def compare_performance(test_name, operation, metric="speedup_mlx"):
    """Compare performance metrics across test runs"""
    results_dir = Path("test_results")
    files = sorted(glob.glob(f"{results_dir}/{test_name}_*.json"))
    
    data = []
    timestamps = []
    
    for file in files:
        with open(file, "r") as f:
            metrics = json.load(f)
            
        for op_key, op_metrics in metrics["operations"].items():
            if operation in op_key and metric in op_metrics:
                data.append(op_metrics[metric])
                timestamps.append(metrics["timestamp"])
    
    # Plot historical performance
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, data, marker='o')
    plt.title(f"Historical {metric} for {operation}")
    plt.ylabel(metric)
    plt.xlabel("Test Run")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/history_{test_name}_{operation}_{metric}.png")
    plt.close()

# Example usage
compare_performance("geometric_activation_test", "geometric_8_256", "speedup_mlx")
```

### 1.3 Production Environment

For production environments, optimize for performance and resource efficiency.

#### Resource Optimization

Configure resource limits in production:

```bash
# Production environment settings
export MLX_LOG_LEVEL=WARNING
export MLX_PRODUCTION=1
export MLX_MAX_MEMORY_GB=4
export MLX_CONCURRENT_COMPILATIONS=2
```

Implement these settings in your code:

```python
# In metal_manager.py or similar
import os

def get_memory_limit():
    """Get the memory limit for Metal allocations"""
    if os.environ.get("MLX_PRODUCTION", "0") == "1":
        mem_limit_gb = float(os.environ.get("MLX_MAX_MEMORY_GB", "4"))
        return int(mem_limit_gb * 1024 * 1024 * 1024)  # Convert to bytes
    return 0  # Unlimited

def get_concurrent_compilations():
    """Get the number of concurrent shader compilations allowed"""
    if os.environ.get("MLX_PRODUCTION", "0") == "1":
        return int(os.environ.get("MLX_CONCURRENT_COMPILATIONS", "2"))
    return 4  # Default for development
```

#### Containerized Deployment

Docker configuration example:

```dockerfile
# Dockerfile for Crystalline Consciousness MLX
FROM python:3.9-slim

# Install dependencies
RUN pip install --no-cache-dir numpy torch mlx matplotlib

# Set environment variables for production
ENV MLX_LOG_LEVEL=WARNING
ENV MLX_PRODUCTION=1
ENV MLX_MAX_MEMORY_GB=4
ENV MLX_CONCURRENT_COMPILATIONS=2

# Copy project files
COPY . /app
WORKDIR /app

# Run tests in container
CMD ["python", "tests/test_metal_ops.py", "--benchmark"]
```

---

## 2. Hardware Resource Management

### 2.1 Metal Device Detection and Validation

Detect and validate Metal devices using the following code pattern:

```python
# In metal_manager.py
def is_metal_available():
    """Check if Metal is available on this system"""
    try:
        # Import Metal-related modules
        import Metal
        import Foundation
        
        # Try to get the default device
        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            print("No Metal device found")
            return False
            
        # Log device information
        print(f"Metal device found: {device.name()}")
        return True
    except ImportError:
        print("Metal framework not available")
        return False
```

In Swift:

```swift
// In MetalTest.swift
func checkMetalAvailability() -> Bool {
    guard let device = MTLCreateSystemDefaultDevice() else {
        print("Error: No Metal device found")
        return false
    }
    
    print("Metal device: \(device.name)")
    
    // Check if device supports features we need
    let supportsNonuniformThreadgroups = device.supportsFeatureSet(.iOS_GPUFamily4_v1)
    print("Supports non-uniform threadgroups: \(supportsNonuniformThreadgroups)")
    
    return true
}
```

### 2.2 Memory Allocation Guidelines

Memory allocation recommendations for different grid sizes:

| Grid Size | Operation Type | Recommended Memory | Buffer Count | Notes |
|-----------|----------------|-------------------|--------------|-------|
| 8 (small) | Geometric | 32 MB | 3-5 buffers | Suitable for testing and development |
| 8 (small) | Resonance | 64 MB | 5-7 buffers | Additional buffers for harmonics |
| 8 (small) | Mutuality | 128 MB | 8-10 buffers | More complex, needs more intermediate buffers |
| 16 (medium) | Geometric | 128 MB | 3-5 buffers | Default for most use cases |
| 16 (medium) | Resonance | 256 MB | 5-7 buffers | Default for resonance patterns |
| 16 (medium) | Mutuality | 512 MB | 8-10 buffers | Recommended for standard mutuality fields |
| 32 (large) | Geometric | 512 MB | 3-5 buffers | High-detail geometry operations |
| 32 (large) | Resonance | 1 GB | 5-7 buffers | High-fidelity resonance patterns |
| 32 (large) | Mutuality | 2 GB | 8-10 buffers | Production-level detailed fields |

Implementation example:

```python
def allocate_buffers(batch_size, input_dim, grid_size, operation_type):
    """Allocate appropriate buffers based on operation and grid size"""
    buffer_sizes = {
        "geometric": {
            8: 32 * 1024 * 1024,  # 32 MB
            16: 128 * 1024 * 1024,  # 128 MB
            32: 512 * 1024 * 1024,  # 512 MB
        },
        "resonance": {
            8: 64 * 1024 * 1024,  # 64 MB
            16: 256 * 1024 * 1024,  # 256 MB
            32: 1024 * 1024 * 1024,  # 1 GB
        },
        "mutuality": {
            8: 128 * 1024 * 1024,  # 128 MB
            16: 512 * 1024 * 1024,  # 512 MB
            32: 2 * 1024 * 1024 * 1024,  # 2 GB
        }
    }
    
    # Get the recommended buffer size
    buffer_size = buffer_sizes.get(operation_type, {}).get(grid_size, 128 * 1024 * 1024)
    
    # Log allocation information
    print(f"Allocating buffers for {operation_type} operation with grid size {grid_size}")
    print(f"Buffer size: {buffer_size / (1024 * 1024):.2f} MB")
    
    # Return the buffer size
    return buffer_size
```

### 2.3 Buffer Management

Efficient buffer management strategies:

#### Pre-allocation and Reuse

```python
# In metal_ops.py or fix_metal_ops.py

# Global buffer cache
_buffer_cache = {}

def get_or_create_buffer(shape, dtype, name=None):
    """Get a buffer from cache or create a new one"""
    key = (shape, dtype, name)
    if key in _buffer_cache:
        return _buffer_cache[key]
    
    # Create a new buffer
    if HAS_MLX:
        buffer = mx.zeros(shape, dtype=dtype)
    elif torch.backends.mps.is_available():
        buffer = torch.zeros(shape, dtype=dtype, device="mps")
    else:
        buffer = torch.zeros(shape, dtype=dtype)
    
    # Cache the buffer
    _buffer_cache[key] = buffer
    return buffer

def clear_buffer_cache():
    """Clear the buffer cache"""
    global _buffer_cache
    _buffer_cache.clear()
```

#### Example for Mutuality Field Operation

For the mutuality field operation, which requires multiple intermediate buffers:

```python
def mutuality_field(x, grid_size, interference_scale=1.0, decay_rate=0.05, dt=0.1):
    """Apply mutuality field operation with efficient buffer management"""
    batch_size, input_dim = x.shape
    
    # Get or create intermediate buffers
    grid = get_or_create_buffer((batch_size, grid_size, grid_size), x.dtype, "mutuality_grid")
    field = get_or_create_buffer((batch_size, grid_size, grid_size), x.dtype, "mutuality_field")
    persistence = get_or_create_buffer((batch_size, grid_size, grid_size), x.dtype, "mutuality_persistence")
    
    # Execute operation with managed buffers
    # ... operation implementation ...
    
    return field
```

#### Buffer Cleanup

Add cleanup mechanisms to release resources when tests are complete:

```python
# In test_metal_ops.py
def cleanup():
    """Clean up resources after tests"""
    if 'clear_buffer_cache' in globals():
        clear_buffer_cache()
    
    # Release any other resources
    if 'manager' in globals() and hasattr(manager, 'cleanup'):
        manager.cleanup()
    
    print("Cleaned up Metal resources")

# Register cleanup function
import atexit
atexit.register(cleanup)
```

---

## 3. Performance Monitoring

### 3.1 Standard Benchmark Parameters

For consistent and meaningful performance evaluations, use these standard parameters across all benchmark tests:

#### Recommended Batch Sizes

```python
# Standard batch sizes for benchmarks
BATCH_SIZES = [1, 2, 4, 8, 16]
```

These batch sizes represent different use cases:

* **1**: Single-sample inference (e.g., real-time applications)
* **2-4**: Small batch processing (e.g., interactive applications)
* **8-16**: Larger batch processing (e.g., batch analysis)

#### Recommended Input Dimensions

```python
# Standard input dimensions for benchmarks
INPUT_DIMENSIONS = [64, 128, 256, 512, 1024]
```

These dimensions represent different model sizes:

* **64-128**: Small models (e.g., mobile or edge deployment)
* **256-512**: Medium models (e.g., standard desktop applications)
* **1024+**: Large models (e.g., high-performance workstations)

#### Iteration Counts

For accurate timing measurements, use multiple iterations and take the average:

```python
# Standard iteration count for benchmarks
ITERATIONS = 10

# Example usage
def benchmark_operation(operation_fn, input_data, *args, **kwargs):
    """Benchmark an operation function"""
    warm_up = 2  # Additional warm-up iterations
    times = []
    
    # Warm-up iterations (not timed)
    for _ in range(warm_up):
        _ = operation_fn(input_data, *args, **kwargs)
    
    # Timed iterations
    for _ in range(ITERATIONS):
        start_time = time.time()
        _ = operation_fn(input_data, *args, **kwargs)
        times.append(time.time() - start_time)
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    return {
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "times": times
    }
```

### 3.2 Performance Threshold Definitions

Define performance thresholds to automatically detect performance regressions:

#### Baseline Performance Thresholds

```python
# Performance thresholds by operation type and device
PERFORMANCE_THRESHOLDS = {
    "geometric": {
        "cpu": {
            "max_time": 0.050,  # 50ms maximum for CPU
            "target_time": 0.020  # 20ms target for CPU
        },
        "mps": {
            "max_time": 0.010,  # 10ms maximum for MPS
            "target_time": 0.005  # 5ms target for MPS
        },
        "mlx": {
            "max_time": 0.008,  # 8ms maximum for MLX
            "target_time": 0.003  # 3ms target for MLX
        }
    },
    "resonance": {
        "cpu": {
            "max_time": 0.100,  # 100ms maximum for CPU
            "target_time": 0.040  # 40ms target for CPU
        },
        "mps": {
            "max_time": 0.020,  # 20ms maximum for MPS
            "target_time": 0.010  # 10ms target for MPS
        },
        "mlx": {
            "max_time": 0.015,  # 15ms maximum for MLX
            "target_time": 0.008  # 8ms target for MLX
        }
    },
    "mutuality": {
        "cpu": {
            "max_time": 0.200,  # 200ms maximum for CPU
            "target_time": 0.080  # 80ms target for CPU
        },
        "mps": {
            "max_time": 0.040,  # 40ms maximum for MPS
            "target_time": 0.020  # 20ms target for MPS
        },
        "mlx": {
            "max_time": 0.030,  # 30ms maximum for MLX
            "target_time": 0.015  # 15ms target for MLX
        }
    }
}
```

#### Threshold Adjustment Based on Input Size

```python
def get_adjusted_threshold(operation, device, batch_size, input_dim):
    """Get threshold adjusted for input size"""
    # Get base threshold
    base_threshold = PERFORMANCE_THRESHOLDS.get(operation, {}).get(device, {}).get("max_time", 0.1)
    
    # Adjustment factors
    batch_factor = batch_size / 4.0  # Normalized to batch size 4
    dim_factor = input_dim / 256.0  # Normalized to dimension 256
    
    # Calculate adjusted threshold
    adjusted_threshold = base_threshold * batch_factor * dim_factor
    
    return adjusted_threshold
```

#### Performance Validation

```python
def validate_performance(operation, device, batch_size, input_dim, measured_time):
    """Validate if performance meets thresholds"""
    threshold = get_adjusted_threshold(operation, device, batch_size, input_dim)
    
    if measured_time <= threshold:
        status = "PASS"
    else:
        status = "FAIL"
    
    print(f"Performance {status}: {operation} on {device} with batch={batch_size}, dim={input_dim}")
    print(f"  Measured: {measured_time:.6f}s, Threshold: {threshold:.6f}s")
    
    return status == "PASS"
```

Example usage in test scripts:

```python
# In test_metal_ops.py
for batch_size in BATCH_SIZES:
    for input_dim in INPUT_DIMENSIONS:
        # Create input data
        input_data = torch.randn(batch_size, input_dim)
        
        # Run benchmark
        results = benchmark_operation(geometric_activation, input_data, "tetrahedron", [0.3])
        
        # Validate performance
        validate_performance("geometric", "cpu", batch_size, input_dim, results["avg_time"])
```

---

## 4. Environment Detection and Validation

### 4.1 Metal Availability Checking

Implement robust Metal availability checking to ensure proper fallback to CPU when necessary:

#### Simple Metal Availability Check

```python
# In metal_ops.py
def is_metal_available():
    """Check if Metal is available on this system"""
    try:
        # Check MLX availability
        import mlx.core
        return True
    except ImportError:
        pass
    
    try:
        # Check PyTorch MPS availability
        import torch
        return torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        pass
    
    return False
```

#### Enhanced Metal Device Validation

```python
def validate_metal_device():
    """Validate Metal device capabilities"""
    if not is_metal_available():
        print("Metal is not available")
        return False
    
    try:
        # More detailed check with MLX
        import mlx.core as mx
        
        # Test a simple operation to verify Metal works
        x = mx.array([1.0, 2.0, 3.0])
        y = mx.exp(x)
        _ = float(mx.sum(y))  # Force execution
        
        print("MLX Metal validation successful")
        return True
    except Exception as e:
        print(f"MLX Metal validation failed: {e}")
        
    try:
        # Try PyTorch MPS
        import torch
        if torch.backends.mps.is_available():
            # Test a simple operation
            x = torch.tensor([1.0, 2.0, 3.0], device="mps")
            y = torch.exp(x)
            _ = y.sum().item()  # Force execution
            
            print("PyTorch MPS validation successful")
            return True
    except Exception as e:
        print(f"PyTorch MPS validation failed: {e}")
    
    return False
```

### 4.2 Fallback Mechanisms

Implement graceful fallbacks when Metal acceleration is unavailable:

#### Operation Fallback

```python
def geometric_activation(x, solid_type, coefficients=None):
    """Geometric activation with automatic fallback"""
    # Try Metal implementation first
    try:
        if is_metal_available():
            # Import based on available frameworks
            if 'mx' in globals() or 'mlx' in sys.modules:
                return _geometric_activation_mlx(x, solid_type, coefficients)
            elif hasattr(torch, 'backends') and torch.backends.mps.is_available():
                return _geometric_activation_mps(x, solid_type, coefficients)
    except Exception as e:
        print(f"Metal implementation failed: {e}, falling back to CPU")
    
    # CPU fallback
    return _geometric_activation_cpu(x, solid_type, coefficients)
```

#### Test Fallback

```python
def run_test(test_fn, *args, **kwargs):
    """Run a test with fallback for non-Metal environments"""
    if is_metal_available():
        print(f"Running {test_fn.__name__} with Metal acceleration")
        return test_fn(*args, **kwargs)
    else:
        print(f"Metal not available, running {test_fn.__name__} in CPU-only mode")
        # Set environment variable to indicate CPU-only mode
        os.environ["MLX_CPU_ONLY"] = "1"
        result = test_fn(*args, **kwargs)
        # Clean up environment
        os.environ.pop("MLX_CPU_ONLY", None)
        return result
```

### 4.3 Shader Library Validation

Validate shader libraries before running tests:

#### Shader Path Validation

This code from `test_shader_manager.py` demonstrates how to locate and validate shader files:

```python
def validate_shader_paths():
    """Validate shader paths and files"""
    # Define possible shader directory locations
    project_root = Path(__file__).resolve().parent.parent
    shader_locations = [
        os.path.join(project_root, "shaders"),                          # /shaders
        os.path.join(project_root, "src", "shaders"),                   # /src/shaders
        os.path.join(os.path.dirname(__file__), "..", "shaders"),       # relative ../shaders
        os.path.join(os.path.dirname(__file__), "shaders"),             # relative ./shaders
    ]
    
    # Find shader directory
    shader_dir = None
    for path in shader_locations:
        if os.path.exists(path) and os.path.isdir(path):
            shader_dir = path
            break
```

#### Pipeline Validation

After locating shader files, validate that the required compute pipelines can be created:

```python
def validate_pipelines():
    """Validate required compute pipelines"""
    # Check that manager is initialized
    if not hasattr(manager, 'device') or manager.device is None:
        print("ERROR: Metal device not initialized")
        return False
    
    # Define required pipelines for each shader library
    required_pipelines = {
        "geometric": [
            "tetrahedron_activation", 
            "cube_activation", 
            "dodecahedron_activation", 
            "icosahedron_activation"
        ],
        "resonance": ["apply_resonance"],
        "mutuality": [
            "reshape_to_grid", 
            "calculate_mutual_field", 
            "apply_persistence"
        ]
    }
    
    # Check each required pipeline
    validation_status = True
    for library, functions in required_pipelines.items():
        for func in functions:
            pipeline_key = f"{library}_{func}"
            if pipeline_key not in manager.pipelines:
                print(f"ERROR: Missing pipeline {pipeline_key}")
                validation_status = False
    
    return validation_status
```

---

## 5. Test Customization Settings

### 5.1 Test Types

Configure different test types to focus on specific aspects of the system:

#### Unit Tests

For testing individual operations in isolation:

```bash
# Run all unit tests
python -m pytest tests/test_*.py -v

# Run specific operation tests
python -m pytest tests/test_geometric.py -v
python -m pytest tests/test_resonance.py -v
python -m pytest tests/test_mutuality.py -v
```

#### Integration Tests

For testing operations in the context of a complete model:

```bash
# Run integration tests
python -m pytest tests/test_integration.py -v

# Run full model tests
python tests/test_in_full_model.py
```

#### Performance Tests

For benchmarking and performance evaluation:

```bash
# Run benchmarks
python tests/test_metal_ops.py --benchmark

# Run with specific parameters
python tests/test_metal_ops.py --benchmark --batch-sizes 1,4,16 --input-dims 128,512
```

### 5.2 Logging and Visualization

Configure logging levels and visualization options for different test scenarios:

#### Logging Configuration

```python
import logging
import os

def configure_logging():
    """Configure logging based on environment variables"""
    # Get log level from environment or use default
    log_level_name = os.environ.get("MLX_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Console handler
            logging.FileHandler("test_results/test_run.log")  # File handler
        ]
    )
    
    # Log configuration
    logging.info(f"Logging configured with level: {log_level_name}")
    
    return logging.getLogger("mlx_tests")
```

#### Visualization Settings

```python
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def configure_visualization():
    """Configure visualization settings"""
    # Check if visualization is enabled
    visualize = os.environ.get("MLX_VISUALIZE", "0") == "1"
    
    # Create output directory if needed
    if visualize:
        Path("test_results").mkdir(exist_ok=True)
    
    return visualize

def visualize_result(result, name, grid_size=None):
    """Visualize a result tensor"""
    if not configure_visualization():
        return
    
    plt.figure(figsize=(10, 6))
    
    # Handle different result types
    if grid_size is not None:
        # For grid-based results (e.g., mutuality field)
        plt.imshow(result[0].reshape(grid_size, grid_size))
        plt.colorbar()
        plt.title(f"{name} Output (Grid Size: {grid_size})")
    else:
        # For generic results
        plt.plot(result[0].flatten())
        plt.title(f"{name} Output")
    
    # Save visualization
    plt.savefig(f"test_results/{name.lower().replace(' ', '_')}.png")
    plt.close()
```

### 5.3 Error Tolerance Thresholds

Define error tolerance thresholds for comparing results across different implementations:

```python
# Default error tolerance thresholds
ERROR_TOLERANCES = {
    "geometric": {
        "rtol": 1e-3,  # Relative tolerance
        "atol": 1e-3   # Absolute tolerance
    },
    "resonance": {
        "rtol": 1e-3,
        "atol": 1e-3
    },
    "mutuality": {
        "rtol": 1e-2,  # Higher tolerance due to parallel reduction differences
        "atol": 1e-2
    }
}

def compare_results(cpu_result, metal_result, operation_type):
    """Compare results with appropriate tolerances"""
    # Get tolerances for this operation
    tolerances = ERROR_TOLERANCES.get(operation_type, {"rtol": 1e-3, "atol": 1e-3})
    rtol = tolerances["rtol"]
    atol = tolerances["atol"]
    
    # Convert to numpy arrays if needed
    if hasattr(cpu_result, "numpy"):
        cpu_result = cpu_result.numpy()
    if hasattr(metal_result, "numpy"):
        metal_result = metal_result.numpy()
    
    # Calculate differences
    max_diff = np.max(np.abs(cpu_result - metal_result))
    mean_diff = np.mean(np.abs(cpu_result - metal_result))
    
    # Check if within tolerance
    is_close = np.allclose(cpu_result, metal_result, rtol=rtol, atol=atol)
    
    print(f"Results comparison for {operation_type}:")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Within tolerance: {is_close}")
    
    return is_close
```

Example usage:

```python
# Run operation with both CPU and Metal
cpu_result = geometric_activation(cpu_input, "tetrahedron", [0.3])
metal_result = geometric_activation(metal_input, "tetrahedron", [0.3])

# Compare results
is_valid = compare_results(cpu_result, metal_result, "geometric")
```

---

## 6. Troubleshooting

This section addresses common issues encountered when running the MLX tests and provides solutions to resolve them.

### 6.1 Python Syntax and Indentation Errors

#### Error: Unexpected Indentation

```
E     File "/Users/okok/crystalineconciousnessai/src/python/metal_ops.py", line 302
E       def _geometric_activation_fallback(x, solid_type, coefficients):
E   IndentationError: unexpected unindent
```

**Solution:**

Fix indentation in the identified Python file. Ensure consistent use of spaces or tabs:

```bash
# Check and fix indentation in the affected file
python -m pycodestyle src/python/metal_ops.py

# Or manually edit the file to fix indentation
nano src/python/metal_ops.py
```

#### Error: Syntax Error in Python File

```
SyntaxError: invalid syntax
```

**Solution:**

Check for syntax errors in the Python file. Common issues include missing parentheses, quotes, or colons:

```bash
# Validate Python syntax without executing
python -m py_compile src/python/metal_ops.py
```

### 6.2 Module Import Path Issues

#### Error: No Module Named 'Python'

```
ModuleNotFoundError: No module named 'Python'
```

**Solution:**

Fix import paths. The error occurs when Python can't find the module in its search path:

```python
# Incorrect import
from Python.metal_manager_updated import get_shader_manager

# Correct import (use relative path)
from src.python.metal_manager_updated import get_shader_manager

# Or add path to PYTHONPATH
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
```

For command-line execution, set the PYTHONPATH environment variable:

```bash
# Set PYTHONPATH to include the project root
export PYTHONPATH=/Users/okok/crystalineconciousnessai:$PYTHONPATH

# Or run with modified path
PYTHONPATH=/Users/okok/crystalineconciousnessai python tests/test_shader_manager.py
```

#### Error: Module Not Found - metal_ops

```
Warning: metal_ops module not found. Tests will be skipped.
```

**Solution:**

Ensure the module exists and is in the Python path:

```bash
# Check if the file exists
ls -la src/python/metal_ops.py

# Add source directory to Python path
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### 6.3 Metal Framework Availability

#### Error: PyObjC Metal Framework Not Found

```
UserWarning: PyObjC Metal framework not found. GPU acceleration unavailable.
```

**Solution:**

Install the PyObjC Metal framework:

```bash
# Install PyObjC with Metal support
pip install pyobjc-framework-Metal pyobjc-framework-Cocoa

# For older versions of macOS/PyObjC, you might need:
pip install pyobjc
```

#### Error: No Metal Device Found

```
Error: No Metal device found
```

**Solution:**

Verify Metal is available on your system:

```bash
# Check Metal device info
system_profiler SPDisplaysDataType | grep "Metal:"

# Check if your Mac supports Metal
system_profiler SPHardwareDataType
```

For M1/M2/M3 Macs, ensure you're running Python natively and not under Rosetta:

```bash
# Check if Python is running natively
file $(which python3)
# Should display "Mach-O 64-bit executable arm64" for native ARM
```

### 6.4 MLX and MPS Configuration

#### Error: MLX not available

```
Warning: MLX not available. Metal execution will be skipped.
```

**Solution:**

Install MLX:

```bash
# Install MLX
pip install mlx

# Verify installation
python -c "import mlx; print(mlx.__version__)"
```

#### Error: PyTorch MPS not available

```
Warning: PyTorch MPS not available.
