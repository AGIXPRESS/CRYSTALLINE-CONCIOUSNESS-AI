# Developer Guide

This guide provides essential information for developers working on the Crystalline Consciousness AI framework, covering development workflow, testing guidelines, and memory management best practices.

## Development Workflow

### Setup and Environment

1. **Environment Requirements**:
   - Python 3.9+
   - MLX 0.25.1+ (for Metal acceleration)
   - NumPy 2.0.2+ (for CPU operations)
   - PyTorch (optional, for additional acceleration paths)

2. **Directory Structure**:
   ```
   crystalineconciousnessai/
   ├── crystal/                  # Core framework
   │   ├── __init__.py
   │   ├── activations.py        # Geometric activation gates
   │   ├── holography.py         # Holographic encoding
   │   ├── resonance.py          # Resonance patterns
   │   └── soc.py                # Self-organizing criticality
   ├── src/
   │   ├── python/               # Low-level operations
   │   │   ├── metal_ops.py      # Metal operations
   │   │   └── metal_manager.py  # Metal shader management
   │   └── shaders/              # Metal compute shaders
   ├── tests/                    # Test suite
   │   └── test_platonic_gates.py
   └── docs/                     # Documentation
   ```

3. **Development Environment Setup**:
   ```bash
   # MLX installation
   pip install mlx==0.25.1
   
   # Required packages
   pip install numpy==2.0.2
   ```

### Coding Standards

1. **Style Guide**:
   - Follow PEP 8 for Python code style
   - Use docstrings (Google style) for all functions and classes
   - Include type hints for function parameters and return values

2. **Code Organization**:
   - Place activation gates in `crystal/activations.py`
   - Low-level Metal operations in `src/python/metal_ops.py`
   - Metal shaders in `src/shaders/`
   - Tests in `tests/` with matching file names (e.g., `test_activations.py`)

3. **Commit Guidelines**:
   - Use descriptive commit messages
   - Reference issue numbers when applicable
   - Commit atomic changes rather than large batches

### Adding New Components

1. **Adding a New Activation Gate**:
   - Create a new class inheriting from `PlatonicGate`
   - Implement `_forward_cpu` and `_backward_cpu` methods
   - Add to `GATE_REGISTRY` using `register_gate()`
   - Create corresponding tests in `tests/`

2. **Adding Metal Acceleration**:
   - Create Metal shader in `src/shaders/`
   - Add shader loading in `metal_ops.py`
   - Implement acceleration function in `metal_ops.py`
   - Add fallback implementation for CPU

## Testing Guidelines

### Test Structure

1. **Test Hierarchy**:
   - Unit tests for individual gates and functions
   - Integration tests for gate sequences
   - Metal/MLX acceleration tests
   - Memory management tests

2. **Standard Test Suite**:
   - `test_gate_creation`: Verifies proper gate instantiation
   - `test_metal_integration`: Tests Metal acceleration
   - `test_phi_resonance`: Verifies phi-ratio pattern preservation
   - `test_element_dynamics`: Tests element-specific properties
   - `test_memory_management`: Verifies proper buffer cleanup

3. **Test Data Patterns**:
   - Phi-ratio frequency patterns
   - Element-specific test patterns
   - Zero input handling
   - Gradient and alternating patterns

### Running Tests

```bash
# Run all tests
python -m unittest discover -v

# Run specific test file
python -m unittest tests/test_platonic_gates.py

# Run specific test case
python -m unittest tests.test_platonic_gates.TestPlatonicGates.test_phi_resonance
```

### Test Best Practices

1. **Test Coverage**:
   - Aim for at least 80% code coverage
   - Test both CPU and Metal paths
   - Test with different tensor types (MLX, NumPy, PyTorch)
   - Test memory management explicitly

2. **Test Assertions**:
   - Use specific assertions (e.g., `assertGreater` instead of `assertTrue`)
   - Include descriptive messages in assertions
   - Check numerical properties within reasonable tolerances

3. **Metal/MLX Testing**:
   - Test with `mx.use_device('metal')` context
   - Verify tensor types and shapes
   - Check memory statistics before and after operations
   - Force evaluation with `mx.eval()` to ensure computation

## Memory Management

### Metal Buffer Management

1. **Buffer Creation and Tracking**:
   - Create buffers using `manager.create_buffer()`
   - Track created buffers for cleanup
   - Use buffer pools when possible to reduce allocations

2. **Buffer Cleanup**:
   - Always release buffers in `finally` blocks
   - Track buffer IDs for proper release
   - Implement explicit cache clearing mechanism

3. **Memory Usage Monitoring**:
   - Use `mx.metal_memory_stats()` to monitor memory usage
   - Watch for buffer and memory leaks during development
   - Verify memory usage after repeated operations

### Memory Optimization Techniques

1. **Tensor Conversion**:
   - Use `to_numpy()` and `from_numpy()` for tensor conversions
   - Avoid unnecessary conversions between tensor types
   - Reuse existing tensors when possible

2. **Buffer Pooling**:
   - Use buffer pools for frequently allocated sizes
   - Set maximum pool sizes to prevent memory bloat
   - Implement automatic buffer release strategy

3. **Shared Memory**:
   - Use shared memory between CPU and GPU when possible
   - Avoid unnecessary data transfers
   - Use in-place operations when appropriate

### Debugging Memory Issues

1. **Memory Leak Detection**:
   - Run repeated operations and monitor memory growth
   - Use `test_memory_management` to verify proper cleanup
   - Check buffer counts before and after operations

2. **Memory Profiling**:
   - Use Metal Memory Debugger for detailed analysis
   - Monitor buffer allocation patterns
   - Check for buffer fragmentation

3. **Common Issues and Solutions**:
   - Missing buffer release: Ensure all buffers are explicitly released
   - Reference cycles: Break reference cycles to allow garbage collection
   - Excessive allocations: Use buffer pooling and reuse tensors

By following these guidelines, you'll contribute effectively to the Crystalline Consciousness AI framework while maintaining code quality, performance, and memory efficiency.

