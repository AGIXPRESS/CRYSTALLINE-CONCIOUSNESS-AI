# Crystalline Consciousness Framework: Troubleshooting Guide

## Table of Contents

1. [Common Implementation Issues](#common-implementation-issues)
2. [Performance Optimization](#performance-optimization)
3. [Memory Management](#memory-management)
4. [GPU Acceleration](#gpu-acceleration)
5. [Error Handling Best Practices](#error-handling-best-practices)
6. [Debugging Strategies](#debugging-strategies)
7. [System Requirements](#system-requirements)
8. [Testing and Validation](#testing-and-validation)

## Common Implementation Issues

### Metal Framework Integration Issues

#### Issue: Metal Library Not Found

**Symptoms**: ImportError when trying to use Metal shaders.

```
ImportError: No module named 'Metal'
```

**Solution**:

1. Verify you're running on macOS:
   ```python
   import platform
   if platform.system() != 'Darwin':
       print("Metal is only available on macOS")
   ```

2. Install the PyObjC bridge for Metal:
   ```bash
   pip install pyobjc-framework-Metal pyobjc-framework-Cocoa
   ```

3. Use the correct import format:
   ```python
   import Metal
   import Cocoa
   ```

#### Issue: Metal Shader Compilation Errors

**Symptoms**: Error messages when loading Metal shaders:

```
Error: Program compilation failed: ERROR: 0:15: 'function_name' : no matching overloaded function found
```

**Solution**:

1. Check shader function names match between Python code and Metal shaders:
   ```python
   # In Python
   metal_manager.execute_kernel("computeQuantumResonance", ...)
   
   # Metal shader should have:
   kernel void computeQuantumResonance(...)
   ```

2. Verify function signatures match expected types:
   ```metal
   // Correct signature for buffer input/output
   kernel void computeQuantumResonance(
       device float* output [[buffer(0)]],
       constant float4* parameters [[buffer(1)]],
       uint2 position [[thread_position_in_grid]],
       uint2 grid_size [[threads_per_grid]])
   ```

3. Use Metal Shader Validation:
   ```python
   def load_shaders(self, validate=True):
       # ...
       if validate:
           options = Metal.MTLCompileOptions.alloc().init()
           options.setFastMathEnabled_(False)
           options.setLanguageVersion_(Metal.MTLLanguageVersion2_2)
   ```

### Geometric Pattern Generation Issues

#### Issue: Incorrect Resonance Patterns

**Symptoms**: Generated patterns look distorted or lack symmetry properties.

**Solution**:

1. Verify platonic solid vertices are properly normalized:
   ```python
   def normalize_vertices(vertices):
       """Normalize vertices to lie on unit sphere."""
       return [v / np.linalg.norm(v) for v in vertices]
   ```

2. Check phase relationships between solids:
   ```python
   # Optimal phase relationships using golden ratio
   phi = (1 + np.sqrt(5)) / 2
   phases = [0.0, 2*np.pi/phi, 4*np.pi/phi, 6*np.pi/phi, 8*np.pi/phi]
   ```

3. Balance the weights appropriately:
   ```python
   # Balanced weight distribution
   solid_weights = [0.5, 0.3, 0.2, 0.1, 0.05]  # Decreasing by importance
   ```

#### Issue: NaN Values in Pattern

**Symptoms**: Generated pattern contains NaN values.

**Solution**:

1. Check for division by zero in feedback function:
   ```python
   # Safe feedback application
   def apply_feedback(value, strength):
       if abs(value) < 1e-10:
           return value
       return value + strength * np.tanh(value)
   ```

2. Sanitize inputs in Metal shader:
   ```metal
   float safe_distance(float3 a, float3 b) {
       float3 diff = a - b;
       float dist_squared = dot(diff, diff);
       // Prevent sqrt of negative values
       return sqrt(max(0.0, dist_squared));
   }
   ```

3. Add validation to parameters:
   ```python
   def validate_params(params):
       """Validate parameter dictionary, fix if needed."""
       if any(np.isnan(params['solid_weights'])):
           print("Warning: NaN weights detected, using defaults")
           params['solid_weights'] = [0.5, 0.3, 0.2, 0.1, 0.05]
       return params
   ```

## Performance Optimization

### Computation Speed Improvements

#### Technique 1: Metal Shader Optimization

Metal shaders can be optimized for specific hardware:

```metal
// Use hardware-specific optimizations
kernel void optimizedResonanceComputation(
    device float* output [[buffer(0)]],
    constant float4* parameters [[buffer(1)]],
    uint2 position [[thread_position_in_grid]],
    uint2 grid_size [[threads_per_grid]],
    uint2 threadgroup_position [[threadgroup_position_in_grid]],
    uint2 threads_per_threadgroup [[threads_per_threadgroup]],
    uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]])
{
    // Calculate position only once
    float2 pos = float2(position) / float2(grid_size);
    pos = pos * 2.0 - 1.0;
    
    // Use thread-local memory for vertices
    threadgroup float3 tetra_vertices[4];
    
    // Initialize only once per threadgroup
    if (thread_position_in_threadgroup.x == 0 && thread_position_in_threadgroup.y == 0) {
        tetra_vertices[0] = normalize(float3(1, 1, 1));
        tetra_vertices[1] = normalize(float3(1, -1, -1));
        tetra_vertices[2] = normalize(float3(-1, 1, -1));
        tetra_vertices[3] = normalize(float3(-1, -1, 1));
    }
    
    // Ensure initialization is complete
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Rest of computations...
}
```

#### Technique 2: NumPy Vectorization

Replace explicit loops with vectorized operations:

```python
# Slow approach with loops
def calculate_distances_slow(points, vertices):
    distances = np.zeros((len(points), len(vertices)))
    for i, point in enumerate(points):
        for j, vertex in enumerate(vertices):
            distances[i, j] = np.linalg.norm(point - vertex)
    return distances

# Fast vectorized approach
def calculate_distances_fast(points, vertices):
    # Reshape for broadcasting
    points_reshaped = points.reshape(points.shape[0], 1, 3)
    vertices_reshaped = vertices.reshape(1, vertices.shape[0], 3)
    
    # Vectorized distance calculation
    differences = points_reshaped - vertices_reshaped
    squared_distances = np.sum(differences**2, axis=2)
    distances = np.sqrt(squared_distances)
    return distances
```

#### Technique 3: Just-In-Time Compilation

Use Numba for critical computational functions:

```python
import numba

@numba.jit(nopython=True, parallel=True)
def compute_resonance_pattern(positions, vertices, params):
    """JIT-compiled resonance pattern calculation."""
    result = np.zeros(len(positions))
    for i in numba.prange(len(positions)):
        pos = positions[i]
        for j in range(len(vertices)):
            vertex = vertices[j]
            dist = np.sqrt(((pos - vertex)**2).sum())
            result[i] += np.exp(-params[0] * dist**2) * np.sin(params[1] * dist + params[2])
    return result
```

### Parallel Processing

For large-scale pattern generation and analysis:

```python
import multiprocessing as mp

def process_chunk(chunk_data, params):
    # Process a chunk of data
    results = compute_resonance_pattern(chunk_data, params)
    return results

def parallel_resonance_computation(full_data, params, num_processes=None):
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Split data into chunks
    chunks = np.array_split(full_data, num_processes)
    
    # Create pool and execute
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_chunk, [(chunk, params) for chunk in chunks])
    
    # Combine results
    return np.concatenate(results)
```

## Memory Management

### Efficient Data Structures

#### Issue: Memory Consumption with Large Patterns

**Symptoms**: Out of memory errors when generating large patterns

**Solution**: Use memory-mapped arrays for large patterns:

```python
import numpy as np

def generate_large_pattern(dimensions, params, output_file=None):
    """Generate large pattern using memory mapping."""
    # Create memory-mapped output array
    if output_file is None:
        output_file = 'temp_pattern.npy'
    
    # Initialize memory-mapped file
    shape = dimensions
    mmap_array = np.lib.format.open_memmap(
        output_file, mode='w+', dtype=np.float32, shape=shape
    )
    
    # Process in chunks
    chunk_size = 1024  # Process 1024 rows at a time
    for i in range(0, shape[0], chunk_size):
        end_i = min(i + chunk_size, shape[0])
        chunk_positions = generate_positions((end_i - i, shape[1]))
        chunk_result = compute_chunk(chunk_positions, params)
        mmap_array[i:end_i, :] = chunk_result
    
    return output_file
```

#### Issue: Memory Leaks in Metal Implementation

**Symptoms**: Increasing memory usage over time with Metal shaders

**Solution**: Properly release Metal resources:

```python
class MetalManager:
    def __init__(self):
        self.device = Metal.MTLCreateSystemDefaultDevice()
        self.command_queue = self.device.newCommandQueue()
        self.buffers = {}
        self.functions = {}
    
    def __del__(self):
        """Clean up Metal resources."""
        # Clear references to allow ARC to release
        self.command_queue = None
        self.buffers.clear()
        self.functions.clear()
    
    def release_buffer(self, name):
        """Explicitly release a buffer."""
        if name in self.buffers:
            del self.buffers[name]
```

### Streaming Processing

For handling patterns that don't fit in memory:

```python
def analyze_large_pattern(pattern_file, chunk_size=1024):
    """Stream processing of large pattern file."""
    # Open memory-mapped file in read mode
    pattern = np.load(pattern_file, mmap_mode='r')
    
    # Initialize results
    result_stats = {
        'min': float('inf'),
        'max': float('-inf'),
        'sum': 0,
        'sum_squared': 0,
        'count': 0
    }
    
    # Process in chunks
    for i in range(0, pattern.shape[0], chunk_size):
        end_i = min(i + chunk_size, pattern.shape[0])
        chunk = pattern[i:end_i, :]
        
        # Update statistics
        result_stats['min'] = min(result_stats['min'], chunk.min())
        result_stats['max'] = max(result_stats['max'], chunk.max())
        result_stats['sum'] += chunk.sum()
        result_stats['sum_squared'] += (chunk**2).sum()
        result_stats['count'] += chunk.size
    
    # Calculate final statistics
    mean = result_stats['sum'] / result_stats['count']
    variance = (result_stats['sum_squared'] / result_stats['count']) - (mean**2)
    std = np.sqrt(variance)
    
    return {
        'min': result_stats['min'],
        'max': result_stats['max'],
        'mean': mean,
        'std': std
    }
```

## GPU Acceleration

### Metal Performance Optimization

#### Technique 1: Thread Group Size Optimization

Optimize thread group sizes for your specific GPU:

```python
def optimize_thread_group_size(metal_manager, test_sizes=None):
    """Find optimal thread group size for Metal computations."""
    if test_sizes is None:
        test_sizes = [(8, 8), (16, 16), (32, 32), (64, 64)]
    
    pattern_size = (1024, 1024)  # Test pattern size
    params = generate_default_params()
    
    best_time = float('inf')
    best_size = None
    
    for size in test_sizes:
        # Time execution
        start = time.time()
        
        # Run test with current thread group size
        metal_manager.thread_group_size = size
        metal_manager.compute_resonance_pattern(pattern_size, params)
        
        end = time.time()
        elapsed = end - start
        
        print(f"Thread group size {size}: {elapsed:.4f} seconds")
        
        if elapsed < best_time:
            best_time = elapsed
            best_size = size
    
    print(f"Optimal thread group size: {best_size}")
    return best_size
```

#### Technique 2: Buffer Reuse

Reuse Metal buffers to avoid allocation overhead:

```python
class BufferPoolManager:
    def __init__(self, device):
        self.device = device
        self.pool = {}  # (size, dtype) -> list of free buffers
    
    def get_buffer(self, size, dtype=np.float32):
        """Get a buffer from the pool or create a new one."""
        key = (size, dtype)
        if key in self.pool and self.pool[key]:
            return self.pool[key].pop()
        
        # Create new buffer
        buffer_size = size * np.dtype(dtype).itemsize
        buffer = self.device.newBufferWithLength_options_(
            buffer_size, Metal.MTLResourceStorageModeShared
        )
        return buffer
    
    def release_buffer(self, buffer, size, dtype=np.float32):
        """Return a buffer to the pool."""
        key = (size, dtype)
        if key not in self.pool:
            self.pool[key] = []
        self.pool[key].append(buffer)
```

#### Technique 3: Asynchronous Computation

Use asynchronous Metal command buffers for overlapping computation:

```python
def execute_async(self, function_name, output_buffer, input_buffers, grid_size):
    """Execute Metal function asynchronously."""
    command_buffer = self.command_queue.commandBuffer()
    
    # Create compute command encoder
    compute_encoder = command_buffer.computeCommandEncoder()
    
    # Set compute pipeline state
    compute_encoder.setComputePipelineState_(self.functions[function_name])
    
    # Set buffers
    compute_encoder.setBuffer_offset_atIndex_(output_buffer, 0, 0)
    for i, buffer in enumerate(input_buffers):
        compute_encoder.setBuffer_offset_atIndex_(buffer, 0, i+1)
    
    # Calculate grid and thread group size
    threads_per_group = Metal.MTLSizeMake(16, 16, 1)
    grid_size_mtl = Metal.MTLSizeMake(grid_size[0], grid_size[1], 1)
    
    # Dispatch threads
    compute_encoder.dispatchThreads_threadsPerThreadgroup_(
        grid_size_mtl, threads_per_group)
    
    # End encoding
    compute_encoder.endEncoding()
    
    # Register completion handler
    def completion_handler(command_buffer):
        # Callback when computation is complete
        pass
    
    # Schedule execution and return immediately
    command_buffer.addCompletedHandler_(completion_handler)
    command_buffer.commit()
    
    return command_buffer  # Return buffer for status checking
```

### Multiple GPU Utilization

For systems with multiple GPUs:

```python
def setup_multi_gpu():
    """Initialize all available GPUs."""
    # Get all devices
    devices = []
    device_count = Metal.MTLCopyAllDevices().count()
    
    for i in range(device_count):
        device = Metal.MTLCopyAllDevices().objectAtIndex_(i)
        devices.append(device)
    
    print(f"Found {len(devices)} Metal device(s):")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device.name()}")
    
    return devices

def distribute_computation(devices, total_work, params):
    """Distribute work across multiple GPUs."""
    # Create managers for each device
    managers = [MetalManager(device) for device in devices]
    
    # Split work
    chunks = np.array_split(total_work, len(devices))
    
    # Create buffers and submit work
    commands = []
    results = []
    
    for i, (manager, chunk) in enumerate(zip(managers, chunks)):
        # Set up computation for this device
        result_buffer = np.zeros(chunk.shape, dtype=np.float32)
        command = manager.execute_async("computeQuantumResonance", 
                                       result_buffer, params, chunk.shape)
        commands.append(command)
        results.append(result_buffer)
    
    # Wait for all to complete
    for command in commands:
        command.waitUntilCompleted()
    
    # Combine results
    return np.concatenate(results)
```

## Error Handling Best Practices

### Graceful Degradation

Implement fallback strategies for when optimal hardware is unavailable:

```python
def initialize_compute_environment():
    """Initialize computation environment with fallbacks."""
    # Try Metal first (macOS)
    try:
        import Metal
        device = Metal.MTLCreateSystemDefaultDevice()
        if device is not None:
            print("Using Metal acceleration")
            return MetalComputeEngine(device)
    except (ImportError, AttributeError) as e:
        print(f"Metal not available: {e}")
    
    # Try CUDA next (NVIDIA GPUs)
    try:
        import cupy as cp
        if cp.cuda.is_available():
            print("Using CUDA acceleration")
            return CudaComputeEngine()
    except ImportError:
        print("CUDA not available")
    
    # Fall back to NumPy with optimizations
    try:
        import numba
        print("Using NumPy with Numba acceleration")
        return NumbaComputeEngine()
    except ImportError:
        print("Numba not available")
    
    # Last resort: pure NumPy
    print("Using pure NumPy (slowest)")
    return NumpyComputeEngine()
```

### Robust Error Handling

Implement comprehensive error handling for Metal operations:

```python
class RobustMetalManager:
    def execute_with_recovery(self, function_name, *args, **kwargs):
        """Execute Metal function with error recovery."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return self.execute(function_name, *args, **kwargs)
            except Metal.MTLLibraryError as e:
                if "out of memory" in str(e) and attempt < max_retries - 1:
                    # Handle out of memory by clearing cache and reducing size
                    self.clear_buffer_cache()
                    print(f"Retrying with reduced memory usage (attempt {attempt+1})")
                    # Reduce work size for next attempt
                    if 'grid_size' in kwargs:
                        kwargs['grid_size'] = tuple(d // 2 for d in kwargs['grid_size'])
                else:
                    # Rethrow other errors or final attempt
                    raise MetalError(f"Metal execution failed: {e}")
        
        raise MetalError(f"Failed after {max_retries} attempts")

    def clear_buffer_cache(self):
        """Clear cached Metal buffers to free memory."""
        for buffer in self.buffer_cache.values():
            # Release buffer
            del buffer
        self.buffer_cache.clear()
        # Force garbage collection
        import gc
        gc.collect()
```

### Validation Layers

Implement parameter validation to catch errors early:

```python
def validate_resonance_parameters(params):
    """Validate resonance parameters before computation."""
    errors = []
    
    # Check required fields
    required_fields = ['solid_weights', 'phases', 'feedback_strength']
    for field in required_fields:
        if field not in params:
            errors.append(f"Missing required parameter: {field}")
    
    # Check solid weights
    if 'solid_weights' in params:
        weights = params['solid_weights']
        if not isinstance(weights, (list, tuple)) or len(weights) != 5:
            errors.append("solid_weights must be a list of 5 values for each platonic solid")
        elif any(not isinstance(w, (int, float)) for w in weights):
            errors.append("solid_weights must contain only numeric values")
        elif any(w < 0 for w in weights):
            errors.append("solid_weights must be non-negative")
    
    # Check phases
    if 'phases' in params:
        phases = params['phases']
        if not isinstance(phases, (list, tuple)) or len(phases) != 5:
            errors.append("phases must be a list of 5 values for each platonic solid")
        elif any(not isinstance(p, (int, float)) for p in phases):
            errors.append("phases must contain only numeric values")
    
    # Check feedback strength
    if 'feedback_strength' in params:
        strength = params['feedback_strength']
        if not isinstance(strength, (int, float)):
            errors.append("feedback_strength must be a numeric value")
        elif not 0 <= strength <= 1:
            errors.append("feedback_strength should be between 0 and 1")
    
    if errors:
        raise ValueError("Parameter validation failed:\n- " + "\n- ".join(errors))
    
    return True
```

## Debugging Strategies

### Debug Visualization

Visualize intermediate results to diagnose issues:

```python
def debug_resonance_components(params, output_dir=None):
    """Generate and visualize each resonance component separately."""
    import matplotlib.pyplot as plt
    
    if output_dir is None:
        output_dir = 'debug_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Store individual components
    components = {}
    solids = ['tetrahedron', 'octahedron', 'cube', 'dodecahedron', 'icosahedron']
    
    # Generate each component separately
    for i, solid in enumerate(solids):
        # Create weights with only this solid
        weights = [0.0] * 5
        weights[i] = 1.0
        
        # Generate component
        component_params = params.copy()
        component_params['solid_weights'] = weights
        component = generate_resonance_pattern(component_params)
        components[solid] = component
        
        # Visualize component
        plt.figure(figsize=(10, 10))
        plt.imshow(component, cmap='viridis')
        plt.colorbar(label='Resonance Value')
        plt.title(f'{solid.capitalize()} Component')
        plt.savefig(os.path.join(output_dir, f"{solid}_component.png"))
        plt.close()
    
    # Generate combined result for comparison
    combined = generate_resonance_pattern(params)
    
    # Visualize combined result
    plt.figure(figsize=(10, 10))
    plt.imshow(combined, cmap='viridis')
    plt.colorbar(label='Resonance Value')
    plt.title('Combined Pattern')
    plt.savefig(os.path.join(output_dir, "combined_pattern.png"))
    plt.close()
    
    # Visualize difference from expected
    if 'expected' in params:
        expected = params['expected']
        diff = combined - expected
        
        plt.figure(figsize=(10, 10))
        plt.imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(label='Difference')
        plt.title('Difference from Expected')
        plt.savefig(os.path.join(output_dir, "difference.png"))
        plt.close()
    
    return components, combined
```

### Gradient Inspection

Inspect gradients to identify optimization issues:

```python
def inspect_resonance_gradients(pattern, output_dir=None):
    """Compute and visualize gradients for debugging."""
    import matplotlib.pyplot as plt
    
    if output_dir is None:
        output_dir = 'debug_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute gradients
    grad_y, grad_x = np.gradient(pattern)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Visualize gradients
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(grad_x, cmap='RdBu_r')
    plt.colorbar(label='X Gradient')
    plt.title('X Gradient')
    
    plt.subplot(132)
    plt.imshow(grad_y, cmap='RdBu_r')
    plt.colorbar(label='Y Gradient')
    plt.title('Y Gradient')
    
    plt.subplot(133)
    plt.imshow(gradient_magnitude, cmap='viridis')
    plt.colorbar(label='Gradient Magnitude')
    plt.title('Gradient Magnitude')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gradients.png"))
    plt.close()
    
    # Check for suspicious gradient patterns
    high_gradient_ratio = np.sum(gradient_magnitude > 5.0) / gradient_magnitude.size
    zero_gradient_ratio = np.sum(gradient_magnitude < 1e-6) / gradient_magnitude.size
    
    print(f"High gradient regions: {high_gradient_ratio:.2%}")
    print(f"Zero gradient regions: {zero_gradient_ratio:.2%}")
    
    if high_gradient_ratio > 0.1:
        print("WARNING: Large high-gradient regions detected. Check for instabilities.")
    if zero_gradient_ratio > 0.5:
        print("WARNING: Excessive zero-gradient regions detected. Check for pattern collapse.")
    
    return {
        'grad_x': grad_x,
        'grad_y': grad_y,
        'magnitude': gradient_magnitude,
        'high_gradient_ratio': high_gradient_ratio,
        'zero_gradient_ratio': zero_gradient_ratio
    }
```

### Metal Shader Debugging

Diagnose Metal shader issues using debug layers:

```python
def setup_metal_debugging():
    """Configure Metal debugging environment."""
    # Set Metal API validation environment variable
    import os
    os.environ["MTL_DEBUG_LAYER"] = "1"
    os.environ["MTL_DEBUG_LAYER_ERROR_ONLY"] = "0"
    
    print("Metal debugging enabled. Shader errors will be reported in detail.")

def inject_debug_code(shader_source):
    """Inject debug print statements into Metal shader code."""
    # Add debug print helpers
    debug_helpers = """
// Debug helper functions
void debug_print_float(constant char* label, float value) {
    // Metal doesn't support direct printing, but this can be useful 
    // when captured by API validation tools
}
"""
    
    # Insert debug prints at key points
    modified_source = shader_source.replace(
        "float resonance = 0.0;",
        "float resonance = 0.0;\n    debug_print_float(\"Processing vertex\", 0.0);"
    )
    
    # Add debug helpers at top of file
    modified_source = debug_helpers + modified_source
    
    return modified_source
```

### Function Tracing

Implement tracing to diagnose call sequence issues:

```python
def enable_function_tracing(verbose=True):
    """Enable function tracing for debugging."""
    trace_depth = [0]
    
    def trace_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            prefix = "| " * trace_depth[0]
            print(f"{prefix}→ Entering {func.__name__}")
            
            if verbose:
                arg_str = ", ".join([
                    f"{a}" if isinstance(a, (int, float, str, bool)) else f"{type(a).__name__}" 
                    for a in args
                ])
                kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                if args and kwargs:
                    print(f"{prefix}  Args: {arg_str}, {kwargs_str}")
                elif args:
                    print(f"{prefix}  Args: {arg_str}")
                elif kwargs:
                    print(f"{prefix}  Args: {kwargs_str}")
            
            trace_depth[0] += 1
            try:
                result = func(*args, **kwargs)
                trace_depth[0] -= 1
                
                if verbose:
                    # For large results, just show the type
                    if isinstance(result, np.ndarray):
                        result_str = f"ndarray(shape={result.shape}, dtype={result.dtype})"
                    elif isinstance(result, (list, tuple)) and len(result) > 5:
                        result_str = f"{type(result).__name__}[{len(result)} items]"
                    else:
                        result_str = str(result)
                    print(f"{prefix}← Exiting {func.__name__} = {result_str}")
                else:
                    print(f"{prefix}← Exiting {func.__name__}")
                
                return result
            except Exception as e:
                trace_depth[0] -= 1
                print(f"{prefix}! ERROR in {func.__name__}: {e}")
                raise
        
        return wrapper
    
    return trace_decorator

# Use like this:
trace = enable_function_tracing()

@trace
def generate_pattern(params):
    """Generate pattern with tracing."""
    # Function body...
```

## System Requirements

### Hardware Requirements

The Crystalline Consciousness framework has different hardware requirements depending on the desired performance level:

#### Minimum Requirements

- **CPU**: Intel Core i5 or equivalent (4+ cores recommended)
- **RAM**: 8 GB
- **Storage**: 1 GB of free space
- **GPU**: 
  - macOS: Any Metal-compatible GPU (integrated is sufficient)
  - Other platforms: NumPy-only implementation (slower)

#### Recommended Requirements

- **CPU**: Intel Core i7/i9 or AMD Ryzen 7/9 (8+ cores)
- **RAM**: 16 GB or more
- **Storage**: 4 GB of free space
- **GPU**:
  - macOS: Apple M1/M2/M3 series or dedicated AMD GPU with 4+ GB VRAM
  - Windows/Linux: NVIDIA GPU with CUDA support (4+ GB VRAM)

#### For Large-Scale Pattern Generation (4K+ resolution)

- **CPU**: 12+ cores
- **RAM**: 32 GB or more
- **Storage**: SSD with 10+ GB free space
- **GPU**: 
  - Apple M2 Pro/Max/Ultra or newer
  - NVIDIA RTX series with 8+ GB VRAM

### Software Requirements

#### Core Dependencies

- **Operating System**:
  - Primary support: macOS 11.0+ (Big Sur or newer)
  - Secondary support: Ubuntu 20.04+, Windows 10/11 with WSL2

- **Python Environment**:
  - Python 3.9 or newer
  - pip 21.0 or newer
  - virtualenv or conda (recommended)

#### Required Packages

```
# Core packages
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.4.0

# Metal support (macOS only)
pyobjc-framework-Metal>=8.0  # macOS only
pyobjc-framework-Cocoa>=8.0  # macOS only

# Analysis tools
pywavelets>=1.2.0
scikit-image>=0.18.0

# Performance optimization
numba>=0.54.0
```

#### Optional Packages

```
# GPU Acceleration alternatives
cupy>=10.0.0          # For NVIDIA GPUs
tensorflow>=2.7.0     # Alternative acceleration

# Interactive visualization
ipython>=7.0.0
jupyter>=1.0.0
ipywidgets>=7.6.0

# Testing
pytest>=6.0.0
pytest-cov>=2.12.0
```

### Configuration Requirements

#### Environment Variables

Several environment variables can be used to control the framework's behavior:

```bash
# Enable Metal debugging
export MTL_DEBUG_LAYER=1
export MTL_DEBUG_LAYER_ERROR_ONLY=0

# Control computation precision
export CC_PRECISION=float32  # Options: float16, float32, float64

# Control parallelism
export CC_NUM_THREADS=4  # Number of CPU threads to use

# Force CPU-only computation (disables GPU)
export CC_FORCE_CPU=0  # Set to 1 to force CPU computation
```

#### Framework Configuration

Create a `config.yml` file in your project directory to customize behavior:

```yaml
# config.yml example
compute:
  device: 'auto'  # 'auto', 'cpu', 'metal', 'cuda'
  precision: 'float32'
  thread_count: 4
  buffer_pool_size: 128  # MB

patterns:
  default_size: [128, 128]
  default_weights: [0.5, 0.3, 0.2, 0.1, 0.05]
  default_phases: [0.0, 0.785, 1.57, 2.355, 3.14]
  feedback_strength: 0.1

visualization:
  default_colormap: 'viridis'
  dpi: 150
  save_format: 'png'
```

For detailed setup instructions, refer to the [Implementation Guide](implementation_guide.md#environment-setup).

## Testing and Validation

### Unit Testing

Implement comprehensive unit tests for each component:

```python
# test_resonance_patterns.py
import pytest
import numpy as np
from crystalline.patterns import generate_resonance_pattern

def test_pattern_generation():
    """Test basic pattern generation."""
    # Define test parameters
    params = {
        'solid_weights': [0.5, 0.3, 0.2, 0.1, 0.05],
        'phases': [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
        'feedback_strength': 0.1,
        'dimensions': (64, 64)
    }
    
    # Generate pattern
    pattern = generate_resonance_pattern(**params)
    
    # Basic validation
    assert pattern.shape == (64, 64)
    assert not np.isnan(pattern).any()
    assert np.isfinite(pattern).all()
    
    # Value range validation
    assert pattern.min() >= -2.0
    assert pattern.max() <= 2.0

def test_pattern_symmetry():
    """Test pattern symmetry properties."""
    # Generate symmetric pattern
    params = {
        'solid_weights': [1.0, 0.0, 0.0, 0.0, 0.0],  # Tetrahedron only
        'phases': [0.0, 0.0, 0.0, 0.0, 0.0],
        'feedback_strength': 0.0,
        'dimensions': (64, 64)
    }
    
    pattern = generate_resonance_pattern(**params)
    
    # Test diagonal symmetry
    diag_symmetry = np.corrcoef(
        pattern.flatten(),
        np.flipud(np.fliplr(pattern)).flatten()
    )[0, 1]
    
    assert diag_symmetry > 0.95, "Expected high diagonal symmetry"
```

### Integration Testing

Test the integration between components:

```python
# test_end_to_end.py
import pytest
import os
import numpy as np
from crystalline.patterns import generate_resonance_pattern
from crystalline.analysis import analyze_quantum_resonance

def test_pattern_generation_and_analysis():
    """Test the full pattern generation and analysis pipeline."""
    # Generate test pattern
    params = {
        'solid_weights': [0.5, 0.3, 0.2, 0.1, 0.05],
        'phases': [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
        'feedback_strength': 0.1,
        'dimensions': (128, 128)
    }
    
    pattern = generate_resonance_pattern(**params)
    
    # Set up temporary output directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run analysis
        results = analyze_quantum_resonance(
            pattern,
            output_dir=tmpdir,
            run_statistics=True,
            run_fft=True,
            run_spatial=True
        )
        
        # Verify outputs
        assert 'statistics' in results
        assert 'fft_analysis' in results
        assert 'spatial_analysis' in results
        
        # Check for expected files
        expected_files = [
            "value_distribution.png",
            "fft_spectrum.png",
            "enhanced_visualization.png"
        ]
        
        for filename in expected_files:
            filepath = os.path.join(tmpdir, filename)
            assert os.path.exists(filepath), f"Missing expected file: {filename}"
```

### Performance Testing

Test performance characteristics and optimizations:

```python
# test_performance.py
import pytest
import time
import numpy as np
from crystalline.patterns import generate_resonance_pattern

def test_pattern_generation_performance():
    """Test performance of pattern generation."""
    # Standard parameters
    params = {
        'solid_weights': [0.5, 0.3, 0.2, 0.1, 0.05],
        'phases': [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
        'feedback_strength': 0.1
    }
    
    # Test different sizes
    sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]
    timings = {}
    
    for size in sizes:
        params['dimensions'] = size
        
        # Warm-up run
        generate_resonance_pattern(**params)
        
        # Timed runs
        runs = 5
        start = time.time()
        for _ in range(runs):
            generate_resonance_pattern(**params)
        end = time.time()
        
        avg_time = (end -
