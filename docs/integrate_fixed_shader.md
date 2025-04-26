# Integrating the Fixed MutualityField.metal Shader with Your Model

This guide provides step-by-step instructions for integrating the fixed MutualityField.metal shader with your full crystalline consciousness model.

## 1. Update Import Statements

In your model's main file(s), update the import statements to use the Metal operations:

```python
from crystalline_mlx.Python.metal_ops import (
    geometric_activation,
    apply_resonance,
    mutuality_field,
    is_metal_available
)
```

## 2. Modify Your CrystallineMutualityField Class

Update your CrystallineMutualityField class to use the fixed mutuality_field function:

```python
class CrystallineMutualityField(nn.Module):
    def __init__(self, dim, grid_size=16, use_metal=True):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size
        self.use_metal = use_metal
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        # Parameters
        self.interference_scale = nn.Parameter(torch.tensor(0.5))
        self.decay_rate = nn.Parameter(torch.tensor(0.1))
        self.persistence_state = None
        
        # Field transformations
        self.to_grid = nn.Linear(dim, grid_size * grid_size)
        self.to_vector = nn.Linear(grid_size * grid_size, dim)
        
        # Create field integrator for CPU fallback
        self.field_integrator = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        
    def forward(self, weaving_field, dt=0.1):
        # Original PyTorch implementation (CPU fallback)
        if not is_metal_available() or not self.use_metal:
            batch_size = weaving_field.shape[0]
            
            # Transform to 2D grid
            grid_field = self.to_grid(weaving_field)
            field = grid_field.view(batch_size, 1, self.grid_size, self.grid_size)
            
            # Create shifted versions
            shifted_field_r = torch.roll(field, shifts=1, dims=2)
            shifted_field_t = torch.roll(field, shifts=1, dims=3)
            
            # Create interference patterns
            interference_r = torch.cat([field, shifted_field_r], dim=1)
            interference_t = torch.cat([field, shifted_field_t], dim=1)
            
            # Process through field integrator
            mutual_field_r = self.field_integrator(interference_r)
            mutual_field_t = self.field_integrator(interference_t)
            
            # Combine interference patterns
            mutual_field = (mutual_field_r + mutual_field_t) / 2.0
            
            # Apply golden ratio scaling
            interference_factor = torch.sin(self.phi * torch.mean(field, dim=[2, 3], keepdim=True))
            mutual_field = mutual_field * (1 + self.interference_scale * interference_factor)
            
            # Apply persistence function
            if self.persistence_state is None or self.persistence_state.shape[0] != batch_size:
                self.persistence_state = torch.zeros_like(mutual_field)
                
            decay_factor = torch.exp(-self.decay_rate * dt)
            self.persistence_state = mutual_field + decay_factor * self.persistence_state
            
            # Final output is flattened back to vector
            output = self.persistence_state.view(batch_size, -1)
            output = self.to_vector(output)
            
            return output
        else:
            # Metal-accelerated implementation
            return mutuality_field(
                weaving_field,
                self.grid_size,
                self.interference_scale.item(),
                self.decay_rate.item(),
                dt
            )
```

## 3. Allocate Intermediate Buffers

The fixed shader requires intermediate buffers for the conv2d_layer1 function. Update your metal_ops.py file to allocate these buffers when calling the process_interference_fields kernel:

```python
def _mutuality_field_metal(x, grid_size, interference_scale, decay_rate, dt):
    """Metal implementation of mutuality field."""
    if not is_metal_available():
        return _mutuality_field_fallback(x, grid_size, interference_scale, decay_rate, dt)
    
    # Get shader manager
    manager = _get_shader_manager()
    if manager is None or manager.device is None:
        return _mutuality_field_fallback(x, grid_size, interference_scale, decay_rate, dt)
    
    # ... existing code ...
    
    # Create intermediate layer buffers for r and t convolution outputs
    # These are used by the process_interference_fields kernel
    layer1_output_r = np.zeros((batch_size * 8 * grid_size * grid_size), dtype=np.float32)
    layer1_output_t = np.zeros((batch_size * 8 * grid_size * grid_size), dtype=np.float32)
    layer1_output_r_buffer = manager.create_buffer(layer1_output_r)
    layer1_output_t_buffer = manager.create_buffer(layer1_output_t)
    
    # Add these buffers to the inputs for process_interference_fields
    process_inputs = [
        r_interference_buffer, t_interference_buffer, 
        r_processed_buffer, t_processed_buffer,
        layer1_output_r_buffer, layer1_output_t_buffer,  # New parameters
        batch_buffer, grid_size_buffer
    ]
    
    # Execute process_interference_fields
    success = manager.execute_shader(
        "process_interference_fields",
        process_inputs,
        [],
        (batch_size, 1, 1),
        (1, 1, 1)
    )
    
    # ... rest of function ...
```

## 4. Add Fallback Mechanism

Add a fallback mechanism to handle any unexpected issues with the Metal implementation:

```python
def mutuality_field(x, grid_size, interference_scale, decay_rate, dt):
    """
    Apply mutuality field operations to input tensor.
    
    Args:
        x: Input tensor of shape (batch_size, input_dim)
        grid_size: Size of the grid field
        interference_scale: Scale of interference patterns
        decay_rate: Decay rate of the field
        dt: Time step for field evolution
            
    Returns:
        Tensor with mutuality field operations applied
    """
    try:
        if HAS_MLX and isinstance(x, mx.array):
            # Call forward directly without gradient tracking
            return MutualityField.forward(
                None, x, grid_size, interference_scale, decay_rate, dt)
        else:
            # Try Metal implementation first
            result = _mutuality_field_metal(x, grid_size, interference_scale, decay_rate, dt)
            
            # Check if result is valid
            if result is not None and not np.isnan(result).any() and not np.isinf(result).any():
                return result
            else:
                # Fall back to CPU if result is invalid
                print("Warning: Metal implementation returned invalid result. Falling back to CPU.")
                return _mutuality_field_fallback(x, grid_size, interference_scale, decay_rate, dt)
    except Exception as e:
        print(f"Warning: Metal implementation failed: {e}. Falling back to CPU.")
        return _mutuality_field_fallback(x, grid_size, interference_scale, decay_rate, dt)
```

## 5. Testing Integration

Add a simple test to verify the integration:

```python
def test_integration():
    # Check if Metal is available
    print(f"Metal available: {is_metal_available()}")
    
    # Create model
    model = YourCrystallineModel(use_metal=True)
    
    # Create test input
    batch_size = 4
    input_dim = 64
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass with Metal
    with torch.no_grad():
        output_metal = model(x)
    
    # Forward pass without Metal
    model.use_metal = False
    with torch.no_grad():
        output_cpu = model(x)
    
    # Compare outputs
    diff = torch.max(torch.abs(output_metal - output_cpu))
    print(f"Maximum difference between Metal and CPU: {diff.item():.6f}")
    
    if diff.item() < 1e-3:
        print("✅ Integration test passed: Metal and CPU outputs match within tolerance")
    else:
        print("⚠️ Integration test warning: Metal and CPU outputs differ significantly")
        
    # Benchmark performance
    import time
    
    # Warm-up
    for _ in range(10):
        _ = model(x)
    
    # Benchmark CPU
    model.use_metal = False
    start = time.time()
    for _ in range(100):
        _ = model(x)
    cpu_time = time.time() - start
    
    # Benchmark Metal
    model.use_metal = True
    start = time.time()
    for _ in range(100):
        _ = model(x)
    metal_time = time.time() - start
    
    print(f"CPU time: {cpu_time:.6f}s")
    print(f"Metal time: {metal_time:.6f}s")
    print(f"Speedup: {cpu_time / metal_time:.2f}x")
```

## 6. Debugging Tips

If you encounter any issues:

1. Check buffer indices: Make sure the buffer indices in your Python wrapper match those in the Metal shader.

2. Verify input shapes: The mutuality_field function expects input of shape (batch_size, input_dim).

3. Enable logging: Add debug prints to track which code path is being used.

4. Inspect intermediate results: Add code to extract and visualize intermediate buffers if problems occur.

5. Handle dynamic grid sizes: If your model uses different grid sizes, make sure your Metal shader can handle them.

## 7. Performance Optimization

After successful integration, consider these optimizations:

1. Batch Processing: Process multiple inputs simultaneously to maximize GPU utilization.

2. Buffer Reuse: Reuse buffers for intermediate computations instead of creating new ones each time.

3. Precision Control: Consider using half-precision (float16) for certain operations if accuracy requirements allow.

4. Pipeline Streamlining: Minimize data transfers between CPU and GPU.

5. Memory Management: Implement proper cleanup to avoid memory leaks.

The fixed MutualityField.metal shader should now be fully integrated with your crystalline consciousness model and ready to accelerate your computations on Apple Silicon hardware!
