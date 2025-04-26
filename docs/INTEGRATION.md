# Integration Guide for Crystalline MLX

This guide explains how to integrate the Metal shader implementation with your existing crystalline consciousness model to achieve significant performance improvements on Apple Silicon hardware.

## Step-by-Step Integration

### 1. Initial Setup

1. Ensure your project structure includes the crystalline_mlx directory
2. Add the required dependencies (PyTorch with MPS support or MLX)
3. Verify the Metal shaders work by running the test script:
   ```bash
   python crystalline_mlx/Tests/test_simple.py
   ```

### 2. Import the Metal Operations

Add the following imports to your model files:

```python
from crystalline_mlx.Python.metal_ops import (
    geometric_activation,
    apply_resonance,
    mutuality_field,
    is_metal_available
)
```

### 3. Modify Geometric Layer Classes

#### TetrahedronLayer, CubeLayer, DodecahedronLayer, IcosahedronLayer

Replace the forward method in each layer class with a Metal-accelerated version:

```python
def forward(self, x):
    # Original PyTorch implementation
    if not is_metal_available() or not self.use_metal:
        # Process through each vertex
        vertex_outputs = [proj(x) for proj in self.projections]
        
        # Combine vertex outputs
        combined = torch.cat(vertex_outputs, dim=1)
        
        # Apply edge-based interactions
        for i, (v1, v2) in enumerate(self.edges):
            edge_weight = torch.sigmoid(self.edge_weights[i])
            
            # Create interaction between connected vertices
            start_idx1 = v1 * (self.output_dim // self.vertices)
            end_idx1 = (v1 + 1) * (self.output_dim // self.vertices)
            
            start_idx2 = v2 * (self.output_dim // self.vertices)
            end_idx2 = (v2 + 1) * (self.output_dim // self.vertices)
            
            # Connections with appropriate scaling
            influence1to2 = combined[:, start_idx1:end_idx1] * 0.5  # Or other scaling
            influence2to1 = combined[:, start_idx2:end_idx2] * 0.5
            
            combined[:, start_idx2:end_idx2] += edge_weight * influence1to2
            combined[:, start_idx1:end_idx1] += edge_weight * influence2to1
        
        # Apply activation formula
        sigma = 1.0  # Appropriate sigma for this solid
        combined = combined * torch.exp(-torch.pow(combined, 2) / sigma)
        
        # Apply solid-specific dynamics
        # (e.g., fire_factor, stability, ether_factor, silence_factor)
        
        # Final output projection
        output = self.output(combined)
        
        return output
    else:
        # Metal-accelerated implementation
        solid_type = self.__class__.__name__.lower().replace('layer', '')
        
        # Extract coefficients based on solid type
        if solid_type == "tetrahedron":
            coefficients = [self.fire_coefficient.item()]
        elif solid_type == "cube":
            coefficients = [self.stability_coefficient.item()]
        elif solid_type == "dodecahedron":
            coefficients = [self.ether_resonance.item()]
        elif solid_type == "icosahedron":
            coefficients = [self.silence_coefficient.item(), self.phase_coherence.item()]
        else:
            coefficients = [0.5]  # Default
        
        # Call the Metal-accelerated function
        return geometric_activation(x, solid_type, coefficients)
```

Add a configurable flag to enable/disable Metal acceleration:

```python
def __init__(self, input_dim, output_dim, use_metal=True):
    super().__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.use_metal = use_metal
    
    # Rest of your initialization code...
```

### 4. Integrate with ResonanceModule

Modify the ResonanceModule's forward method:

```python
def forward(self, x, t=None):
    # Original PyTorch implementation
    if not is_metal_available() or not self.use_metal:
        batch_size = x.shape[0]
        
        # Default time if not provided
        if t is None:
            t = torch.ones(batch_size, 1, device=x.device)
        
        # Initialize resonance output
        resonance = torch.zeros_like(x)
        
        # Generate resonance patterns
        for i in range(self.harmonics):
            freq = torch.sigmoid(self.frequencies[i]) * 10.0
            tau = torch.exp(self.decay_rates[i])
            
            # Phase based on input pattern
            phase = torch.matmul(x, self.phase_embedding.view(-1, 1)).view(batch_size, 1)
            
            # Calculate resonance term
            harmonic = self.amplitudes[i] * torch.cos(freq * self.phi**i * t + phase) * \
                       torch.exp(-(t**2) / (tau**2))
            
            # Add harmonic to resonance
            resonance = resonance + harmonic * x
            
        return resonance
    else:
        # Metal-accelerated implementation
        return apply_resonance(
            x, 
            self.frequencies, 
            self.decay_rates, 
            self.amplitudes, 
            self.phase_embedding, 
            t
        )
```

Add the use_metal flag to the constructor:

```python
def __init__(self, dim, harmonics=5, use_metal=True):
    super().__init__()
    self.dim = dim
    self.harmonics = harmonics
    self.phi = (1 + np.sqrt(5)) / 2
    self.use_metal = use_metal
    
    # Rest of your initialization code...
```

### 5. Integrate with CrystallineMutualityField

Modify the CrystallineMutualityField's forward method:

```python
def forward(self, weaving_field, dt=0.1):
    # Original PyTorch implementation
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

Add the use_metal flag to the constructor:

```python
def __init__(self, dim, grid_size=16, use_metal=True):
    super().__init__()
    self.dim = dim
    self.grid_size = grid_size
    self.use_metal = use_metal
    
    # Rest of your initialization code...
```

## Performance Optimization

1. **Batch Size**: Metal shaders perform best with larger batch sizes (8-32)
2. **Data Types**: For maximum performance, use `torch.float16` when possible:
   ```python
   input_tensor = input_tensor.half()  # Convert to half precision
   ```
3. **Avoid Unnecessary Transfers**: Keep data on the GPU as much as possible
   ```python
   # Good
   result1 = geometric_activation(tensor_on_gpu, "tetrahedron")
   result2 = apply_resonance(result1, ...)  # No CPU transfer between operations
   
   # Bad
   result1 = geometric_activation(tensor_on_gpu, "tetrahedron").cpu()
   result2 = apply_resonance(result1.to("mps"), ...)  # Unnecessary transfers
   ```
4. **Pre-compile Shaders**: For initialization-sensitive applications, warm up the shaders:
   ```python
   # In your model initialization
   def warmup_metal_shaders(self):
       dummy_input = torch.zeros(1, 64, device="mps")
       _ = geometric_activation(dummy_input, "tetrahedron")
       # Similar warmup for other operations
   ```

## Troubleshooting

### Common Issues

1. **"Metal not available" warning**:
   - Ensure you're on macOS with Apple Silicon
   - Check that PyTorch is installed with MPS support or MLX is installed

2. **Shader compilation failures**:
   - Make sure Metal developer tools are installed
   - Check macOS version (12+ recommended)

3. **Performance not improving**:
   - Verify data is actually on the GPU
   - Check batch sizes (too small = less benefit)
   - Ensure your main bottlenecks are in the accelerated operations

4. **Output mismatch between CPU and Metal**:
   - Small numerical differences are expected
   - Larger differences might indicate a shader parameter mismatch
   - Check coefficient extraction in the layer classes

5. **Memory issues**:
   - Metal has different memory management than PyTorch
   - For large models, try reducing batch size
   - Add explicit cleanup to free resources:
     ```python
     # After batches or when resetting
     torch.mps.empty_cache()
     ```

### Debugging Tips

1. Add temporary flags to easily switch between CPU and Metal:
   ```python
   # In your model class
   self.force_cpu = False  # Set to True to debug
   
   # In forward methods
   if self.force_cpu or not is_metal_available():
       # CPU implementation
   ```

2. Compare outputs between CPU and Metal implementations:
   ```python
   cpu_output = layer.forward_cpu(x)
   metal_output = layer.forward_metal(x)
   max_diff = torch.max(torch.abs(cpu_output - metal_output))
   print(f"Max difference: {max_diff}")
   ```
