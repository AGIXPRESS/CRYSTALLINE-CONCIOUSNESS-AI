# Testing the Shader in the Full Crystalline Consciousness Model

To test the fixed MutualityField.metal shader in the full crystalline consciousness model, follow these steps. For detailed environment-specific configuration options, please refer to the [Runtime Configuration Guide](runtime_config_guide.md).

## 1. Update Python Wrapper (if needed)
The current Python wrapper (metal_manager_updated.py) has issues with buffer creation. You may need to:
- Use the new metal_manager_ctypes.py we started implementing
- Or use the Swift-based approach for critical parts of the model

## 2. Integration Steps
1. Update import statements in your model files to use the fixed Metal operations:
   ```python
   from crystalline_mlx.Python.metal_ops import (
       geometric_activation,
       apply_resonance,
       mutuality_field,
       is_metal_available
   )
   ```

2. Ensure the CrystallineMutualityField class properly allocates intermediate buffers:
   ```python
   # When creating layer1_output_r and layer1_output_t buffers
   layer1_output_r = np.zeros((batch_size * 8 * grid_size * grid_size), dtype=np.float32)
   layer1_output_t = np.zeros((batch_size * 8 * grid_size * grid_size), dtype=np.float32)
   
   # Pass these as additional parameters to process_interference_fields
   ```

3. Update any code that calls or creates pipelines to match the fixed parameter types

## 3. Testing Approach
1. **Unit Testing**: Test the mutuality field component in isolation:
   ```python
   # Create a test instance of CrystallineMutualityField
   field = CrystallineMutualityField(dim=64, grid_size=16)
   
   # Create test input
   test_input = torch.randn(2, 64)
   
   # Run with Metal acceleration
   metal_output = field(test_input)
   
   # Compare with CPU implementation
   field.use_metal = False
   cpu_output = field(test_input)
   
   # Check if results are close (allowing for floating-point differences)
   assert torch.allclose(metal_output, cpu_output, rtol=1e-3, atol=1e-3)
   ```

2. **Integration Testing**: Test the full model with our fixed shader:
   ```python
   # Load your full crystalline model
   model = CrystallineModel(...)
   
   # Process some test input
   input_data = ...
   output = model(input_data)
   
   # Verify the output makes sense (e.g., no NaNs, reasonable values)
   assert not torch.isnan(output).any()
   ```

3. **Performance Testing**: Measure the performance improvement:
   ```python
   import time
   
   # With Metal
   model.use_metal = True
   start = time.time()
   for _ in range(100):
       output = model(input_data)
   metal_time = time.time() - start
   
   # Without Metal
   model.use_metal = False
   start = time.time()
   for _ in range(100):
       output = model(input_data)
   cpu_time = time.time() - start
   
   print(f'Metal speedup: {cpu_time / metal_time:.2f}x')
   ```

   For standardized performance testing with different batch sizes, input dimensions, and proper thresholds,
   refer to the [Runtime Configuration Guide](runtime_config_guide.md#3-performance-monitoring).

## 4. Debugging Tips
1. Add logging to track which code path is being used:
   ```python
   print(f'Using Metal: {is_metal_available() and self.use_metal}')
   ```

2. Implement a fallback mechanism:
   ```python
   try:
       # Try Metal implementation
       if is_metal_available() and self.use_metal:
           return mutuality_field(...)
   except Exception as e:
       print(f'Metal implementation failed: {e}, falling back to CPU')
       # Fallback to CPU implementation
       return self._cpu_implementation(...)
   ```

3. Add visualization of intermediate results:
   ```python
   # Visualize the mutual field
   import matplotlib.pyplot as plt
   plt.figure(figsize=(10, 5))
   plt.subplot(121)
   plt.imshow(mutual_field[0, 0].cpu().numpy())
   plt.title('Mutual Field (Metal)')
   plt.colorbar()
   plt.subplot(122)
   plt.imshow(cpu_mutual_field[0, 0].numpy())
   plt.title('Mutual Field (CPU)')
   plt.colorbar()
   plt.savefig('mutual_field_comparison.png')
   ```

By following these steps, you should be able to verify that the fixed shader works correctly in the full crystalline consciousness model context.

## Additional Resources

For comprehensive instructions on configuring your test environment, including development, CI, and production settings, hardware resource management, and performance monitoring, please refer to the [Runtime Configuration Guide](runtime_config_guide.md).
