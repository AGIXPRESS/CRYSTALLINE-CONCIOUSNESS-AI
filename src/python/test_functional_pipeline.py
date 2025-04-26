#!/usr/bin/env python3
"""
Functional test for the MutualityField Metal shader pipeline.
This test creates input data and runs it through the entire pipeline to verify
that the shader works correctly end-to-end.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the Metal manager
from Python.metal_manager_updated import get_shader_manager, HAS_METAL

def functional_test():
    """Run a functional test on the MutualityField Metal shader pipeline."""
    print("MutualityField Metal Shader Functional Test")
    print("==========================================")
    
    # Check if Metal is available
    if not HAS_METAL:
        print("Metal is not available. Test skipped.")
        return
    
    # Get shader directory
    shader_dir = os.path.join(Path(__file__).parent.parent, "Shaders")
    print(f"Shader directory: {shader_dir}")
    
    # Initialize shader manager
    manager = get_shader_manager(shader_dir)
    
    if manager.device is None:
        print("Failed to initialize Metal device.")
        return
    
    print(f"Metal device initialized: {manager.device.name()}")
    
    # Load mutuality field shader
    mutuality_shader = os.path.join(shader_dir, "MutualityField.metal")
    if os.path.exists(mutuality_shader):
        if manager.load_shader_library(mutuality_shader, "mutuality"):
            print("Successfully loaded mutuality field shader.")
        else:
            print("Failed to load mutuality field shader.")
            return
    else:
        print(f"Shader file not found: {mutuality_shader}")
        return
    
    # Create all necessary pipelines
    pipelines = [
        "reshape_to_grid",
        "create_shifted_fields",
        "create_interference_patterns",
        "process_interference_fields",
        "combine_interference_patterns",
        "apply_persistence",
        "flatten_to_vector"
    ]
    
    for pipeline in pipelines:
        if not manager.create_compute_pipeline("mutuality", pipeline):
            print(f"Failed to create {pipeline} pipeline. Test aborted.")
            return
    
    print("All pipelines created successfully.")
    
    # Test parameters
    batch_size = 2
    input_dim = 32
    grid_size = 8  # Small grid for testing
    output_dim = input_dim
    
    # Create test input data (random values between 0 and 1)
    input_data = np.random.rand(batch_size, input_dim).astype(np.float32)
    print(f"Created input data with shape: {input_data.shape}")
    
    # 1. Execute reshape_to_grid
    print("\nStep 1: Reshaping input to grid...")
    input_buffer = manager.create_buffer(input_data)
    grid_output = np.zeros((batch_size, 1, grid_size, grid_size), dtype=np.float32)
    grid_buffer = manager.create_buffer(grid_output)
    
    # Create parameter buffers
    batch_buffer = manager.create_buffer(np.array([batch_size], dtype=np.uint32))
    input_dim_buffer = manager.create_buffer(np.array([input_dim], dtype=np.uint32))
    grid_size_buffer = manager.create_buffer(np.array([grid_size], dtype=np.uint32))
    
    # Execute reshape_to_grid
    success = manager.execute_shader(
        "reshape_to_grid",
        [input_buffer, grid_buffer, batch_buffer, input_dim_buffer, grid_size_buffer],
        [],
        (batch_size, grid_size * grid_size, 1),
        (1, 1, 1)
    )
    
    if not success:
        print("Failed to execute reshape_to_grid. Test aborted.")
        return
    
    # Read back the data
    grid_output = manager.get_buffer_data(grid_buffer, grid_output.shape, grid_output.dtype)
    print(f"Grid output shape: {grid_output.shape}")
    print("First few values of grid output:")
    print(grid_output[0, 0, 0, :4])
    
    # 2. Execute create_shifted_fields
    print("\nStep 2: Creating shifted fields...")
    r_shifted_output = np.zeros_like(grid_output)
    t_shifted_output = np.zeros_like(grid_output)
    r_shifted_buffer = manager.create_buffer(r_shifted_output)
    t_shifted_buffer = manager.create_buffer(t_shifted_output)
    
    # Execute create_shifted_fields
    success = manager.execute_shader(
        "create_shifted_fields",
        [grid_buffer, r_shifted_buffer, t_shifted_buffer, batch_buffer, grid_size_buffer],
        [],
        (batch_size, grid_size, grid_size),
        (1, 1, 1)
    )
    
    if not success:
        print("Failed to execute create_shifted_fields. Test aborted.")
        return
    
    # Read back the data
    r_shifted_output = manager.get_buffer_data(r_shifted_buffer, r_shifted_output.shape, r_shifted_output.dtype)
    t_shifted_output = manager.get_buffer_data(t_shifted_buffer, t_shifted_output.shape, t_shifted_output.dtype)
    print(f"R-shifted output shape: {r_shifted_output.shape}")
    print(f"T-shifted output shape: {t_shifted_output.shape}")
    
    # 3. Execute create_interference_patterns
    print("\nStep 3: Creating interference patterns...")
    r_interference_output = np.zeros((batch_size, 2, grid_size, grid_size), dtype=np.float32)
    t_interference_output = np.zeros((batch_size, 2, grid_size, grid_size), dtype=np.float32)
    r_interference_buffer = manager.create_buffer(r_interference_output)
    t_interference_buffer = manager.create_buffer(t_interference_output)
    
    # Execute create_interference_patterns
    success = manager.execute_shader(
        "create_interference_patterns",
        [grid_buffer, r_shifted_buffer, t_shifted_buffer, r_interference_buffer, t_interference_buffer, batch_buffer, grid_size_buffer],
        [],
        (batch_size, grid_size, grid_size),
        (1, 1, 1)
    )
    
    if not success:
        print("Failed to execute create_interference_patterns. Test aborted.")
        return
    
    # Read back the data
    r_interference_output = manager.get_buffer_data(r_interference_buffer, r_interference_output.shape, r_interference_output.dtype)
    t_interference_output = manager.get_buffer_data(t_interference_buffer, t_interference_output.shape, t_interference_output.dtype)
    print(f"R-interference output shape: {r_interference_output.shape}")
    print(f"T-interference output shape: {t_interference_output.shape}")
    
    # 4. Execute process_interference_fields
    print("\nStep 4: Processing interference fields...")
    r_processed_output = np.zeros((batch_size, 1, grid_size, grid_size), dtype=np.float32)
    t_processed_output = np.zeros((batch_size, 1, grid_size, grid_size), dtype=np.float32)
    r_processed_buffer = manager.create_buffer(r_processed_output)
    t_processed_buffer = manager.create_buffer(t_processed_output)
    
    # Create intermediate layer buffers
    layer1_output_r = np.zeros((batch_size * 8 * grid_size * grid_size), dtype=np.float32)
    layer1_output_t = np.zeros((batch_size * 8 * grid_size * grid_size), dtype=np.float32)
    layer1_output_r_buffer = manager.create_buffer(layer1_output_r)
    layer1_output_t_buffer = manager.create_buffer(layer1_output_t)
    
    # Execute process_interference_fields
    success = manager.execute_shader(
        "process_interference_fields",
        [r_interference_buffer, t_interference_buffer, r_processed_buffer, t_processed_buffer, 
         layer1_output_r_buffer, layer1_output_t_buffer, batch_buffer, grid_size_buffer],
        [],
        (batch_size, 1, 1),
        (1, 1, 1)
    )
    
    if not success:
        print("Failed to execute process_interference_fields. Test aborted.")
        return
    
    # Read back the data
    r_processed_output = manager.get_buffer_data(r_processed_buffer, r_processed_output.shape, r_processed_output.dtype)
    t_processed_output = manager.get_buffer_data(t_processed_buffer, t_processed_output.shape, t_processed_output.dtype)
    print(f"R-processed output shape: {r_processed_output.shape}")
    print(f"T-processed output shape: {t_processed_output.shape}")
    
    # 5. Execute combine_interference_patterns
    print("\nStep 5: Combining interference patterns...")
    mutual_field_output = np.zeros((batch_size, 1, grid_size, grid_size), dtype=np.float32)
    mutual_field_buffer = manager.create_buffer(mutual_field_output)
    
    # Create parameter buffer for interference scale
    interference_scale = 0.5  # Example value
    interference_scale_buffer = manager.create_buffer(np.array([interference_scale], dtype=np.float32))
    
    # Execute combine_interference_patterns
    success = manager.execute_shader(
        "combine_interference_patterns",
        [r_processed_buffer, t_processed_buffer, grid_buffer, mutual_field_buffer, 
         batch_buffer, grid_size_buffer, interference_scale_buffer],
        [],
        (batch_size, grid_size, grid_size),
        (1, 1, 1)
    )
    
    if not success:
        print("Failed to execute combine_interference_patterns. Test aborted.")
        return
    
    # Read back the data
    mutual_field_output = manager.get_buffer_data(mutual_field_buffer, mutual_field_output.shape, mutual_field_output.dtype)
    print(f"Mutual field output shape: {mutual_field_output.shape}")
    
    # 6. Execute apply_persistence
    print("\nStep 6: Applying persistence...")
    persistence_state = np.zeros_like(mutual_field_output)
    persistence_buffer = manager.create_buffer(persistence_state)
    
    # Create parameter buffers for decay rate and dt
    decay_rate = 0.1  # Example value
    dt = 0.5  # Example value
    decay_rate_buffer = manager.create_buffer(np.array([decay_rate], dtype=np.float32))
    dt_buffer = manager.create_buffer(np.array([dt], dtype=np.float32))
    
    # Execute apply_persistence
    success = manager.execute_shader(
        "apply_persistence",
        [mutual_field_buffer, persistence_buffer, batch_buffer, grid_size_buffer, 
         decay_rate_buffer, dt_buffer],
        [],
        (batch_size, grid_size, grid_size),
        (1, 1, 1)
    )
    
    if not success:
        print("Failed to execute apply_persistence. Test aborted.")
        return
    
    # Read back the data
    persistence_output = manager.get_buffer_data(persistence_buffer, persistence_state.shape, persistence_state.dtype)
    print(f"Persistence output shape: {persistence_output.shape}")
    
    # 7. Execute flatten_to_vector
    print("\nStep 7: Flattening to vector...")
    vector_output = np.zeros((batch_size, output_dim), dtype=np.float32)
    vector_buffer = manager.create_buffer(vector_output)
    
    # Create parameter buffer for output_dim
    output_dim_buffer = manager.create_buffer(np.array([output_dim], dtype=np.uint32))
    
    # Execute flatten_to_vector
    success = manager.execute_shader(
        "flatten_to_vector",
        [persistence_buffer, vector_buffer, batch_buffer, grid_size_buffer, output_dim_buffer],
        [],
        (batch_size, output_dim),
        (1, 1, 1)
    )
    
    if not success:
        print("Failed to execute flatten_to_vector. Test aborted.")
        return
    
    # Read back the data
    vector_output = manager.get_buffer_data(vector_buffer, vector_output.shape, vector_output.dtype)
    print(f"Final vector output shape: {vector_output.shape}")
    
    # Print original input and final output for comparison
    print("\nComparison of first few values:")
    print(f"Original input (first batch): {input_data[0, :4]}")
    print(f"Final output (first batch): {vector_output[0, :4]}")
    
    # Simple verification - check that outputs are non-zero and in a reasonable range
    all_zeros = np.all(vector_output == 0)
    any_nan = np.any(np.isnan(vector_output))
    any_inf = np.any(np.isinf(vector_output))
    
    if all_zeros or any_nan or any_inf:
        print("\n❌ TEST FAILED: Output contains all zeros, NaN, or infinite values.")
    else:
        print("\n✅ TEST PASSED: Output values are reasonable.")
        print("All kernels executed successfully!")

if __name__ == "__main__":
    functional_test()

