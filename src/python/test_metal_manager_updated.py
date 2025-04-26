#!/usr/bin/env python3
"""
Test script for updated Metal shader manager.

This script tests loading and initializing Metal shaders for the
crystalline consciousness model.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the updated Metal manager
from Python.metal_manager_updated import get_shader_manager, HAS_METAL

def test_metal_manager():
    """Test the Metal shader manager functionality."""
    print("Testing Updated Metal Shader Manager")
    print("===================================")
    
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
    else:
        print(f"Shader file not found: {mutuality_shader}")
    
    # Try creating a compute pipeline
    if "mutuality" in manager.libraries:
        if manager.create_compute_pipeline("mutuality", "reshape_to_grid"):
            print("Successfully created reshape_to_grid pipeline.")
        else:
            print("Failed to create reshape_to_grid pipeline.")
    
    # List all loaded libraries and pipelines
    print("\nLoaded Libraries:")
    for name, library in manager.libraries.items():
        print(f"- {name}")
    
    print("\nCreated Pipelines:")
    for name, pipeline in manager.pipelines.items():
        print(f"- {name}: max threads per group = {pipeline.maxTotalThreadsPerThreadgroup()}")

    # Try to create a buffer
    print("\nTesting buffer creation:")
    import numpy as np
    test_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    buffer = manager.create_buffer(test_data)
    if buffer is not None:
        print(f"Successfully created buffer with length: {buffer.length()}")
        
        # Read back the data
        output_data = manager.get_buffer_data(buffer, test_data.shape, test_data.dtype)
        if output_data is not None:
            print(f"Successfully read buffer data: {output_data}")
            if np.array_equal(test_data, output_data):
                print("Data integrity check passed!")
            else:
                print("Data integrity check failed!")
        else:
            print("Failed to read buffer data")
    else:
        print("Failed to create buffer")

if __name__ == "__main__":
    test_metal_manager()
