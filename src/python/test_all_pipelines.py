#!/usr/bin/env python3
"""
Test script for all pipelines in the MutualityField.metal shader.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the Metal manager
from Python.metal_manager_updated import get_shader_manager, HAS_METAL

def test_all_pipelines():
    """Test creating all pipelines from MutualityField.metal."""
    print("Testing All MutualityField Pipelines")
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
            return
    else:
        print(f"Shader file not found: {mutuality_shader}")
        return
    
    # Define all the functions in MutualityField.metal
    functions = [
        "reshape_to_grid",
        "create_shifted_fields",
        "create_interference_patterns",
        "process_interference_fields",
        "combine_interference_patterns",
        "apply_persistence",
        "flatten_to_vector"
    ]
    
    # Try creating all compute pipelines
    success_count = 0
    fail_count = 0
    
    print("\nCreating pipelines:")
    for func_name in functions:
        if manager.create_compute_pipeline("mutuality", func_name):
            print(f"✓ Successfully created {func_name} pipeline.")
            success_count += 1
        else:
            print(f"✗ Failed to create {func_name} pipeline.")
            fail_count += 1
    
    # Summary
    print(f"\nSummary: {success_count} successful, {fail_count} failed")
    
    if fail_count == 0:
        print("All pipelines created successfully!")
    
    # List all created pipelines
    print("\nCreated Pipelines:")
    for name, pipeline in manager.pipelines.items():
        print(f"- {name}: max threads per group = {pipeline.maxTotalThreadsPerThreadgroup()}")

if __name__ == "__main__":
    test_all_pipelines()

