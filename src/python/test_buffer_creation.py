#!/usr/bin/env python3
"""
Simple test for buffer creation in the Metal shader manager.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the Metal manager
from Python.metal_manager_updated import get_shader_manager, HAS_METAL

def test_buffer_creation():
    """Test the buffer creation functionality."""
    print("Testing Metal Buffer Creation")
    print("===========================")
    
    # Check if Metal is available
    if not HAS_METAL:
        print("Metal is not available. Test skipped.")
        return
    
    # Initialize shader manager
    manager = get_shader_manager()
    
    if manager.device is None:
        print("Failed to initialize Metal device.")
        return
    
    print(f"Metal device initialized: {manager.device.name()}")
    
    # Try different ways to create a buffer
    print("\nTesting different buffer creation methods:")
    
    # 1. Simple float32 array
    test_data1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    print(f"Test data 1: {test_data1}, shape: {test_data1.shape}, dtype: {test_data1.dtype}")
    buffer1 = manager.create_buffer(test_data1)
    print(f"Buffer 1 created: {buffer1 is not None}")
    
    if buffer1 is not None:
        # Try to read the data back
        read_data1 = manager.get_buffer_data(buffer1, test_data1.shape, test_data1.dtype)
        print(f"Read data 1: {read_data1 is not None}")
        if read_data1 is not None:
            print(f"Data 1 values: {read_data1}")
    
    # 2. 2D array
    test_data2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    print(f"\nTest data 2: shape: {test_data2.shape}, dtype: {test_data2.dtype}")
    buffer2 = manager.create_buffer(test_data2)
    print(f"Buffer 2 created: {buffer2 is not None}")
    
    if buffer2 is not None:
        # Try to read the data back
        read_data2 = manager.get_buffer_data(buffer2, test_data2.shape, test_data2.dtype)
        print(f"Read data 2: {read_data2 is not None}")
        if read_data2 is not None:
            print(f"Data 2 values: {read_data2}")
    
    # 3. Try with explicit length
    test_data3 = np.array([1, 2, 3, 4], dtype=np.int32)
    print(f"\nTest data 3: shape: {test_data3.shape}, dtype: {test_data3.dtype}")
    buffer3 = manager.create_buffer(test_data3, length=test_data3.nbytes)
    print(f"Buffer 3 created: {buffer3 is not None}")
    
    # 4. Try with a small array and a larger buffer
    test_data4 = np.array([1.0, 2.0], dtype=np.float32)
    print(f"\nTest data 4: shape: {test_data4.shape}, dtype: {test_data4.dtype}")
    # Create a buffer 4x the size needed
    buffer4 = manager.create_buffer(test_data4, length=test_data4.nbytes * 4)
    print(f"Buffer 4 created: {buffer4 is not None}")
    
    print("\nBuffer creation test complete.")

if __name__ == "__main__":
    test_buffer_creation()

