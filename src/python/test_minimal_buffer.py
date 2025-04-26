#!/usr/bin/env python3
"""
Minimal test for Metal buffer creation using direct PyObjC calls.
"""

import sys
import numpy as np
import ctypes

# Try to import Metal
try:
    import objc
    objc.loadBundle('Metal', globals(), '/System/Library/Frameworks/Metal.framework')
    import Metal
    from Foundation import NSData
    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    print("PyObjC Metal framework not found. Test cannot run.")
    sys.exit(1)

def test_minimal_buffer():
    """Test minimal buffer creation with direct Metal API calls."""
    print("Testing Minimal Metal Buffer Creation")
    print("===================================")
    
    # Create Metal device
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        print("Failed to create Metal device.")
        return
    
    print(f"Metal device initialized: {device.name()}")
    
    # Test data
    test_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    data_size = test_data.nbytes
    print(f"Test data: {test_data}, size: {data_size} bytes")
    
    # Try multiple approaches to create a Metal buffer
    
    print("\nApproach 1: Using newBufferWithLength_options_")
    try:
        # Standard approach
        buffer1 = device.newBufferWithLength_options_(
            data_size, Metal.MTLResourceStorageModeShared)
        
        if buffer1 is not None:
            print("✓ Buffer created successfully")
            
            # Copy data to buffer
            buffer_ptr = buffer1.contents()
            if buffer_ptr is not None:
                data_ptr = test_data.ctypes.data_as(ctypes.c_void_p)
                ctypes.memmove(buffer_ptr, data_ptr, data_size)
                print("✓ Data copied to buffer")
            else:
                print("✗ Failed to get buffer contents")
        else:
            print("✗ Failed to create buffer")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\nApproach 2: Using newBufferWithBytes_length_options_")
    try:
        # Direct data copy approach
        data_ptr = test_data.ctypes.data_as(ctypes.c_void_p)
        buffer2 = device.newBufferWithBytes_length_options_(
            data_ptr, data_size, Metal.MTLResourceStorageModeShared)
        
        if buffer2 is not None:
            print("✓ Buffer created successfully")
        else:
            print("✗ Failed to create buffer")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\nApproach 3: Using NSData")
    try:
        # Using NSData as an intermediate
        ns_data = NSData.dataWithBytes_length_(
            test_data.ctypes.data_as(ctypes.c_void_p), data_size)
        
        if ns_data is not None:
            print("✓ NSData created successfully")
            
            buffer3 = device.newBufferWithLength_options_(
                data_size, Metal.MTLResourceStorageModeShared)
            
            if buffer3 is not None:
                print("✓ Buffer created successfully")
                
                # Copy from NSData to buffer
                buffer_ptr = buffer3.contents()
                if buffer_ptr is not None:
                    ns_data_ptr = ns_data.bytes()
                    ctypes.memmove(buffer_ptr, ns_data_ptr, data_size)
                    print("✓ Data copied to buffer")
                else:
                    print("✗ Failed to get buffer contents")
            else:
                print("✗ Failed to create buffer")
        else:
            print("✗ Failed to create NSData")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\nBuffer creation test complete.")

if __name__ == "__main__":
    test_minimal_buffer()

