#!/usr/bin/env python3
"""
Test script for Metal module import.
"""

import sys
import os
import importlib

def print_path_info():
    """Print information about the Python path."""
    print("Python Path:")
    for path in sys.path:
        print(f"- {path}")
        if os.path.exists(path):
            print("  (exists)")
        else:
            print("  (does not exist)")

def test_import(module_name):
    """Test importing a module and print any errors."""
    print(f"\nTrying to import {module_name}...")
    try:
        module = importlib.import_module(module_name)
        print(f"Successfully imported {module_name}")
        return module
    except ImportError as e:
        print(f"Import error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def print_module_info(module_name):
    """Print information about a module."""
    try:
        module = importlib.import_module(module_name)
        print(f"\nModule info for {module_name}:")
        print(f"- File: {module.__file__}")
        print("- Dir:")
        for item in sorted(dir(module)):
            if not item.startswith('__'):
                print(f"  - {item}")
    except ImportError as e:
        print(f"Could not import {module_name}: {e}")

def main():
    """Run tests."""
    print("Metal Import Test")
    print("================")
    
    print_path_info()
    
    # Test importing the PyObjC packages
    print("\nTesting PyObjC imports:")
    test_import("objc")
    test_import("Foundation")
    test_import("Metal")
    test_import("Quartz")
    
    # Check module contents
    print("\nChecking Metal module contents:")
    print_module_info("Metal")
    
    # Try a direct import of Metal
    try:
        print("\nTrying direct import of Metal...")
        from Metal import MTLCreateSystemDefaultDevice, MTL
        print("Metal imported successfully!")
        
        # Try to create a Metal device
        print("\nTrying to create a Metal device...")
        device = MTLCreateSystemDefaultDevice()
        if device:
            print(f"Metal device created: {device.name()}")
        else:
            print("Failed to create Metal device!")
    except ImportError as e:
        print(f"Metal import error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()

