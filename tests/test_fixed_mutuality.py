#!/usr/bin/env python3
"""
Test the fixed MutualityField.metal shader by directly calling the mutuality_field function.
This script verifies that the shader is working properly without needing the full model code.
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.append('.')

try:
    # Try importing from PyTorch for visualization if available
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import the Metal operations
from Python.metal_ops import mutuality_field, is_metal_available

def test_mutuality_field():
    """Test the mutuality_field function with various parameters."""
    print("Testing MutualityField.metal Shader")
    print("=================================")
    
    print(f"Metal available: {is_metal_available()}")
    
    if not is_metal_available():
        print("Metal is not available. Test aborted.")
        return
    
    # Test parameters
    batch_size = 2
    grid_sizes = [8, 16, 32]  # Test different grid sizes
    interference_scales = [0.1, 0.5, 1.0]  # Test different interference scales
    decay_rates = [0.05, 0.1, 0.2]  # Test different decay rates
    
    # Record test results
    results = []
    
    for grid_size in grid_sizes:
        for interference_scale in interference_scales:
            for decay_rate in decay_rates:
                print(f"\nTesting with grid_size={grid_size}, interference_scale={interference_scale}, decay_rate={decay_rate}")
                
                # Create test data
                input_dim = grid_size * grid_size
                x = np.random.randn(batch_size, input_dim).astype(np.float32)
                
                # Set time step
                dt = 0.1
                
                # Measure execution time
                start_time = time.time()
                
                # Run the mutuality field computation
                try:
                    result = mutuality_field(x, grid_size, interference_scale, decay_rate, dt)
                    elapsed_time = time.time() - start_time
                    
                    # Check if result is valid
                    is_valid = (
                        result is not None and
                        result.shape == x.shape and
                        not np.isnan(result).any() and
                        not np.isinf(result).any()
                    )
                    
                    if is_valid:
                        print(f"✅ Test passed: shape={result.shape}, min={result.min():.4f}, max={result.max():.4f}")
                        print(f"   Execution time: {elapsed_time:.4f} seconds")
                        
                        # Save result for later plotting
                        results.append({
                            'grid_size': grid_size,
                            'interference_scale': interference_scale,
                            'decay_rate': decay_rate,
                            'result': result,
                            'input': x,
                            'time': elapsed_time
                        })
                    else:
                        print(f"❌ Test failed: Invalid result")
                        print(f"   Result shape: {result.shape if result is not None else None}")
                        print(f"   Contains NaN: {np.isnan(result).any() if result is not None else None}")
                        print(f"   Contains Inf: {np.isinf(result).any() if result is not None else None}")
                except Exception as e:
                    print(f"❌ Test failed with exception: {e}")
    
    # Print summary
    print("\nTest Summary:")
    print(f"Successful tests: {len(results)}/{len(grid_sizes) * len(interference_scales) * len(decay_rates)}")
    
    # Plot results if we have matplotlib and some successful tests
    try:
        if len(results) > 0:
            # Create a directory for output images
            output_dir = Path("./test_results")
            output_dir.mkdir(exist_ok=True)
            
            # Plot the first successful result
            result_data = results[0]
            grid_size = result_data['grid_size']
            
            plt.figure(figsize=(15, 5))
            
            # Plot input
            plt.subplot(121)
            inp = result_data['input'][0].reshape(grid_size, grid_size)
            plt.imshow(inp, cmap='viridis')
            plt.colorbar()
            plt.title('Input')
            
            # Plot output
            plt.subplot(122)
            out = result_data['result'][0].reshape(grid_size, grid_size)
            plt.imshow(out, cmap='viridis')
            plt.colorbar()
            plt.title('Mutuality Field Output')
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(output_dir / "mutuality_field_test.png")
            print(f"\nVisualization saved to: {output_dir / 'mutuality_field_test.png'}")
            
            # Plot execution times
            plt.figure(figsize=(10, 6))
            grid_sizes_unique = sorted(set(r['grid_size'] for r in results))
            times = [r['time'] for r in results if r['interference_scale'] == 0.5 and r['decay_rate'] == 0.1]
            
            plt.bar(range(len(grid_sizes_unique)), times[:len(grid_sizes_unique)])
            plt.xticks(range(len(grid_sizes_unique)), [f"{gs}" for gs in grid_sizes_unique])
            plt.xlabel('Grid Size')
            plt.ylabel('Execution Time (seconds)')
            plt.title('Mutuality Field Execution Time vs. Grid Size')
            plt.savefig(output_dir / "mutuality_field_performance.png")
            print(f"Performance plot saved to: {output_dir / 'mutuality_field_performance.png'}")
    except Exception as e:
        print(f"Could not create visualization: {e}")

if __name__ == "__main__":
    test_mutuality_field()

