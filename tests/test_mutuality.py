import sys
sys.path.append('.')
import numpy as np
from Python.metal_ops import mutuality_field, is_metal_available

print(f"Metal available: {is_metal_available()}")

# Test different grid sizes
for grid_size in [8, 16, 32]:
    # Test data
    batch_size = 2
    input_dim = grid_size * grid_size
    x = np.random.randn(batch_size, input_dim).astype(np.float32)
    
    # Parameters
    interference_scale = 1.0
    decay_rate = 0.05
    dt = 0.1
    
    # Test mutuality field
    result = mutuality_field(x, grid_size, interference_scale, decay_rate, dt)
    print(f"Grid size {grid_size}: shape={result.shape}, min={result.min():.4f}, max={result.max():.4f}")
