import sys
sys.path.append('.')
import numpy as np
from Python.metal_ops import geometric_activation, is_metal_available

print(f"Metal available: {is_metal_available()}")

# Test data
x = np.random.randn(4, 128).astype(np.float32)

# Test each solid type
for solid_type in ["tetrahedron", "cube", "dodecahedron", "icosahedron"]:
    result = geometric_activation(x, solid_type)
    print(f"{solid_type}: shape={result.shape}, min={result.min():.4f}, max={result.max():.4f}")
