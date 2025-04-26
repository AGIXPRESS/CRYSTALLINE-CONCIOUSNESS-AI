import sys
sys.path.append('.')
import numpy as np
from Python.metal_ops import apply_resonance, PHI, is_metal_available

print(f"Metal available: {is_metal_available()}")

# Test data
batch_size = 2
input_dim = 128
x = np.random.randn(batch_size, input_dim).astype(np.float32)

# Create resonance parameters
harmonics = 3
frequencies = np.random.randn(harmonics).astype(np.float32)
decay_rates = np.random.randn(harmonics).astype(np.float32)
amplitudes = np.array([1.0, 1.0/PHI, 1.0/PHI**2], dtype=np.float32)
phase_embedding = np.random.randn(input_dim).astype(np.float32)

# Test resonance patterns
result = apply_resonance(x, frequencies, decay_rates, amplitudes, phase_embedding)
print(f"Resonance: shape={result.shape}, min={result.min():.4f}, max={result.max():.4f}")
