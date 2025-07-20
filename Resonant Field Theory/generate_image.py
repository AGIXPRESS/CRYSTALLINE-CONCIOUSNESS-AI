import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

try:
    import mlx.core as mx
    mx.set_default_device(mx.Device("mps"))
    print("Using MPS device for acceleration")
    test_array = mx.array([1, 2, 3])
    print(f"MLX device: {test_array.device}")
except Exception as e:
    print("MPS device not found, falling back to CPU.")

# 1. Define golden ratio
phi = (1 + np.sqrt(5)) / 2

# 2. Create a dense 2-D meshgrid (x, t)
x = np.linspace(0, 4 * np.pi, 1000)
t = np.linspace(0, 2 * np.pi, 500)
X, T = np.meshgrid(x, t)

# 3. Create a base resonant field
base = np.sin(X) + 0.3 * np.sin(phi * X)

# 4. Add multi-harmonic phi-spaced terms
phase = np.pi / 4
for k in range(1, 6):
    base += 0.5 / phi**k * np.sin(phi**k * X + k * phase)

# 5. Generate a secondary shifted copy to compute interference pattern
phase_shift = np.pi / 2
interf = np.cos(X / phi + phase_shift)
resonant_field = base + 0.2 * interf

# 6. Define holographic colormap
colors = ["#a8dadc", "#457b9d", "#1d3557", "#e63946"]
n_bins = 256  # Increased for smoother gradients
hues = np.linspace(0, 1, len(colors))**(1/phi)
holographic_cmap = LinearSegmentedColormap.from_list("holographic", colors, N=n_bins)

# 7. Enhanced plotting
plt.figure(figsize=(12, 8), dpi=200, facecolor='black')
#plt.plot(X[0], resonant_field[0], color='#2c3e50', alpha=0.8, linewidth=1.5)  # Use X[0] and resonant_field[0] to plot 1D data

# Add contourf three times at scales phi**0, phi**1, phi**2
levels = np.linspace(resonant_field.min(), resonant_field.max(), 100)
alphas = [0.3, 0.2, 0.15]
for i, alpha in enumerate(alphas):
    plt.contourf(X, T, resonant_field / phi**i, levels=levels, cmap=holographic_cmap, alpha=alpha)

# Overlay thin dashed white curves for additional phase-shifted harmonics
for k in range(1, 4):
    plt.plot(X[0], 0.1 * np.sin(k * X[0] + phase), color='white', linestyle='--', alpha=0.15)

# Mark phi-nodes: for i in range(-2, 4), plot vertical dashed lines at 2 * np.pi * phi**i
for i in range(-2, 4):
    node_x = 2 * np.pi * phi**i
    plt.axvline(x=node_x, color='white', linestyle='--', alpha=0.2)

# 8. Interference Overlay
I = np.sin(X) * np.sin(X / phi)
plt.imshow(I, extent=[0, 4*np.pi, 0, 2*np.pi], aspect='auto', origin='lower', cmap=holographic_cmap, alpha=0.15, interpolation='bilinear')

# Add labels and title
plt.title('Holographic Resonance Field Visualization', fontsize=16, color='white')
plt.xlabel('Phase (radians)', fontsize=12, color='white')
plt.ylabel('Time (radians)', fontsize=12, color='white')
plt.colorbar(label='Amplitude', orientation='vertical', shrink=0.8)
plt.tick_params(axis='x', colors='white')
plt.tick_params(axis='y', colors='white')

    """Returns φ-scaled sequence for n steps"""
def golden_spacing(n, base_step):
    """Returns φ-scaled sequence for n steps"""
    return [base_step * phi**k for k in range(n)]

def make_resonant_field(phi, X, T, harmonics=5):
    """
    Generates a resonant field based on the golden ratio and multiple harmonics.
    This function creates a base sine wave and adds several phi-scaled harmonics
    to simulate interference patterns and holographic effects.

    Parameters:
        phi (float): The golden ratio.
        X (np.array): The X spatial coordinates.
      T (np.array): The T spatial coordinates.
        harmonics (int): Number of harmonics to include.

    Returns:
        np.array: The resonant field data.

    TODO: Incorporate equations from Core-EQUATIONS for more accurate
          holographic field representation and quantum effects.
    """
    # 3. Create a base resonant field
    base = np.sin(X) + 0.3 * np.sin(phi * X)

    # 4. Add multi-harmonic phi-spaced terms
    phase = np.pi / 4
    for k in range(1, harmonics + 1):
        base += 0.5 / phi**k * np.sin(phi**k * X + k * phase)

    # 5. Generate a secondary shifted copy to compute interference pattern
    phase_shift = np.pi / 2
    interf = np.cos(X / phi + phase_shift)
    resonant_field = base + 0.2 * interf
    return resonant_field

# 6. Define holographic colormap
colors = ["#a8dadc", "#457b9d", "#1d3557", "#e63946"]
n_bins = 256  # Increased for smoother gradients
hues = np.linspace(0, 1, len(colors))**(1/phi)
holographic_cmap = LinearSegmentedColormap.from_list("holographic", colors, N=n_bins)

# 7. Enhanced plotting
plt.figure(figsize=(12, 8), dpi=200, facecolor='black')

# 2. Create a dense 2-D meshgrid (x, t)
x = np.linspace(0, 4 * np.pi, 1000)
t = np.linspace(0, 2 * np.pi, 500)
X, T = np.meshgrid(x, t)

# Generate resonant field data
resonant_field = make_resonant_field(phi, X, T)

# Add contourf three times at scales phi**0, phi**1, phi**2
levels = np.linspace(resonant_field.min(), resonant_field.max(), 100)
alphas = [0.3, 0.2, 0.15]
for i, alpha in enumerate(alphas):
    plt.contourf(X, T, resonant_field / phi**i, levels=levels, cmap=holographic_cmap, alpha=alpha)

# 8. Interference Overlay
I = np.sin(X) * np.sin(X / phi)
plt.imshow(I, extent=[0, 4*np.pi, 0, 2*np.pi], aspect='auto', origin='lower', cmap=holographic_cmap, alpha=0.15, interpolation='bilinear')

# Mark phi-nodes: for i in range(-2, 4), plot vertical dashed lines at 2 * np.pi * phi**i
for i in range(-2, 4):
    node_x = 2 * np.pi * phi**i
    plt.axvline(x=node_x, color='white', linestyle='--', alpha=0.2)

# Add labels and title
plt.title('Holographic Resonance Field Visualization', fontsize=16, color='white')
plt.xlabel('Phase (radians)', fontsize=12, color='white')
plt.ylabel('Time (radians)', fontsize=12, color='white')
plt.colorbar(label='Amplitude', orientation='vertical', shrink=0.8)
plt.tick_params(axis='x', colors='white')
plt.tick_params(axis='y', colors='white')

# Apply golden spacing to x-ticks
x_ticks = golden_spacing(5, np.pi)
plt.xticks(x_ticks)

# Save the image
plt.savefig('holographic_field.png', dpi=450, transparent=False, bbox_inches='tight', pad_inches=0.05)
print("Image saved as 'holographic_field.png'")
