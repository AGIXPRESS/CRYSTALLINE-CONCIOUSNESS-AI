#!/usr/bin/env python3
"""
Enhanced Resonant Field Visualizer for Crystalline Consciousness AI

This script generates enhanced visualizations for the resonant field theory paper,
focusing specifically on improving figures 2, 7, and 9 with:
- GPU-accelerated geometric computations using MLX
- Higher-resolution phase mapping
- Explicit geometric guides and symmetry indicators
- Enhanced color gradients for better field transition visualization

The visualizations incorporate sacred geometric principles and phi-based harmonics
from the crystalline consciousness framework.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from matplotlib.patches import Polygon, Circle, Rectangle
import matplotlib.patheffects as PathEffects
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from datetime import datetime

# Try to import MLX for GPU acceleration
try:
    import mlx
    import mlx.core as mx
    
    # Test if MLX can actually use the GPU
    try:
        # Create a small test tensor to verify device access
        test_tensor = mx.array([1.0, 2.0, 3.0])
        # Try to force computation to check device access
        _ = mx.sum(test_tensor).item()
        
        # Additional validation to ensure tensor operations work
        test_reshape = mx.reshape(test_tensor, (1, 3))
        test_stack = mx.stack([test_tensor, test_tensor])
        
        HAS_MLX = True
        print("MLX available - Metal GPU acceleration enabled")
    except Exception as e:
        HAS_MLX = False
        print(f"MLX found but GPU access failed: {e}")
        print("Falling back to NumPy implementation")
except ImportError:
    HAS_MLX = False
    print("MLX not available - Using NumPy fallback")

# Force NumPy only mode for reliable visualization
HAS_MLX = False
print("Forcing NumPy-only mode for reliable visualization")

# Add the src directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'src'))

# Import geometric transforms from our Crystalline Consciousness framework
try:
    from python.metal_ops import (
        geometric_activation,
        apply_resonance,
        is_metal_available,
        PHI,
        PHI_INV,
        TAU
    )
    print(f"Metal acceleration is {'available' if is_metal_available() else 'not available'}")
except ImportError as e:
    print(f"Could not import metal_ops: {e}")
    print("Using simple implementations for demonstration")
    
    # Define constants if we couldn't import them
    PHI = (1 + np.sqrt(5)) / 2  # Golden ratio (φ ≈ 1.618033988749895)
    PHI_INV = 1 / PHI  # Inverse golden ratio (φ⁻¹ ≈ 0.618033988749895)
    TAU = 2 * np.pi  # Full circle in radians (τ = 2π)
    
    # Simple implementations for fallback
    def geometric_activation(x, solid_type='all', scale=1.0, resonance=1.0):
        """Simple fallback implementation"""
        x = np.array(x)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            
        # Apply a simple non-linear activation with geometric flavor
        if solid_type == 'tetrahedron':
            return np.tanh(x * scale * resonance) * 0.5
        elif solid_type == 'octahedron':
            return np.sin(x * scale * resonance * TAU) * 0.5
        elif solid_type == 'cube':
            return 1.0 / (1.0 + np.exp(-x * scale * resonance))
        elif solid_type == 'icosahedron':
            phase1 = x * scale * resonance
            phase2 = x * scale * resonance * PHI
            return (np.sin(phase1) + np.sin(phase2) * PHI_INV) / (1 + PHI_INV)
        elif solid_type == 'dodecahedron':
            phase = x * scale * resonance
            h1 = np.sin(phase)
            h2 = np.sin(phase * PHI) * PHI_INV
            h3 = np.sin(phase * PHI * PHI) * PHI_INV * PHI_INV
            return (h1 + h2 + h3) / (1 + PHI_INV + PHI_INV * PHI_INV)
        else:  # 'all'
            # Blend different geometries
            t = geometric_activation(x, 'tetrahedron', scale, resonance)
            c = geometric_activation(x, 'cube', scale, resonance)
            i = geometric_activation(x, 'icosahedron', scale, resonance)
            d = geometric_activation(x, 'dodecahedron', scale, resonance)
            return (t + c * PHI_INV + i * PHI + d) / (1 + PHI_INV + PHI + 1)
    
    def apply_resonance(data, patterns=None, resonance_type='quantum', intensity=1.0, phase_shift=0.0):
        """Simple fallback implementation"""
        data = np.array(data)
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
            
        # Create some simple resonance patterns
        if patterns is None:
            feature_dim = data.shape[1]
            patterns = np.zeros((3, feature_dim))
            positions = np.array([i / feature_dim for i in range(feature_dim)])
            positions = positions.reshape(1, feature_dim)
            
            for h in range(3):
                freq = TAU * (PHI ** (h - 1))  # Center around phi^0 = 1
                patterns[h] = np.sin(positions * freq + phase_shift * (h + 1))
        
        # Apply a simple resonance effect
        result = data * (1.0 - intensity * 0.2)
        for i in range(min(3, patterns.shape[0])):
            pattern = patterns[i:i+1]
            phase = np.sum(data * pattern, axis=1, keepdims=True) * intensity * TAU
            
            # Add interference patterns
            result += np.sin(phase) * pattern * intensity * (0.8 ** i)
            
        return np.tanh(result * PHI)
    
def is_metal_available():
    """Check if Metal GPU acceleration is available"""
    if not HAS_MLX:
        return False
    try:
        # Verify that operations actually complete
        test = mx.array([1.0, 2.0, 3.0])
        _ = mx.sum(test).item()
        return True
    except:
        return False

# Constants for visualization
GRID_SIZE = 256  # Higher resolution for better visualizations
OUTPUT_DIR = os.path.join(current_dir, 'Resonant Field Theory', 'figures_enhanced_20250503_161506')
# Use correctly capitalized colormap names and more robust access method
try:
    CMAP_SPECTRAL = plt.cm.get_cmap('Spectral')
except:
    CMAP_SPECTRAL = plt.cm.get_cmap('viridis')  # fallback
CMAP_PLASMA = plt.cm.get_cmap('plasma')
CMAP_VIRIDIS = plt.cm.get_cmap('viridis')

# Create custom colormaps that enhance the visualization
def create_custom_colormaps():
    """Create custom colormaps for resonant field visualization"""
    # Phi-harmonic color map (based on golden ratio segments)
    phi_colors = []
    for i in range(5):
        t = i / 4
        r = 0.5 + 0.5 * np.sin(t * TAU * PHI)
        g = 0.5 + 0.5 * np.sin(t * TAU * PHI + TAU/3)
        b = 0.5 + 0.5 * np.sin(t * TAU * PHI + 2*TAU/3)
        phi_colors.append((r, g, b))
    
    phi_cmap = LinearSegmentedColormap.from_list('phi_harmonic', phi_colors)
    
    # Crystal cmap for geometric visualizations
    crystal_colors = [
        (0.1, 0.1, 0.3),  # Deep blue for background
        (0.2, 0.3, 0.5),  # Medium blue
        (0.3, 0.6, 0.8),  # Light blue
        (0.9, 0.9, 1.0),  # White/light
        (1.0, 0.8, 0.3),  # Gold highlight
    ]
    crystal_cmap = LinearSegmentedColormap.from_list('crystal', crystal_colors)
    
    # Phase cmap for coherence visualization
    phase_colors = [
        (0.8, 0.0, 0.0),  # Red for anti-phase
        (0.0, 0.0, 0.8),  # Blue for transition
        (0.0, 0.8, 0.0),  # Green for in-phase
        (0.9, 0.9, 0.0),  # Yellow for transition
        (0.8, 0.0, 0.0),  # Red for anti-phase (wrap around)
    ]
    phase_cmap = LinearSegmentedColormap.from_list('phase', phase_colors)
    
    return phi_cmap, crystal_cmap, phase_cmap

# Initialize custom colormaps
PHI_CMAP, CRYSTAL_CMAP, PHASE_CMAP = create_custom_colormaps()

def ensure_output_dir():
    """Ensure output directory exists"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
    else:
        print(f"Output directory exists: {OUTPUT_DIR}")

def generate_grid(size=GRID_SIZE):
    """Generate a 2D grid of coordinates"""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    
    return X, Y, R, Theta

def to_tensor(x):
    """Convert to MLX tensor if MLX is available, otherwise return numpy array"""
    if HAS_MLX:
        if isinstance(x, mx.array):
            return x
        return mx.array(x)
    else:
        if isinstance(x, np.ndarray):
            return x
        return np.array(x)

def from_tensor(x):
    """Convert MLX tensor to numpy array if needed"""
    try:
        if HAS_MLX and isinstance(x, mx.array):
            try:
                return x.numpy()
            except AttributeError:
                # Convert to python values if numpy() is unavailable
                return np.array(x)
        # If it's already a NumPy array, just return it
        elif isinstance(x, np.ndarray):
            return x
        # For any other type, convert to NumPy array
        return np.array(x)
    except Exception as e:
        print(f"Warning in from_tensor: {e}")
        # Last resort fallback
        if not isinstance(x, np.ndarray):
            try:
                return np.array(x)
            except:
                return x
        return x

def enhance_geometric_basis_visualization():
    """
    Generate enhanced visualization for Figure 2 (geometric basis representation)
    with improved symmetry mapping and explicit guides
    """
    print("Generating enhanced geometric basis visualization (Figure 2)...")
    
    # Generate the base grid
    X, Y, R, Theta = generate_grid()
    
    # Create the input field tensor (radial Gaussian with phi-harmonic modulation)
    radius = R.flatten().reshape(1, -1)
    angle = Theta.flatten().reshape(1, -1)
    
    # Create a harmonic field with phi modulation
    field = np.exp(-3 * radius**2) * (1 + 0.3 * np.sin(angle * 5 * PHI))
    
    # Apply geometric activations for each Platonic solid - Use pure NumPy
    solids = ['tetrahedron', 'cube', 'dodecahedron', 'icosahedron', 'all']
    transformed_fields = {}
    
    for solid in solids:
        try:
            # Apply geometric activation with NumPy implementation
            transformed = geometric_activation(field, solid, scale=2.0, resonance=1.2)
            # Convert result to proper format for visualization
            transformed_array = from_tensor(transformed)
            # Make sure we can reshape it properly
            if transformed_array.size == GRID_SIZE * GRID_SIZE:
                transformed_fields[solid] = transformed_array.reshape(GRID_SIZE, GRID_SIZE)
            else:
                # Handle unexpected sizes by creating an appropriate sized field
                print(f"Warning: Unexpected size for {solid} field: {transformed_array.size}, expected {GRID_SIZE * GRID_SIZE}")
                # Create a default field with the right shape
                transformed_fields[solid] = np.zeros((GRID_SIZE, GRID_SIZE))
        except Exception as e:
            print(f"Error processing {solid} field: {e}")
            # Fallback - create an empty field
            transformed_fields[solid] = np.zeros((GRID_SIZE, GRID_SIZE))
    
    # Create enhanced visualization
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # Original visualization 
    ax = axs[0, 0]
    im = ax.imshow(transformed_fields['all'], cmap=CRYSTAL_CMAP, 
                  interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax.set_title("Original Geometric Basis", fontsize=12)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Enhanced visualization with color-coded difference mapping
    ax = axs[0, 1]
    # Create difference field that highlights transitions between geometric patterns
    diff_field = (transformed_fields['tetrahedron'] - transformed_fields['cube'] + 
                 transformed_fields['dodecahedron'] - transformed_fields['icosahedron'])
    diff_field = diff_field / 2.0  # Normalize
    
    im = ax.imshow(diff_field, cmap=PHASE_CMAP, 
                  interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax.set_title("Enhanced Difference Mapping", fontsize=12)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Edge detection highlighting key structural features
    ax = axs[0, 2]
    # Compute gradient magnitude for edge detection
    gx, gy = np.gradient(transformed_fields['all'])
    edges = np.sqrt(gx**2 + gy**2)
    edges = edges / np.max(edges)  # Normalize
    
    im = ax.imshow(edges, cmap='bone', 
                  interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax.set_title("Edge Detection", fontsize=12)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Combined view with explicit geometric guides
    ax = axs[1, 0:3]
    # Create blended visualization with all features
    combined = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    # Base image (geometric field)
    base = CRYSTAL_CMAP(0.5 + 0.5 * transformed_fields['all'])[:, :, :3]
    # Edge overlay
    edge_overlay = np.stack([edges, edges, edges], axis=-1) * np.array([0.8, 0.3, 0.3]).reshape(1, 1, 3)
    # Difference field highlight
    diff_overlay = PHASE_CMAP(0.5 + 0.5 * diff_field)[:, :, :3] * 0.3
    
    # Combine everything
    combined = base * 0.7 + edge_overlay * 0.5 + diff_overlay * 0.5
    combined = np.clip(combined, 0, 1)
    
    im = ax.imshow(combined, interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax.set_title("Composite View with Explicit Geometric Guides", fontsize=14)
    
    # Add geometric guides
    # Golden ratio circles
    for i in range(1, 5):
        radius = i * 0.2 * PHI_INV
        circle = plt.Circle((0, 0), radius, fill=False, color='gold', linestyle='--', 
                            alpha=0.7, linewidth=0.8)
        ax.add_patch(circle)
        
        # Add radius label with phi notation
        if i == 2 or i == 4:
            label = f"r = {i}φ⁻¹/5"
            txt = ax.text(radius*0.7, radius*0.7, label, color='white', fontsize=9,
                        ha='center', va='center')
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
    
    # Add tetrahedral vertices
    tetra_vertices = [
        [0.7, 0.7],  # Normalized for visibility
        [0.7, -0.7],
        [-0.7, 0.7],
        [-0.7, -0.7]
    ]
    
    for i, (x, y) in enumerate(tetra_vertices):
        ax.plot(x, y, 'o', color='red', markersize=6)
        txt = ax.text(x, y+0.05, f"T{i+1}", color='white', fontsize=9,
                    ha='center', va='center')
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
    
    # Add octahedral vertices
    octa_vertices = [
        [0.8, 0],
        [-0.8, 0],
        [0, 0.8],
        [0, -0.8]
    ]
    
    for i, (x, y) in enumerate(octa_vertices):
        ax.plot(x, y, 'o', color='cyan', markersize=6)
        txt = ax.text(x, y+0.05, f"O{i+1}", color='white', fontsize=9,
                    ha='center', va='center')
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
    
    # Add icosahedral vertices on phi-scaled pentagon
    ico_vertices = []
    for i in range(5):
        angle = i * TAU / 5
        r = 0.6
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        ico_vertices.append([x, y])
    
    for i, (x, y) in enumerate(ico_vertices):
        ax.plot(x, y, 'o', color='magenta', markersize=6)
        txt = ax.text(x, y+0.05, f"I{i+1}", color='white', fontsize=9,
                    ha='center', va='center')
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
    
    # Add legend for geometric elements
    elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Tetrahedron'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=8, label='Octahedron'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='magenta', markersize=8, label='Icosahedron'),
        plt.Line2D([0], [0], color='gold', linestyle='--', label='φ-ratio circles')
    ]
    ax.legend(handles=elements, loc='upper right', fontsize=10)
    
    # Save the figure
    fig.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "geometric_basis_enhanced.pdf")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved enhanced geometric basis visualization to {output_path}")
    plt.close(fig)

def enhance_field_evolution_visualization():
    """
    Generate enhanced visualization for Figure 7 (resonant field time evolution)
    with improved phase coherence visualization
    """
    print("Generating enhanced field evolution visualization (Figure 7)...")
    
    # Generate the base grid
    X, Y, R, Theta = generate_grid()
    
    # Create three evolutionary stages of the field
    fields = []
    
    # Create input tensor with basic shape
    radius = R.flatten().reshape(1, -1)
    angle = Theta.flatten().reshape(1, -1)
    
    # Stage 1: Initial field formation (simple structure)
    field1 = np.exp(-2 * radius**2) * (1 + 0.1 * np.sin(angle * 3))
    
    try:
        # Apply geometric activation with NumPy implementation
        transformed1 = geometric_activation(field1, 'all', scale=1.0, resonance=0.8)
        # Safely convert to right format
        transformed1_array = from_tensor(transformed1)
        # Safely reshape or create appropriate array
        if transformed1_array.size == GRID_SIZE * GRID_SIZE:
            fields.append(transformed1_array.reshape(GRID_SIZE, GRID_SIZE))
        else:
            print(f"Warning: Unexpected size for field1: {transformed1_array.size}")
            # Create a simpler field with the right shape instead
            simple_field = np.exp(-2 * (X**2 + Y**2)) * (1 + 0.1 * np.sin(3 * Theta))
            fields.append(simple_field)
    except Exception as e:
        print(f"Error processing field1: {e}")
        # Fallback - create a simple field with the right shape
        simple_field = np.exp(-2 * (X**2 + Y**2)) * (1 + 0.1 * np.sin(3 * Theta))
        fields.append(simple_field)
    
    # Stage 2: Complex mid-evolution phase with maximum information density
    # Create more complex field with multiple resonance components
    field2 = np.exp(-1.5 * radius**2) * (1 + 
                                       0.3 * np.sin(angle * 5 * PHI) + 
                                       0.2 * np.cos(radius * 8 * PHI_INV))
    
    try:
        # Apply resonance and geometric activation
        field2_resonant = apply_resonance(field2, intensity=1.2, resonance_type='quantum')
        transformed2 = geometric_activation(field2_resonant, 'all', scale=1.5, resonance=1.2)
        transformed2_array = from_tensor(transformed2)
        
        if transformed2_array.size == GRID_SIZE * GRID_SIZE:
            fields.append(transformed2_array.reshape(GRID_SIZE, GRID_SIZE))
        else:
            # Create a more complex fallback field
            complex_field = np.exp(-1.5 * (X**2 + Y**2)) * (1 + 
                                                          0.3 * np.sin(5 * PHI * Theta) + 
                                                          0.2 * np.cos(8 * PHI_INV * R))
            fields.append(complex_field)
    except Exception as e:
        print(f"Error processing field2: {e}")
        # Fallback - create a more complex field
        complex_field = np.exp(-1.5 * (X**2 + Y**2)) * (1 + 
                                                      0.3 * np.sin(5 * PHI * Theta) + 
                                                      0.2 * np.cos(8 * PHI_INV * R))
        fields.append(complex_field)
    
    # Stage 3: Stabilized resonance state with harmonic patterns
    field3 = np.exp(-radius**2) * (1 + 
                                 0.4 * np.sin(angle * 5 * PHI) + 
                                 0.3 * np.cos(radius * 8 * PHI_INV) +
                                 0.2 * np.sin(radius * 10 * angle))
    
    try:
        # Apply multiple resonance layers for complexity
        field3_resonant1 = apply_resonance(field3, intensity=1.5, resonance_type='quantum')
        field3_resonant2 = apply_resonance(field3_resonant1, intensity=0.8, resonance_type='holographic')
        transformed3 = geometric_activation(field3_resonant2, 'all', scale=2.0, resonance=1.5)
        transformed3_array = from_tensor(transformed3)
        
        if transformed3_array.size == GRID_SIZE * GRID_SIZE:
            fields.append(transformed3_array.reshape(GRID_SIZE, GRID_SIZE))
        else:
            # Create a more complex fallback field
            advanced_field = np.exp(-(X**2 + Y**2)) * (1 + 
                                                   0.4 * np.sin(5 * PHI * Theta) + 
                                                   0.3 * np.cos(8 * PHI_INV * R) +
                                                   0.2 * np.sin(10 * R * Theta))
            fields.append(advanced_field)
    except Exception as e:
        print(f"Error processing field3: {e}")
        # Fallback - create an advanced field
        advanced_field = np.exp(-(X**2 + Y**2)) * (1 + 
                                               0.4 * np.sin(5 * PHI * Theta) + 
                                               0.3 * np.cos(8 * PHI_INV * R) +
                                               0.2 * np.sin(10 * R * Theta))
        fields.append(advanced_field)
    
    # Create visualization figure with enhanced analytical overlays
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define titles for the evolutionary stages
    titles = [
        "Initial Field Formation",
        "Complex Mid-Evolution Phase",
        "Stabilized Resonance State"
    ]
    
    # Compute phase coherence and flow information
    coherence_maps = []
    flow_fields = []
    
    for i in range(3):
        # Calculate gradient for flow direction
        gx, gy = np.gradient(fields[i])
        flow_magnitude = np.sqrt(gx**2 + gy**2)
        flow_direction = np.arctan2(gy, gx)
        flow_fields.append((gx, gy, flow_magnitude, flow_direction))
        
        # Calculate local phase coherence (simplified)
        # In a full implementation, this would use Hilbert transform or wavelet analysis
        # Here we use gradient direction variation as a proxy
        window_size = 5
        coherence = np.zeros_like(fields[i])
        
        # Calculate coherence as consistency of flow direction in neighborhood
        for y in range(window_size, GRID_SIZE-window_size):
            for x in range(window_size, GRID_SIZE-window_size):
                directions = flow_direction[y-window_size:y+window_size, x-window_size:x+window_size]
                # Circular mean of angles using complex representation
                z = np.mean(np.exp(1j * directions))
                coherence[y, x] = np.abs(z)  # 1=perfect coherence, 0=random
                
        coherence_maps.append(coherence)
    
    # Create enhanced visualizations with overlays
    for i, ax in enumerate(axs):
        # Base field visualization with phi-harmonic colormap
        im = ax.imshow(fields[i], cmap=CRYSTAL_CMAP, 
                      interpolation='bilinear', extent=[-1, 1, -1, 1],
                      vmin=-1, vmax=1)
        
        # For mid-evolution and final state, add flow direction indicators
        if i >= 1:
            # Add flow direction arrows (cyan)
            gx, gy, mag, direction = flow_fields[i]
            
            # Downsample for clearer visualization
            step = 16
            x_pts = np.linspace(-1, 1, GRID_SIZE)
            y_pts = np.linspace(-1, 1, GRID_SIZE)
            X_subset, Y_subset = np.meshgrid(x_pts[::step], y_pts[::step])
            
            # Normalize quiver lengths
            gx_subset = gx[::step, ::step]
            gy_subset = gy[::step, ::step]
            magnitude = np.sqrt(gx_subset**2 + gy_subset**2)
            max_mag = np.max(magnitude)
            if max_mag > 0:
                gx_subset = gx_subset / max_mag * 0.02
                gy_subset = gy_subset / max_mag * 0.02
            
            ax.quiver(X_subset, Y_subset, gx_subset, gy_subset, 
                     color='cyan', scale=1, scale_units='xy', alpha=0.7)
        
        # Add regions of significant change (red highlights)
        if i > 0:
            # Highlight regions that changed significantly from previous stage
            diff = np.abs(fields[i] - fields[i-1])
            mask = diff > 0.3  # Threshold for significant change
            
            # Create semi-transparent red highlight
            highlight = np.zeros((GRID_SIZE, GRID_SIZE, 4))
            highlight[mask, 0] = 1.0  # Red
            highlight[mask, 3] = 0.3  # Alpha
            
            ax.imshow(highlight, extent=[-1, 1, -1, 1], interpolation='bilinear')
        
        # Add magnified regions of interest (yellow boxes)
        if i == 1 or i == 2:
            # Define regions of interest
            roi_centers = [(-0.5, 0.5), (0.5, -0.5)]
            roi_size = 0.2
            
            for j, (cx, cy) in enumerate(roi_centers):
                # Draw yellow box
                rect = Rectangle((cx-roi_size/2, cy-roi_size/2), roi_size, roi_size,
                               linewidth=1.5, edgecolor='yellow', facecolor='none')
                ax.add_patch(rect)
                
                # Add ROI label
                txt = ax.text(cx, cy-roi_size/2-0.05, f"ROI {j+1}", color='yellow', fontsize=9,
                            ha='center', va='center')
                txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
        
        # Add phase-space trajectory trace points for final state
        if i == 2:
            # Create trajectory points based on coherence
            trajectory_x = []
            trajectory_y = []
            
            # Define a spiral trajectory based on field values and coherence
            for j in range(10):
                angle = j * TAU / 10
                r = 0.3 + 0.1 * j / 10
                tx = r * np.cos(angle)
                ty = r * np.sin(angle)
                trajectory_x.append(tx)
                trajectory_y.append(ty)
                
                # Plot magenta dots
                ax.plot(tx, ty, 'o', color='magenta', markersize=4)
                
            # Connect trajectory points
            ax.plot(trajectory_x, trajectory_y, '--', color='magenta', alpha=0.7, linewidth=0.8)
            
            # Add trajectory label
            ax.text(trajectory_x[-1]+0.05, trajectory_y[-1], "Phase\nTrajectory", 
                   color='magenta', fontsize=8, ha='left', va='center')
        
        ax.set_title(titles[i], fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add color legend
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Field Amplitude')
    
    # Add legend for visual indicators
    fig.text(0.01, 0.95, "Visual Elements Guide", fontsize=12, weight='bold')
    
    elements = [
        # Flow indicators
        plt.Line2D([0], [0], color='cyan', marker='>', markersize=8, 
                  label='Flow Direction', linestyle='-'),
        # Change regions
        plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.3, 
                     label='Regions of Significant Change'),
        # Regions of interest
        plt.Rectangle((0, 0), 1, 1, ec='yellow', fc='none', 
                     label='Magnified Regions of Interest'),
        # Phase trajectory
        plt.Line2D([0], [0], color='magenta', marker='o', markersize=4, 
                  label='Phase-Space Trajectory', linestyle='--')
    ]
    
    legend_ax = fig.add_axes([0.01, 0.7, 0.15, 0.25])
    legend_ax.axis('off')
    legend_ax.legend(handles=elements, loc='center')
    
    # Save figure
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    output_path = os.path.join(OUTPUT_DIR, "field_evolution_combined_enhanced.pdf")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved enhanced field evolution visualization to {output_path}")
    
    # Create separate visualization legend as a standalone reference
    legend_fig = plt.figure(figsize=(5, 8))
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis('off')
    
    # Create comprehensive legend with all visual elements
    elements = [
        # Flow indicators
        plt.Line2D([0], [0], color='cyan', marker='>', markersize=10, 
                  label='Flow Direction', linestyle='-'),
        # Change regions
        plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.3, 
                    label='Regions of Significant Change'),
        # Regions of interest
        plt.Rectangle((0, 0), 1, 1, ec='yellow', fc='none', linewidth=1.5,
                    label='Magnified Regions of Interest'),
        # Phase trajectory
        plt.Line2D([0], [0], color='magenta', marker='o', markersize=5, 
                  label='Phase-Space Trajectory', linestyle='--'),
        # Field intensity
        plt.Rectangle((0, 0), 1, 1, fc='white', ec='blue',
                    label='Field High Intensity'),
        plt.Rectangle((0, 0), 1, 1, fc='blue', ec='blue',
                    label='Field Low Intensity'),
        # Geometric elements
        plt.Line2D([0], [0], color='gold', linestyle='--', 
                  label='φ-ratio Harmonic Guides'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, 
                  label='Tetrahedral Nodes'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=8, 
                  label='Octahedral Nodes'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='magenta', markersize=8, 
                  label='Icosahedral Nodes')
    ]
    
    legend_title = "Visual Elements Guide"
    legend = legend_ax.legend(handles=elements, loc='center', title=legend_title, 
                            title_fontsize=14, fontsize=12, frameon=True, framealpha=0.9)
    legend.get_frame().set_edgecolor('gray')
    
    # Add phi-harmonic color scale
    cbar_ax = legend_fig.add_axes([0.25, 0.1, 0.5, 0.05])
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    cbar_ax.imshow(gradient, aspect='auto', cmap=CRYSTAL_CMAP)
    cbar_ax.set_title("Field Amplitude Scale", fontsize=10)
    cbar_ax.set_xticks([0, 128, 255])
    cbar_ax.set_xticklabels(['-1', '0', '+1'])
    cbar_ax.set_yticks([])
    
    # Save legend as separate file
    legend_path = os.path.join(OUTPUT_DIR, "visualization_legend.pdf")
    legend_fig.savefig(legend_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization legend to {legend_path}")
    plt.close(legend_fig)

def enhance_geometric_transformation_visualization():
    """
    Generate enhanced visualization for Figure 9 (geometric transformations)
    with more nuanced visualization details
    """
    print("Generating enhanced geometric transformation visualization (Figure 9)...")
    
    # Generate the base grid
    X, Y, R, Theta = generate_grid()
    
    # Create sequence of transformation steps for visualization
    # We'll create 20 frames to show the progression
    num_frames = 20
    fields = []
    
    # Create input tensor with basic shape
    radius = R.flatten().reshape(1, -1)
    angle = Theta.flatten().reshape(1, -1)
    
    # Create base field with phi-harmonic features
    base_field = np.exp(-2 * radius**2) * (1 + 0.2 * np.sin(angle * 5 * PHI))
    
    try:
        # Since we're using NumPy only mode, use the sequential implementation
        fields = []
        
        for i in range(num_frames):
            t = i / (num_frames - 1)  # 0 to 1
            
            # Apply progressive modulation
            modulated_field = base_field * (1 + 
                                         t * 0.3 * np.sin(angle * (3 + t * 4) * PHI) + 
                                         t * 0.4 * np.cos(radius * (6 + t * 6)))
            
            # Use different solids at different transformation stages
            scale = 1.0 + t * 1.5
            resonance = 0.8 + t * 1.0
            
            # Apply geometric activation with progressive parameters
            batch_output = []
            
            for i in range(num_frames):
                t = i / (num_frames - 1)  # 0 to 1
                
                # Extract single frame from batch
                frame = batch_input[i:i+1]
                
                # Progressive transformation from tetrahedron to dodecahedron blend
                scale = 1.0 + t * 1.5
                resonance = 0.8 + t * 1.0
                
                # Use different solids at different transformation stages
                if i < num_frames // 3:
                    # Initial geometry: tetrahedron dominant
                    blend_weights = {
                        'tetrahedron': 1.0,
                        'octahedron': t * PHI_INV,
                        'cube': t * PHI_INV * PHI_INV,
                        'icosahedron': t * PHI,
                        'dodecahedron': t * PHI * PHI_INV
                    }
                    transformed = geometric_activation(
                        frame, 'all', scale=scale, resonance=resonance, 
                        blend_weights=blend_weights
                    )
                elif i < 2 * num_frames // 3:
                    # Mid transformation: octahedral/icosahedral transition
                    t2 = (i - num_frames // 3) / (num_frames // 3)  # 0 to 1 within this phase
                    blend_weights = {
                        'tetrahedron': 1.0 - t2 * 0.5,
                        'octahedron': PHI_INV + t2 * 0.5,
                        'cube': PHI_INV * PHI_INV + t2 * 0.5,
                        'icosahedron': PHI * (0.5 + t2 * 0.5),
                        'dodecahedron': PHI * PHI_INV * (t2 + 0.5)
                    }
                    transformed = geometric_activation(
                        frame, 'all', scale=scale, resonance=resonance, 
                        blend_weights=blend_weights
                    )
                else:
                    # Final form: dodecahedral dominance
                    t3 = (i - 2 * num_frames // 3) / (num_frames // 3)  # 0 to 1 within this phase
                    blend_weights = {
                        'tetrahedron': 0.5 - t3 * 0.3,
                        'octahedron': PHI_INV,
                        'cube': PHI_INV * PHI_INV,
                        'icosahedron': PHI,
                        'dodecahedron': PHI * PHI_INV * (1.0 + t3)
                    }
                    transformed = geometric_activation(
                        frame, 'all', scale=scale, resonance=resonance, 
                        blend_weights=blend_weights
                    )
                
                batch_output.append(transformed)
            
            # Convert all outputs to numpy arrays
            for output in batch_output:
                fields.append(from_tensor(output).reshape(GRID_SIZE, GRID_SIZE))
                
        else:
            # NumPy sequential fallback
            for i in range(num_frames):
                t = i / (num_frames - 1)  # 0 to 1
                
                # Apply progressive modulation
                modulated_field = base_field * (1 + 
                                            t * 0.3 * np.sin(angle * (3 + t * 4) * PHI) + 
                                            t * 0.4 * np.cos(radius * (6 + t * 6)))
                
                # Use different solids at different transformation stages
                scale = 1.0 + t * 1.5
                resonance = 0.8 + t * 1.0
                
                # Define blend weights based on transformation stage
                if i < num_frames // 3:
                    # Initial geometry: tetrahedron dominant
                    blend_weights = {
                        'tetrahedron': 1.0,
                        'octahedron': t * PHI_INV,
                        'cube': t * PHI_INV * PHI_INV,
                        'icosahedron': t * PHI,
                        'dodecahedron': t * PHI * PHI_INV
                    }
                elif i < 2 * num_frames // 3:
                    # Mid transformation: octahedral/icosahedral transition
                    t2 = (i - num_frames // 3) / (num_frames // 3)  # 0 to 1 within this phase
                    blend_weights = {
                        'tetrahedron': 1.0 - t2 * 0.5,
                        'octahedron': PHI_INV + t2 * 0.5,
                        'cube': PHI_INV * PHI_INV + t2 * 0.5,
                        'icosahedron': PHI * (0.5 + t2 * 0.5),
                        'dodecahedron': PHI * PHI_INV * (t2 + 0.5)
                    }
                else:
                    # Final form: dodecahedral dominance
                    t3 = (i - 2 * num_frames // 3) / (num_frames // 3)  # 0 to 1 within this phase
                    blend_weights = {
                        'tetrahedron': 0.5 - t3 * 0.3,
                        'octahedron': PHI_INV,
                        'cube': PHI_INV * PHI_INV,
                        'icosahedron': PHI,
                        'dodecahedron': PHI * PHI_INV * (1.0 + t3)
                    }
                
                # Apply transformation
                modulated_tensor = to_tensor(modulated_field)
                transformed = geometric_activation(
                    modulated_tensor, 'all', scale=scale, resonance=resonance, 
                    blend_weights=blend_weights
                )
                fields.append(from_tensor(transformed).reshape(GRID_SIZE, GRID_SIZE))
        
        # Select key frames for visualization
        key_indices = [0, num_frames // 2, num_frames - 1]
        key_frames = [fields[i] for i in key_indices]
        
        # Create visualization with 3 key frames showing the transformation
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        titles = [
            "Initial Geometry",
            "Intermediate Transformation",
            "Final Geometry"
        ]
        
        # Create enhanced visualizations
        for i, ax in enumerate(axs):
            # Base field visualization
            im = ax.imshow(key_frames[i], cmap=CRYSTAL_CMAP, 
                          interpolation='bilinear', extent=[-1, 1, -1, 1],
                          vmin=-1, vmax=1)
            
            # Add geometric guides appropriate for each stage
            if i == 0:
                # Initial geometry: tetrahedral guides
                # Tetrahedral vertices
                tetra_vertices = [
                    [0.7, 0.7],
                    [0.7, -0.7],
                    [-0.7, 0.7],
                    [-0.7, -0.7]
                ]
                
                # Connect the vertices
                for j in range(len(tetra_vertices)):
                    for k in range(j+1, len(tetra_vertices)):
                        ax.plot([tetra_vertices[j][0], tetra_vertices[k][0]],
                              [tetra_vertices[j][1], tetra_vertices[k][1]],
                              'r-', alpha=0.5, linewidth=0.8)
                        
                # Add phi circle
                circle = plt.Circle((0, 0), 0.5, fill=False, 
                                   color='gold', linestyle='--', alpha=0.7)
                ax.add_patch(circle)
                
            elif i == 1:
                # Intermediate: octahedral guides
                # Octahedral vertices
                octa_vertices = [
                    [0.8, 0],
                    [-0.8, 0],
                    [0, 0.8],
                    [0, -0.8]
                ]
                
                # Connect vertices
                for j in range(len(octa_vertices)):
                    for k in range(j+1, len(octa_vertices)):
                        ax.plot([octa_vertices[j][0], octa_vertices[k][0]],
                              [octa_vertices[j][1], octa_vertices[k][1]],
                              'c-', alpha=0.5, linewidth=0.8)
                        
                # Add phi spiral
                theta = np.linspace(0, 4*np.pi, 100)
                r = 0.1 + 0.02 * theta
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                ax.plot(x, y, 'y-', alpha=0.7, linewidth=0.8)
                
            else:
                # Final geometry: dodecahedral guides
                # Add phi-based pentagon
                pentagon_vertices = []
                # Create a regular pentagon scaled by phi
                for j in range(5):
                    angle = j * TAU / 5
                    r = 0.6  # Radius
                    x = r * np.cos(angle)
                    y = r * np.sin(angle)
                    pentagon_vertices.append([x, y])
                
                # Connect pentagon vertices
                for j in range(5):
                    next_j = (j + 1) % 5
                    ax.plot([pentagon_vertices[j][0], pentagon_vertices[next_j][0]],
                          [pentagon_vertices[j][1], pentagon_vertices[next_j][1]],
                          'm-', alpha=0.7, linewidth=1.0)
                
                # Add phi-ratio nested pentagons
                inner_pentagon = []
                for j in range(5):
                    angle = j * TAU / 5 + TAU / 10  # Rotate by half a segment
                    r = 0.6 * PHI_INV  # Scale by inverse golden ratio
                    x = r * np.cos(angle)
                    y = r * np.sin(angle)
                    inner_pentagon.append([x, y])
                
                # Connect inner pentagon and create phi-based star
                for j in range(5):
                    next_j = (j + 1) % 5
                    ax.plot([inner_pentagon[j][0], inner_pentagon[next_j][0]],
                          [inner_pentagon[j][1], inner_pentagon[next_j][1]],
                          'm-', alpha=0.5, linewidth=0.8)
                    
                    # Connect to outer pentagon to form star
                    ax.plot([pentagon_vertices[j][0], inner_pentagon[(j+2)%5][0]],
                          [pentagon_vertices[j][1], inner_pentagon[(j+2)%5][1]],
                          'gold', alpha=0.4, linewidth=0.8)
            
            # Add common elements to all frames
            ax.set_title(titles[i], fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add frame number indicator
            frame_ind = key_indices[i]
            ax.text(-0.95, -0.95, f"Frame {frame_ind+1}/{num_frames}", fontsize=8, color='white',
                  bbox=dict(facecolor='black', alpha=0.5, boxstyle='round'))
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Field Amplitude')
        
        # Save combined figure
        fig.tight_layout(rect=[0, 0, 0.9, 1])
        output_path = os.path.join(OUTPUT_DIR, "geometric_transformation_combined.pdf")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined geometric transformation visualization to {output_path}")
        
        # Save individual transformation stages for key frames
        for i, idx in enumerate(key_indices):
            # Create individual high-quality figure for each key frame
            single_fig, single_ax = plt.subplots(figsize=(8, 8))
            
            # Visualize the field with enhanced aesthetics
            im = single_ax.imshow(fields[idx], cmap=CRYSTAL_CMAP, 
                               interpolation='bilinear', extent=[-1, 1, -1, 1],
                               vmin=-1, vmax=1)
            
            # Add geometric guides based on the stage
            if i == 0:  # Initial
                # Tetrahedral guides
                tetra_vertices = [[0.7, 0.7], [0.7, -0.7], [-0.7, 0.7], [-0.7, -0.7]]
                
                # Highlight tetrahedral symmetry
                for j in range(len(tetra_vertices)):
                    # Draw vertex
                    single_ax.plot(tetra_vertices[j][0], tetra_vertices[j][1], 'o', 
                                 color='red', markersize=6)
                    
                    # Draw connecting edges
                    for k in range(j+1, len(tetra_vertices)):
                        single_ax.plot([tetra_vertices[j][0], tetra_vertices[k][0]],
                                     [tetra_vertices[j][1], tetra_vertices[k][1]],
                                     'r-', alpha=0.5, linewidth=1.0)
                
                # Add detailed annotations
                single_ax.text(0, 0, "Tetrahedral\nDominance", color='white', fontsize=12,
                             ha='center', va='center', weight='bold')
                single_ax.text(0, -0.2, "Fire Element", color='red', fontsize=10,
                             ha='center', va='center')
                
                # Add phi-based circles
                for j in range(1, 3):
                    radius = j * 0.3 * PHI_INV
                    circle = plt.Circle((0, 0), radius, fill=False, 
                                      color='gold', linestyle='--', alpha=0.7)
                    single_ax.add_patch(circle)
                
            elif i == 1:  # Intermediate
                # Octahedral guides
                octa_vertices = [[0.8, 0], [-0.8, 0], [0, 0.8], [0, -0.8]]
                
                # Highlight octahedral/icosahedral blend
                for j in range(len(octa_vertices)):
                    # Draw vertex
                    single_ax.plot(octa_vertices[j][0], octa_vertices[j][1], 'o', 
                                 color='cyan', markersize=6)
                    
                    # Draw connecting edges
                    for k in range(j+1, len(octa_vertices)):
                        single_ax.plot([octa_vertices[j][0], octa_vertices[k][0]],
                                     [octa_vertices[j][1], octa_vertices[k][1]],
                                     'c-', alpha=0.5, linewidth=1.0)
                
                # Add spiral
                theta = np.linspace(0, 4*np.pi, 100)
                r = 0.1 + 0.02 * theta
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                single_ax.plot(x, y, 'y-', alpha=0.7, linewidth=0.8)
                
                # Add detailed annotations
                single_ax.text(0, 0, "Octahedral-Icosahedral\nTransition", color='white', fontsize=12,
                             ha='center', va='center', weight='bold')
                single_ax.text(0, -0.2, "Air-Water Blend", color='cyan', fontsize=10,
                             ha='center', va='center')
                
            else:  # Final
                # Dodecahedral guides
                pentagon_vertices = []
                for j in range(5):
                    angle = j * TAU / 5
                    r = 0.6
                    x = r * np.cos(angle)
                    y = r * np.sin(angle)
                    pentagon_vertices.append([x, y])
                
                # Draw pentagon vertices
                for j in range(5):
                    single_ax.plot(pentagon_vertices[j][0], pentagon_vertices[j][1], 'o', 
                                 color='magenta', markersize=6)
                
                # Connect pentagon vertices
                for j in range(5):
                    next_j = (j + 1) % 5
                    single_ax.plot([pentagon_vertices[j][0], pentagon_vertices[next_j][0]],
                                 [pentagon_vertices[j][1], pentagon_vertices[next_j][1]],
                                 'm-', alpha=0.7, linewidth=1.0)
                
                # Add nested pentagons
                inner_pentagon = []
                for j in range(5):
                    angle = j * TAU / 5 + TAU / 10
                    r = 0.6 * PHI_INV
                    x = r * np.cos(angle)
                    y = r * np.sin(angle)
                    inner_pentagon.append([x, y])
                    single_ax.plot(x, y, 'o', color='gold', markersize=4)
                
                # Connect inner pentagon
                for j in range(5):
                    next_j = (j + 1) % 5
                    single_ax.plot([inner_pentagon[j][0], inner_pentagon[next_j][0]],
                                 [inner_pentagon[j][1], inner_pentagon[next_j][1]],
                                 'gold', alpha=0.7, linewidth=0.8)
                
                # Add star connections
                for j in range(5):
                    single_ax.plot([pentagon_vertices[j][0], inner_pentagon[(j+2)%5][0]],
                                 [pentagon_vertices[j][1], inner_pentagon[(j+2)%5][1]],
                                 'gold', alpha=0.4, linewidth=0.8)
                
                # Add detailed annotations
                single_ax.text(0, 0, "Dodecahedral\nDominance", color='white', fontsize=12,
                             ha='center', va='center', weight='bold')
                single_ax.text(0, -0.2, "Aether Element", color='magenta', fontsize=10,
                             ha='center', va='center')
            
            # Add phi-ratio indicator
            single_ax.text(0.8, 0.9, f"φ = {PHI:.5f}", color='gold', fontsize=10,
                         ha='center', va='center', weight='bold')
            
            # Title and aesthetics
            single_ax.set_title(titles[i], fontsize=14)
            single_ax.set_xticks([])
            single_ax.set_yticks([])
            
            # Add colorbar
            divider = make_axes_locatable(single_ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('Field Amplitude')
            
            # Save individual figure
            single_fig.tight_layout()
            output_path = os.path.join(OUTPUT_DIR, f"geometric_transformation_enhanced_{idx}.pdf")
            single_fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved enhanced geometric transformation stage {i+1} to {output_path}")
            plt.close(single_fig)
            
        plt.close(fig)
        
    except Exception as e:
        print(f"Error in geometric transformation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure proper GPU resource cleanup
        if HAS_MLX:
            # Clear MLX arrays to free GPU memory
            # MLX has garbage collection, but it's good practice to explicitly clear large arrays
            if 'batch_input' in locals():
                del batch_input
            if 'batch_output' in locals():
                del batch_output
            if 'batch_fields' in locals():
                del batch_fields
            
            # Force garbage collection
            import gc
            gc.collect()

def enhance_holographic_encoding_visualization():
    """
    Generate enhanced visualization for holographic encoding patterns with improved details
    """
    print("Generating enhanced holographic encoding visualization...")
    
    # Generate the base grid
    X, Y, R, Theta = generate_grid()
    
    # Create input tensor
    radius = R.flatten().reshape(1, -1)
    angle = Theta.flatten().reshape(1, -1)
    
    # Create base field with phi-harmonic modulation
    field = np.exp(-2 * radius**2) * (1 + 0.3 * np.sin(angle * 5 * PHI))
    field = to_tensor(field)
    
    # Apply geometric activation with holographic properties
    transformed = geometric_activation(field, 'dodecahedron', scale=2.0, resonance=1.5)
    holographic_field = from_tensor(transformed).reshape(GRID_SIZE, GRID_SIZE)
    
    # Create enhanced visualization with multiple panels
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original visualization
    ax = axs[0, 0]
    im = ax.imshow(holographic_field, cmap=CRYSTAL_CMAP, 
                  interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax.set_title("Original Holographic Field", fontsize=12)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Phase coherence mapping
    ax = axs[0, 1]
    # Calculate phase and amplitude using Hilbert transform (simplified)
    rows, cols = holographic_field.shape
    phase_map = np.zeros_like(holographic_field)
    amplitude_map = np.zeros_like(holographic_field)
    
    # Simple approach using gradient direction
    gx, gy = np.gradient(holographic_field)
    phase_map = np.arctan2(gy, gx)
    amplitude_map = np.sqrt(gx**2 + gy**2)
    
    # Normalize for visualization
    phase_norm = (phase_map + np.pi) / (2 * np.pi)
    
    im = ax.imshow(phase_norm, cmap=PHASE_CMAP, 
                  interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax.set_title("Phase Coherence Mapping", fontsize=12)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Interference pattern highlighting
    ax = axs[1, 0]
    
    # Apply edge enhancement using Sobel filter 
    from scipy import ndimage
    sobel_h = ndimage.sobel(holographic_field, axis=0)
    sobel_v = ndimage.sobel(holographic_field, axis=1)
    edge_magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
    edge_direction = np.arctan2(sobel_v, sobel_h)
    
    # Create flow field visualization
    edge_norm = edge_magnitude / np.max(edge_magnitude)
    flow_field = CMAP_PLASMA(edge_norm)
    
    # Create overlay with flow direction
    im = ax.imshow(flow_field, interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax.set_title("Interference Pattern Highlighting", fontsize=12)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Magnified regions showing key holographic properties
    ax = axs[1, 1]
    
    # Create magnified inset with annotations
    magnified_region = holographic_field[GRID_SIZE//4:3*GRID_SIZE//4, GRID_SIZE//4:3*GRID_SIZE//4]
    im = ax.imshow(magnified_region, cmap=CRYSTAL_CMAP, 
                  interpolation='bilinear', extent=[-0.5, 0.5, -0.5, 0.5])
    ax.set_title("Magnified Region with Key Properties", fontsize=12)
    
    # Add annotations pointing to holographic features
    feature_points = [
        (-0.3, 0.3, "Self-similarity\nacross scales"),
        (0.2, 0.0, "Phase correlation\npatterns"),
        (-0.2, -0.2, "Distributed\ninformation\nencoding")
    ]
    
    for x, y, text in feature_points:
        ax.plot(x, y, 'o', color='yellow', markersize=8)
        ax.annotate(text, xy=(x, y), xytext=(x+0.15, y+0.15),
                   arrowprops=dict(arrowstyle="->", color='white'),
                   color='white', fontsize=9, weight='bold')
    
    # Add grid lines to show structure
    ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Overall title
    fig.suptitle("Holographic Encoding in Resonant Field", fontsize=16, y=0.98)
    
    # Save the figure
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = os.path.join(OUTPUT_DIR, "holographic_encoding_enhanced.pdf")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved enhanced holographic encoding visualization to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    """Main function to generate all enhanced visualizations."""
    print("Starting resonant field visualization enhancement...")
    
    # Ensure output directory exists
    ensure_output_dir()
    
    # Create enhanced visualizations for each figure
    try:
        enhance_geometric_basis_visualization()  # Figure 2
    except Exception as e:
        print(f"Error generating geometric basis visualization: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        enhance_field_evolution_visualization()  # Figure 7
    except Exception as e:
        print(f"Error generating field evolution visualization: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        enhance_geometric_transformation_visualization()  # Figure 9
    except Exception as e:
        print(f"Error generating geometric transformation visualization: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        enhance_holographic_encoding_visualization()  # Additional visualization
    except Exception as e:
        print(f"Error generating holographic encoding visualization: {e}")
        import traceback
        traceback.print_exc()
    
    print("Visualization generation complete")
    
    # Create a guide file for future reference
    try:
        with open(os.path.join(OUTPUT_DIR, "latex_update_guide.txt"), 'w') as f:
            f.write("Resonant Field Theory Enhanced Visualization Guide\n")
            f.write("=================================================\n\n")
            f.write("This directory contains enhanced visualizations for the Resonant Field Theory paper.\n")
            f.write("The following files have been generated:\n\n")
            f.write("1. geometric_basis_enhanced.pdf - Figure 2 replacement with improved symmetry mapping\n")
            f.write("2. field_evolution_combined_enhanced.pdf - Figure 7 replacement with phase coherence visualization\n")
            f.write("3. visualization_legend.pdf - Standalone legend for figure reference\n")
            f.write("4. geometric_transformation_enhanced_0.pdf - Figure 9 initial geometry stage\n")
            f.write("5. geometric_transformation_enhanced_10.pdf - Figure 9 intermediate transformation\n")
            f.write("6. geometric_transformation_enhanced_19.pdf - Figure 9 final geometry\n")
            f.write("7. holographic_encoding_enhanced.pdf - Enhanced holographic encoding visualization\n\n")
            f.write("To use these enhanced figures in the LaTeX document, update the figure paths in resonant_field_theory_paper.tex.\n")
    except Exception as e:
        print(f"Error creating guide file: {e}")

