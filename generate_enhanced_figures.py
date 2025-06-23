#!/usr/bin/env python3
"""
Enhanced Figure Generator for Resonant Field Theory Paper

This script creates high-quality, nuanced visualizations specifically for 
figures 2, 7, and 9 of the Resonant Field Theory paper. It uses pure NumPy
for computational stability and adds detailed geometric elements based on
phi-harmonic resonance for better visual representation.

Author: Crystalline Consciousness Research Group
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon, Circle, Rectangle
import matplotlib.patheffects as PathEffects
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage
import time
from datetime import datetime

# ==== Constants and Settings ====
# Mathematical constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio (φ ≈ 1.618033988749895)
PHI_INV = 1 / PHI          # Inverse golden ratio (φ⁻¹ ≈ 0.618033988749895)
TAU = 2 * np.pi            # Full circle in radians (τ = 2π)

# Visualization settings
GRID_SIZE = 512  # Higher resolution for improved details
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                         'figures_enhanced_20250503_161506')

# ==== Color Maps ====
def create_custom_colormaps():
    """Create custom phi-harmonic colormaps for resonant field visualization"""
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

# ==== Helper Functions ====
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

def geometric_activation(field, solid_type='all', scale=1.0, resonance=1.0):
    """
    Apply geometric activation using pure NumPy.
    This function provides a stable implementation of the geometric transforms.
    
    Args:
        field: 2D NumPy array - the field to transform
        solid_type: string - which Platonic solid to use for transformation
        scale: float - intensity scaling
        resonance: float - resonance factor
        
    Returns:
        2D NumPy array - transformed field
    """
    # Reshape field for 2D grid if needed
    field_2d = field
    if len(field.shape) == 1:
        field_2d = field.reshape(GRID_SIZE, GRID_SIZE)
    elif field.shape != (GRID_SIZE, GRID_SIZE):
        # If field needs reshaping but can't be done directly, create default field
        if field.size != GRID_SIZE * GRID_SIZE:
            print(f"Warning: Cannot reshape field of size {field.size} to {(GRID_SIZE, GRID_SIZE)}")
            field_2d = np.zeros((GRID_SIZE, GRID_SIZE))
        else:
            field_2d = field.reshape(GRID_SIZE, GRID_SIZE)
    
    # Generate coordinates
    X, Y, R, Theta = generate_grid()
    
    # Apply transformation based on solid type
    if solid_type == 'tetrahedron':
        # Tetrahedron: Fire element - sharp, directed energy
        result = np.tanh(field_2d * scale * resonance) * np.cos(Theta * 4)
        
    elif solid_type == 'octahedron':
        # Octahedron: Air element - mobility and phase fluidity
        phase = field_2d * scale * resonance * TAU * PHI_INV
        result = np.sin(phase) * np.cos(phase * PHI)
        
    elif solid_type == 'cube':
        # Cube: Earth element - stability and structure
        result = 1.0 / (1.0 + np.exp(-field_2d * scale * resonance))
        result = result * np.cos(R * 5 * PHI_INV) * 0.2 + result * 0.8
        
    elif solid_type == 'icosahedron':
        # Icosahedron: Water element - flow and adaptive coherence
        phase1 = field_2d * scale * resonance
        phase2 = field_2d * scale * resonance * PHI
        phase3 = 5 * Theta
        
        result = (np.sin(phase1) + np.sin(phase2 + phase3) * PHI_INV) / (1 + PHI_INV)
        
    elif solid_type == 'dodecahedron':
        # Dodecahedron: Aether/spirit element - harmonic synthesis
        phase = field_2d * scale * resonance
        h1 = np.sin(phase)
        h2 = np.sin(phase * PHI) * PHI_INV 
        h3 = np.sin(phase * PHI * PHI) * PHI_INV * PHI_INV
        
        result = (h1 + h2 + h3) / (1 + PHI_INV + PHI_INV * PHI_INV)
        
    else:  # 'all' - blend different geometries
        # Create base field for blending
        t = np.exp(-3 * R**2) * (1 + 0.3 * np.sin(5 * PHI * Theta))
        
        # Tetrahedron component (red)
        tetra = np.tanh(field_2d * scale * resonance) * np.cos(Theta * 4)
        
        # Octahedron component (air/cyan)
        phase_octa = field_2d * scale * resonance * TAU * PHI_INV
        octa = np.sin(phase_octa) * np.cos(phase_octa * PHI)
        
        # Cube component (earth/structure)
        cube = 1.0 / (1.0 + np.exp(-field_2d * scale * resonance))
        cube = cube * np.cos(R * 5 * PHI_INV) * 0.2 + cube * 0.8
        
        # Icosahedron component (water/fluidity)
        phase1_ico = field_2d * scale * resonance
        phase2_ico = field_2d * scale * resonance * PHI
        phase3_ico = 5 * Theta
        ico = (np.sin(phase1_ico) + np.sin(phase2_ico + phase3_ico) * PHI_INV) / (1 + PHI_INV)
        
        # Dodecahedron component (aether/harmonic)
        phase_dod = field_2d * scale * resonance
        h1 = np.sin(phase_dod)
        h2 = np.sin(phase_dod * PHI) * PHI_INV 
        h3 = np.sin(phase_dod * PHI * PHI) * PHI_INV * PHI_INV
        dod = (h1 + h2 + h3) / (1 + PHI_INV + PHI_INV * PHI_INV)
        
        # Blend components with phi-weighted harmonics
        result = tetra + octa * PHI_INV + cube * PHI_INV * PHI_INV + ico * PHI + dod
        result = result / (1 + PHI_INV + PHI_INV * PHI_INV + PHI + 1)
    
    # Normalize to -1 to 1 range for consistent visualization
    result_min, result_max = np.min(result), np.max(result)
    if result_min != result_max:  # Avoid division by zero
        result = 2 * (result - result_min) / (result_max - result_min) - 1
        
    return result

# The following function generates the file holographic_encoding_enhanced.pdf.
# Modifications to colormap, alpha, and blending will be performed here to fix the red tint and doubling issue.
def figure11_holographic_encoding():
    """
    Generate enhanced holographic encoding visualization (Figure 11)
    This function generates the holographic_encoding_enhanced.pdf file.
    Modifications to colormap, alpha, and blending will be performed here to fix the red tint and doubling issue.
    """
    print("Generating enhanced holographic encoding visualization (Figure 11)...")

    # Generate the base grid
    X, Y, R, Theta = generate_grid()

    # Create a base field for holographic encoding
    base_field = np.exp(-2 * R**2) * (1 + 0.5 * np.sin(5 * PHI * Theta))

    try:
        # Apply holographic resonance
        holographic_field = apply_resonance(base_field, intensity=1.0, resonance_type='holographic')

        # Create visualization
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(holographic_field, cmap='gray',  # Set colormap to gray to address red tint
                      interpolation='bilinear', extent=[-1, 1, -1, 1],
                      vmin=-1, vmax=1)

        ax.set_title("Enhanced Holographic Encoding", fontsize=14)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        # Save the figure
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, "holographic_encoding_enhanced.pdf")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved enhanced holographic encoding visualization to {output_path}")
        plt.close(fig)

    except Exception as e:
        print(f"Error generating holographic encoding figure: {e}")
        import traceback
        traceback.print_exc()

# ==== Figure Generation Functions ====
def generate_figure2_geometric_basis():
    """
    Generate enhanced visualization for Figure 2 (geometric basis representation)
    with improved symmetry mapping and explicit guides
    """
    print("Generating enhanced geometric basis visualization (Figure 2)...")
    
    # Generate the base grid
    X, Y, R, Theta = generate_grid()
    
    # Create a harmonic field with phi modulation
    field = np.exp(-3 * R**2) * (1 + 0.3 * np.sin(Theta * 5 * PHI))
    
    # Apply geometric activations for each Platonic solid
    solids = ['tetrahedron', 'cube', 'dodecahedron', 'icosahedron', 'all']
    transformed_fields = {}
    
    for solid in solids:
        try:
            # Apply geometric activation
            transformed = geometric_activation(field, solid, scale=2.0, resonance=1.2)
            transformed_fields[solid] = transformed
        except Exception as e:
            print(f"Error processing {solid} field: {e}")
            # Fallback - create an empty field with the right shape
            transformed_fields[solid] = np.zeros((GRID_SIZE, GRID_SIZE))
    
    # Create enhanced visualization with proper gridspec
    from matplotlib import gridspec
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.4)
    
    # Create individual axes
    ax1 = fig.add_subplot(gs[0, 0])  # Original visualization
    ax2 = fig.add_subplot(gs[0, 1])  # Enhanced difference mapping
    ax3 = fig.add_subplot(gs[0, 2])  # Edge detection
    ax_combined = fig.add_subplot(gs[1, :])  # Spanning all columns in second row
    
    # Original visualization 
    im = ax1.imshow(transformed_fields['all'], cmap=CRYSTAL_CMAP, 
                  interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax1.set_title("Original Geometric Basis", fontsize=12)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Enhanced visualization with color-coded difference mapping
    # Create difference field that highlights transitions between geometric patterns
    diff_field = (transformed_fields['tetrahedron'] - transformed_fields['cube'] + 
                 transformed_fields['dodecahedron'] - transformed_fields['icosahedron'])
    diff_field = diff_field / 2.0  # Normalize
    
    im = ax2.imshow(diff_field, cmap=PHASE_CMAP, 
                  interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax2.set_title("Enhanced Difference Mapping", fontsize=12)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Edge detection highlighting key structural features
    # Compute gradient magnitude for edge detection
    gx, gy = np.gradient(transformed_fields['all'])
    edges = np.sqrt(gx**2 + gy**2)
    edges_max = np.max(edges)
    if edges_max > 0:
        edges = edges / edges_max  # Normalize
    
    im = ax3.imshow(edges, cmap='bone', 
                  interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax3.set_title("Edge Detection", fontsize=12)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Combined view with explicit geometric guides is created using gs[1, :]
    # and stored in ax_combined variable
    
    # Create blended visualization with all features
    combined = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    # Base image (geometric field)
    base = CRYSTAL_CMAP(0.5 + 0.5 * transformed_fields['all'])[:, :, :3]
    # Edge overlay
    edge_overlay = np.stack([edges, edges, edges], axis=-1) * np.array([0.8, 0.3, 0.3]).reshape(1, 1, 3)
    # Difference field highlight
    diff_normalized = (diff_field - np.min(diff_field)) / (np.max(diff_field) - np.min(diff_field))
    diff_overlay = PHASE_CMAP(diff_normalized)[:, :, :3] * 0.3
    
    # Combine everything
    combined = base * 0.7 + edge_overlay * 0.5 + diff_overlay * 0.5
    combined = np.clip(combined, 0, 1)
    
    im = ax_combined.imshow(combined, interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax_combined.set_title("Composite View with Explicit Geometric Guides", fontsize=14)
    
    # Add geometric guides
    # Golden ratio circles
    for i in range(1, 5):
        radius = i * 0.2 * PHI_INV
        circle = plt.Circle((0, 0), radius, fill=False, color='gold', linestyle='--', 
                            alpha=0.7, linewidth=0.8)
        ax_combined.add_patch(circle)
        
        # Add radius label with phi notation
        if i == 2 or i == 4:
            label = f"r = {i}φ⁻¹/5"
            txt = ax_combined.text(radius*0.7, radius*0.7, label, color='white', fontsize=9,
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
        ax_combined.plot(x, y, 'o', color='red', markersize=6)
        txt = ax_combined.text(x, y+0.05, f"T{i+1}", color='white', fontsize=9,
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
        ax_combined.plot(x, y, 'o', color='cyan', markersize=6)
        txt = ax_combined.text(x, y+0.05, f"O{i+1}", color='white', fontsize=9,
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
        ax_combined.plot(x, y, 'o', color='magenta', markersize=6)
        txt = ax_combined.text(x, y+0.05, f"I{i+1}", color='white', fontsize=9,
                    ha='center', va='center')
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
    
    # Add legend for geometric elements
    elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Tetrahedron'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=8, label='Octahedron'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='magenta', markersize=8, label='Icosahedron'),
        plt.Line2D([0], [0], color='gold', linestyle='--', label='φ-ratio circles')
    ]
    ax_combined.legend(handles=elements, loc='upper right', fontsize=10)
    
    # Save the figure
    fig.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "geometric_basis_enhanced.pdf")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved enhanced geometric basis visualization to {output_path}")
    plt.close(fig)

def figure7_field_evolution():
    """
    Generate enhanced visualization for Figure 7 (resonant field time evolution)
    with improved phase coherence visualization
    """
    print("Generating enhanced field evolution visualization (Figure 7)...")
    
    # Generate the base grid
    X, Y, R, Theta = generate_grid()
    
    # Create three evolutionary stages of the field
    fields = []
    
    # Stage 1: Initial field formation (simple structure)
    field1 = np.exp(-2 * R**2) * (1 + 0.1 * np.sin(Theta * 3))
    
    try:
        # Apply geometric activation with NumPy implementation
        transformed1 = geometric_activation(field1, 'all', scale=1.0, resonance=0.8)
        fields.append(transformed1)
    except Exception as e:
        print(f"Error processing field1: {e}")
        # Fallback - create a simple field with the right shape
        simple_field = np.exp(-2 * (X**2 + Y**2)) * (1 + 0.1 * np.sin(3 * Theta))
        fields.append(simple_field)
    
    # Stage 2: Complex mid-evolution phase with maximum information density
    # Create more complex field with multiple resonance components
    field2 = np.exp(-1.5 * R**2) * (1 + 
                                  0.3 * np.sin(Theta * 5 * PHI) + 
                                  0.2 * np.cos(R * 8 * PHI_INV))
    
    try:
        # Apply resonance and geometric activation
        field2_resonant = apply_resonance(field2, intensity=1.2, resonance_type='quantum')
        transformed2 = geometric_activation(field2_resonant, 'all', scale=1.5, resonance=1.2)
        fields.append(transformed2)
    except Exception as e:
        print(f"Error processing field2: {e}")
        # Fallback - create a more complex field
        complex_field = np.exp(-1.5 * (X**2 + Y**2)) * (1 + 
                                                     0.3 * np.sin(5 * PHI * Theta) + 
                                                     0.2 * np.cos(8 * PHI_INV * R))
        fields.append(complex_field)
    
    # Stage 3: Stabilized resonance state with harmonic patterns
    field3 = np.exp(-R**2) * (1 + 
                             0.4 * np.sin(Theta * 5 * PHI) + 
                             0.3 * np.cos(R * 8 * PHI_INV) +
                             0.2 * np.sin(R * 10 * Theta))
    
    try:
        # Apply multiple resonance layers for complexity
        field3_resonant1 = apply_resonance(field3, intensity=1.5, resonance_type='quantum')
        field3_resonant2 = apply_resonance(field3_resonant1, intensity=0.8, resonance_type='holographic')
        transformed3 = geometric_activation(field3_resonant2, 'all', scale=2.0, resonance=1.5)
        fields.append(transformed3)
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
        for y in range(window_size, GRID_SIZE-window_size, window_size):
            for x in range(window_size, GRID_SIZE-window_size, window_size):
                directions = flow_direction[y-window_size:y+window_size, x-window_size:x+window_size]
                # Circular mean of angles using complex representation
                z = np.mean(np.exp(1j * directions))
                coherence[y-window_size:y+window_size, x-window_size:x+window_size] = np.abs(z)  # 1=perfect coherence, 0=random
                
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
                gx_subset = gx_subset / max_mag * 0.05
                gy_subset = gy_subset / max_mag * 0.05

            # Add flow arrows
            ax.quiver(X_subset, Y_subset, gx_subset, gy_subset,
                      color='cyan', alpha=0.7, scale=1.0, scale_units='inches')
            
        # Add phase coherence contour lines for all visualizations
        coherence = coherence_maps[i]
        # Smooth coherence map for better contours
        coherence_smooth = ndimage.gaussian_filter(coherence, sigma=2.0)
        
        # Plot phase coherence contours
        levels = np.linspace(0.2, 0.9, 5)  # Phase coherence levels
        contour = ax.contour(np.linspace(-1, 1, GRID_SIZE), np.linspace(-1, 1, GRID_SIZE),
                           coherence_smooth, levels=levels, 
                           colors='white', alpha=0.5, linewidths=0.5)
        
        # Add high-coherence region annotations for the final state
        if i == 2:
            # Find regions of high coherence
            high_coherence = coherence_smooth > 0.8
            labeled_regions, num_regions = ndimage.label(high_coherence)
            
            # Add annotation for up to 3 high-coherence regions
            for region_idx in range(1, min(num_regions + 1, 4)):
                # Find centroid for each region
                region_coords = np.array(ndimage.center_of_mass(high_coherence, labeled_regions, region_idx)).astype(int)
                center_x, center_y = region_coords[1], region_coords[0]
                
                # Convert to display coordinates
                display_x = np.linspace(-1, 1, GRID_SIZE)[center_x]
                display_y = np.linspace(-1, 1, GRID_SIZE)[center_y]
                
                # Add annotation
                ax.annotate(f"High Coherence Region", xy=(display_x, display_y), xytext=(display_x + 0.2, display_y + 0.2),
                            arrowprops=dict(facecolor='black', shrink=0.05),
                            fontsize=8, ha='center', color='white',
                            path_effects=[PathEffects.withStroke(linewidth=1.5, foreground="black")])
        
        ax.set_title(titles[i], fontsize=12)
        ax.axis('off')  # Hide axes
    
    # Save the figure
    fig.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "field_evolution_combined_enhanced.pdf")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved enhanced field evolution visualization to {output_path}")
    plt.close(fig)

def figure9_geometric_transformation():
    """
    Enhanced visualization for Figure 9: Geometric transformations
    demonstrating transitions between Platonic forms through harmonic resonance
    """
    print("Generating enhanced geometric transformation visualizations (Figure 9)...")
    
    # Define parameter ranges
    num_frames = 20  # Number of transition frames
    morph_range = np.linspace(0, 1, num_frames)
    
    # Generate base grid
    X, Y, R, Theta = generate_grid()
    
    # Create base field with initial phi modulation
    field_base = np.exp(-1.5 * R**2) * (1 + 0.3 * np.sin(Theta * 5 * PHI))
    
    # Create different geometric fields
    solids = ['tetrahedron', 'cube', 'dodecahedron', 'icosahedron']
    transformed_fields = {solid: geometric_activation(field_base, solid, scale=2.0, resonance=1.2) for solid in solids}
    
    # Perform interpolation between two distinct geometric forms
    def geometric_morph(morph_value):
        """Morph between two geometric forms"""
        # Define start/end geometries
        geom1 = transformed_fields['tetrahedron']  
        geom2 = transformed_fields['icosahedron']   
        
        # Ensure shapes are compatible
        if geom1.shape != geom2.shape:
            print("Incompatible array dimensions")
            return np.zeros_like(geom1)
        

# ==== Constants and Settings ====
# Mathematical constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio (φ ≈ 1.618033988749895)
PHI_INV = 1 / PHI          # Inverse golden ratio (φ⁻¹ ≈ 0.618033988749895)
TAU = 2 * np.pi            # Full circle in radians (τ = 2π)

# Visualization settings
GRID_SIZE = 512  # Higher resolution for improved details
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                         'figures_enhanced_20250503_161506')

# ==== Color Maps ====
def create_custom_colormaps():
    """Create custom phi-harmonic colormaps for resonant field visualization"""
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

# ==== Helper Functions ====
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

def geometric_activation(field, solid_type='all', scale=1.0, resonance=1.0):
    """
    Apply geometric activation using pure NumPy.
    This function provides a stable implementation of the geometric transforms.
    
    Args:
        field: 2D NumPy array - the field to transform
        solid_type: string - which Platonic solid to use for transformation
        scale: float - intensity scaling
        resonance: float - resonance factor
        
    Returns:
        2D NumPy array - transformed field
    """
    # Reshape field for 2D grid if needed
    field_2d = field
    if len(field.shape) == 1:
        field_2d = field.reshape(GRID_SIZE, GRID_SIZE)
    elif field.shape != (GRID_SIZE, GRID_SIZE):
        # If field needs reshaping but can't be done directly, create default field
        if field.size != GRID_SIZE * GRID_SIZE:
            print(f"Warning: Cannot reshape field of size {field.size} to {(GRID_SIZE, GRID_SIZE)}")
            field_2d = np.zeros((GRID_SIZE, GRID_SIZE))
        else:
            field_2d = field.reshape(GRID_SIZE, GRID_SIZE)
    
    # Generate coordinates
    X, Y, R, Theta = generate_grid()
    
    # Apply transformation based on solid type
    if solid_type == 'tetrahedron':
        # Tetrahedron: Fire element - sharp, directed energy
        result = np.tanh(field_2d * scale * resonance) * np.cos(Theta * 4)
        
    elif solid_type == 'octahedron':
        # Octahedron: Air element - mobility and phase fluidity
        phase = field_2d * scale * resonance * TAU * PHI_INV
        result = np.sin(phase) * np.cos(phase * PHI)
        
    elif solid_type == 'cube':
        # Cube: Earth element - stability and structure
        result = 1.0 / (1.0 + np.exp(-field_2d * scale * resonance))
        result = result * np.cos(R * 5 * PHI_INV) * 0.2 + result * 0.8
        
    elif solid_type == 'icosahedron':
        # Icosahedron: Water element - flow and adaptive coherence
        phase1 = field_2d * scale * resonance
        phase2 = field_2d * scale * resonance * PHI
        phase3 = 5 * Theta
        
        result = (np.sin(phase1) + np.sin(phase2 + phase3) * PHI_INV) / (1 + PHI_INV)
        
    elif solid_type == 'dodecahedron':
        # Dodecahedron: Aether/spirit element - harmonic synthesis
        phase = field_2d * scale * resonance
        h1 = np.sin(phase)
        h2 = np.sin(phase * PHI) * PHI_INV 
        h3 = np.sin(phase * PHI * PHI) * PHI_INV * PHI_INV
        
        result = (h1 + h2 + h3) / (1 + PHI_INV + PHI_INV * PHI_INV)
        
    else:  # 'all' - blend different geometries
        # Create base field for blending
        t = np.exp(-3 * R**2) * (1 + 0.3 * np.sin(5 * PHI * Theta))
        
        # Tetrahedron component (red)
        tetra = np.tanh(field_2d * scale * resonance) * np.cos(Theta * 4)
        
        # Octahedron component (air/cyan)
        phase_octa = field_2d * scale * resonance * TAU * PHI_INV
        octa = np.sin(phase_octa) * np.cos(phase_octa * PHI)
        
        # Cube component (earth/structure)
        cube = 1.0 / (1.0 + np.exp(-field_2d * scale * resonance))
        cube = cube * np.cos(R * 5 * PHI_INV) * 0.2 + cube * 0.8
        
        # Icosahedron component (water/fluidity)
        phase1_ico = field_2d * scale * resonance
        phase2_ico = field_2d * scale * resonance * PHI
        phase3_ico = 5 * Theta
        ico = (np.sin(phase1_ico) + np.sin(phase2_ico + phase3_ico) * PHI_INV) / (1 + PHI_INV)
        
        # Dodecahedron component (aether/harmonic)
        phase_dod = field_2d * scale * resonance
        h1 = np.sin(phase_dod)
        h2 = np.sin(phase_dod * PHI) * PHI_INV 
        h3 = np.sin(phase_dod * PHI * PHI) * PHI_INV * PHI_INV
        dod = (h1 + h2 + h3) / (1 + PHI_INV + PHI_INV * PHI_INV)
        
        # Blend components with phi-weighted harmonics
        result = tetra + octa * PHI_INV + cube * PHI_INV * PHI_INV + ico * PHI + dod
        result = result / (1 + PHI_INV + PHI_INV * PHI_INV + PHI + 1)
    
    # Normalize to -1 to 1 range for consistent visualization
    result_min, result_max = np.min(result), np.max(result)
    if result_min != result_max:  # Avoid division by zero
        result = 2 * (result - result_min) / (result_max - result_min) - 1
        
    return result

# The following function generates the file holographic_encoding_enhanced.pdf.
# Modifications to colormap, alpha, and blending will be performed here to fix the red tint and doubling issue.
def figure11_holographic_encoding():
    """
    Generate enhanced holographic encoding visualization (Figure 11)
    This function generates the holographic_encoding_enhanced.pdf file.
    Modifications to colormap, alpha, and blending will be performed here to fix the red tint and doubling issue.
    """
    print("Generating enhanced holographic encoding visualization (Figure 11)...")

    # Generate the base grid
    X, Y, R, Theta = generate_grid()

    # Create a base field for holographic encoding
    base_field = np.exp(-2 * R**2) * (1 + 0.5 * np.sin(5 * PHI * Theta))

    try:
        # Apply holographic resonance
        holographic_field = apply_resonance(base_field, intensity=1.0, resonance_type='holographic')

        # Create visualization
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(holographic_field, 
                      interpolation='bilinear', extent=[-1, 1, -1, 1],
                      vmin=-1, vmax=1)

        ax.set_title("Enhanced Holographic Encoding", fontsize=14)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        # Save the figure
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, "holographic_encoding_enhanced.pdf")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved enhanced holographic encoding visualization to {output_path}")
        plt.close(fig)

    except Exception as e:
        print(f"Error generating holographic encoding figure: {e}")
        import traceback
        traceback.print_exc()

# ==== Figure Generation Functions ====
def generate_figure2_geometric_basis():
    """
    Generate enhanced visualization for Figure 2 (geometric basis representation)
    with improved symmetry mapping and explicit guides
    """
    print("Generating enhanced geometric basis visualization (Figure 2)...")
    
    # Generate the base grid
    X, Y, R, Theta = generate_grid()
    
    # Create a harmonic field with phi modulation
    field = np.exp(-3 * R**2) * (1 + 0.3 * np.sin(Theta * 5 * PHI))
    
    # Apply geometric activations for each Platonic solid
    solids = ['tetrahedron', 'cube', 'dodecahedron', 'icosahedron', 'all']
    transformed_fields = {}
    
    for solid in solids:
        try:
            # Apply geometric activation
            transformed = geometric_activation(field, solid, scale=2.0, resonance=1.2)
            transformed_fields[solid] = transformed
        except Exception as e:
            print(f"Error processing {solid} field: {e}")
            # Fallback - create an empty field with the right shape
            transformed_fields[solid] = np.zeros((GRID_SIZE, GRID_SIZE))
    
    # Create enhanced visualization with proper gridspec
    from matplotlib import gridspec
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.4)
    
    # Create individual axes
    ax1 = fig.add_subplot(gs[0, 0])  # Original visualization
    ax2 = fig.add_subplot(gs[0, 1])  # Enhanced difference mapping
    ax3 = fig.add_subplot(gs[0, 2])  # Edge detection
    ax_combined = fig.add_subplot(gs[1, :])  # Spanning all columns in second row
    
    # Original visualization 
    im = ax1.imshow(transformed_fields['all'], cmap=CRYSTAL_CMAP, 
                  interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax1.set_title("Original Geometric Basis", fontsize=12)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Enhanced visualization with color-coded difference mapping
    # Create difference field that highlights transitions between geometric patterns
    diff_field = (transformed_fields['tetrahedron'] - transformed_fields['cube'] + 
                 transformed_fields['dodecahedron'] - transformed_fields['icosahedron'])
    diff_field = diff_field / 2.0  # Normalize
    
    im = ax2.imshow(diff_field, cmap=PHASE_CMAP, 
                  interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax2.set_title("Enhanced Difference Mapping", fontsize=12)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Edge detection highlighting key structural features
    # Compute gradient magnitude for edge detection
    gx, gy = np.gradient(transformed_fields['all'])
    edges = np.sqrt(gx**2 + gy**2)
    edges_max = np.max(edges)
    if edges_max > 0:
        edges = edges / edges_max  # Normalize
    
    im = ax3.imshow(edges, cmap='bone', 
                  interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax3.set_title("Edge Detection", fontsize=12)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Combined view with explicit geometric guides is created using gs[1, :]
    # and stored in ax_combined variable
    
    # Create blended visualization with all features
    combined = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    # Base image (geometric field)
    base = CRYSTAL_CMAP(0.5 + 0.5 * transformed_fields['all'])[:, :, :3]
    # Edge overlay
    edge_overlay = np.stack([edges, edges, edges], axis=-1) * np.array([0.8, 0.3, 0.3]).reshape(1, 1, 3)
    # Difference field highlight
    diff_normalized = (diff_field - np.min(diff_field)) / (np.max(diff_field) - np.min(diff_field))
    diff_overlay = PHASE_CMAP(diff_normalized)[:, :, :3] * 0.3
    
    # Combine everything
    combined = base * 0.7 + edge_overlay * 0.5 + diff_overlay * 0.5
    combined = np.clip(combined, 0, 1)
    
    im = ax_combined.imshow(combined, interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax_combined.set_title("Composite View with Explicit Geometric Guides", fontsize=14)
    
    # Add geometric guides
    # Golden ratio circles
    for i in range(1, 5):
        radius = i * 0.2 * PHI_INV
        circle = plt.Circle((0, 0), radius, fill=False, color='gold', linestyle='--', 
                            alpha=0.7, linewidth=0.8)
        ax_combined.add_patch(circle)
        
        # Add radius label with phi notation
        if i == 2 or i == 4:
            label = f"r = {i}φ⁻¹/5"
            txt = ax_combined.text(radius*0.7, radius*0.7, label, color='white', fontsize=9,
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
        ax_combined.plot(x, y, 'o', color='red', markersize=6)
        txt = ax_combined.text(x, y+0.05, f"T{i+1}", color='white', fontsize=9,
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
        ax_combined.plot(x, y, 'o', color='cyan', markersize=6)
        txt = ax_combined.text(x, y+0.05, f"O{i+1}", color='white', fontsize=9,
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
        ax_combined.plot(x, y, 'o', color='magenta', markersize=6)
        txt = ax_combined.text(x, y+0.05, f"I{i+1}", color='white', fontsize=9,
                    ha='center', va='center')
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
    
    # Add legend for geometric elements
    elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Tetrahedron'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=8, label='Octahedron'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='magenta', markersize=8, label='Icosahedron'),
        plt.Line2D([0], [0], color='gold', linestyle='--', label='φ-ratio circles')
    ]
    ax_combined.legend(handles=elements, loc='upper right', fontsize=10)
    
    # Save the figure
    fig.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "geometric_basis_enhanced.pdf")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved enhanced geometric basis visualization to {output_path}")
    plt.close(fig)

def figure7_field_evolution():
    """
    Generate enhanced visualization for Figure 7 (resonant field time evolution)
    with improved phase coherence visualization
    """
    print("Generating enhanced field evolution visualization (Figure 7)...")
    
    # Generate the base grid
    X, Y, R, Theta = generate_grid()
    
    # Create three evolutionary stages of the field
    fields = []
    
    # Stage 1: Initial field formation (simple structure)
    field1 = np.exp(-2 * R**2) * (1 + 0.1 * np.sin(Theta * 3))
    
    try:
        # Apply geometric activation with NumPy implementation
        transformed1 = geometric_activation(field1, 'all', scale=1.0, resonance=0.8)
        fields.append(transformed1)
    except Exception as e:
        print(f"Error processing field1: {e}")
        # Fallback - create a simple field with the right shape
        simple_field = np.exp(-2 * (X**2 + Y**2)) * (1 + 0.1 * np.sin(3 * Theta))
        fields.append(simple_field)
    
    # Stage 2: Complex mid-evolution phase with maximum information density
    # Create more complex field with multiple resonance components
    field2 = np.exp(-1.5 * R**2) * (1 + 
                                  0.3 * np.sin(Theta * 5 * PHI) + 
                                  0.2 * np.cos(R * 8 * PHI_INV))
    
    try:
        # Apply resonance and geometric activation
        field2_resonant = apply_resonance(field2, intensity=1.2, resonance_type='quantum')
        transformed2 = geometric_activation(field2_resonant, 'all', scale=1.5, resonance=1.2)
        fields.append(transformed2)
    except Exception as e:
        print(f"Error processing field2: {e}")
        # Fallback - create a more complex field
        complex_field = np.exp(-1.5 * (X**2 + Y**2)) * (1 + 
                                                     0.3 * np.sin(5 * PHI * Theta) + 
                                                     0.2 * np.cos(8 * PHI_INV * R))
        fields.append(complex_field)
    
    # Stage 3: Stabilized resonance state with harmonic patterns
    field3 = np.exp(-R**2) * (1 + 
                             0.4 * np.sin(Theta * 5 * PHI) + 
                             0.3 * np.cos(R * 8 * PHI_INV) +
                             0.2 * np.sin(R * 10 * Theta))
    
    try:
        # Apply multiple resonance layers for complexity
        field3_resonant1 = apply_resonance(field3, intensity=1.5, resonance_type='quantum')
        field3_resonant2 = apply_resonance(field3_resonant1, intensity=0.8, resonance_type='holographic')
        transformed3 = geometric_activation(field3_resonant2, 'all', scale=2.0, resonance=1.5)
        fields.append(transformed3)
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
        for y in range(window_size, GRID_SIZE-window_size, window_size):
            for x in range(window_size, GRID_SIZE-window_size, window_size):
                directions = flow_direction[y-window_size:y+window_size, x-window_size:x+window_size]
                # Circular mean of angles using complex representation
                z = np.mean(np.exp(1j * directions))
                coherence[y-window_size:y+window_size, x-window_size:x+window_size] = np.abs(z)  # 1=perfect coherence, 0=random
                
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
                gx_subset = gx_subset / max_mag * 0.05
                gy_subset = gy_subset / max_mag * 0.05
            # Add flow arrows
            ax.quiver(X_subset, Y_subset, gx_subset, gy_subset, 
                      color='cyan', alpha=0.7, scale=1.0, scale_units='inches')
            
        # Add phase coherence contour lines for all visualizations
        coherence = coherence_maps[i]
        # Smooth coherence map for better contours
        coherence_smooth = ndimage.gaussian_filter(coherence, sigma=2.0)
        
        # Plot phase coherence contours
        levels = np.linspace(0.2, 0.9, 5)  # Phase coherence levels
        contour = ax.contour(np.linspace(-1, 1, GRID_SIZE), np.linspace(-1, 1, GRID_SIZE),
                           coherence_smooth, levels=levels, 
                           colors='white', alpha=0.5, linewidths=0.5)
        
        # Add high-coherence region annotations for the final state
        if i == 2:
            # Find regions of high coherence
            high_coherence = coherence_smooth > 0.8
            labeled_regions, num_regions = ndimage.label(high_coherence)
            
            # Add annotation for up to 3 high-coherence regions
            for region_idx in range(1, min(num_regions + 1, 4)):
                # Find centroid for each region
                region_coords = np.array(ndimage.center_of_mass(high_coherence, labeled_regions, region_idx)).astype(int)
                center_x, center_y = region_coords[1], region_coords[0]
                
                # Convert to display coordinates
                display_x = np.linspace(-1, 1, GRID_SIZE)[center_x]
                display_y = np.linspace(-1, 1, GRID_SIZE)[center_y]
                
                # Add annotation
                ax.annotate(f"High Coherence Region", xy=(display_x, display_y), xytext=(display_x + 0.2, display_y + 0.2),
                            arrowprops=dict(facecolor='black', shrink=0.05),
                            fontsize=8, ha='center', color='white',
                            path_effects=[PathEffects.withStroke(linewidth=1.5, foreground="black")])
        
        ax.set_title(titles[i], fontsize=12)
        ax.axis('off')  # Hide axes
    
    # Save the figure
    fig.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "field_evolution_combined_enhanced.pdf")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved enhanced field evolution visualization to {output_path}")
    plt.close(fig)

def figure9_geometric_transformation():
    """
    Enhanced visualization for Figure 9: Geometric transformations
    demonstrating transitions between Platonic forms through harmonic resonance
    """
    print("Generating enhanced geometric transformation visualizations (Figure 9)...")
    
    # Define parameter ranges
    num_frames = 20  # Number of transition frames
    morph_range = np.linspace(0, 1, num_frames)
    
    # Generate base grid
    X, Y, R, Theta = generate_grid()
    
    # Create base field with initial phi modulation
    field_base = np.exp(-1.5 * R**2) * (1 + 0.3 * np.sin(Theta * 5 * PHI))
    
    # Create different geometric fields
    solids = ['tetrahedron', 'cube', 'dodecahedron', 'icosahedron']
    transformed_fields = {solid: geometric_activation(field_base, solid, scale=2.0, resonance=1.2) for solid in solids}
    
    # Perform interpolation between two distinct geometric forms
    def geometric_morph(morph_value):
        """Morph between two geometric forms"""
        # Define start/end geometries
        geom1 = transformed_fields['tetrahedron']  
        geom2 = transformed_fields['icosahedron']   
        
        # Ensure shapes are compatible
        if geom1.shape != geom2.shape:
            print("Incompatible array dimensions")
            return np.zeros_like(geom1)
        
        # Use morph_value to blend
        morphed_field = geom1 * (1 - morph_value) + geom2 * morph_value
        return morphed_field
    
    # Generate each transition frame
    for i, morph_value in enumerate(morph_range):
        # Generate blended field
        blended_field = geometric_morph(morph_value)
        
        # Setup figure
        fig, ax = plt.subplots()
# Mathematical constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio (φ ≈ 1.618033988749895)
PHI_INV = 1 / PHI          # Inverse golden ratio (φ⁻¹ ≈ 0.618033988749895)
TAU = 2 * np.pi            # Full circle in radians (τ = 2π)

# Visualization settings
GRID_SIZE = 512  # Higher resolution for improved details
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'figures_enhanced_20250503_161506')

# ==== Color Maps ====
def create_custom_colormaps():
    """Create custom phi-harmonic colormaps for resonant field visualization"""
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

# ==== Helper Functions ====
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

def geometric_activation(field, solid_type='all', scale=1.0, resonance=1.0):
    """
    Apply geometric activation using pure NumPy.
    This function provides a stable implementation of the geometric transforms.
    
    Args:
        field: 2D NumPy array - the field to transform
        solid_type: string - which Platonic solid to use for transformation
        scale: float - intensity scaling
        resonance: float - resonance factor
        
    Returns:
        2D NumPy array - transformed field
    """
    # Reshape field for 2D grid if needed
    field_2d = field
    if len(field.shape) == 1:
        field_2d = field.reshape(GRID_SIZE, GRID_SIZE)
    elif field.shape != (GRID_SIZE, GRID_SIZE):
        # If field needs reshaping but can't be done directly, create default field
        if field.size != GRID_SIZE * GRID_SIZE:
            print(f"Warning: Cannot reshape field of size {field.size} to {(GRID_SIZE, GRID_SIZE)}")
            field_2d = np.zeros((GRID_SIZE, GRID_SIZE))
        else:
            field_2d = field.reshape(GRID_SIZE, GRID_SIZE)
    
    # Generate coordinates
    X, Y, R, Theta = generate_grid()
    
    # Apply transformation based on solid type
    if solid_type == 'tetrahedron':
        # Tetrahedron: Fire element - sharp, directed energy
        result = np.tanh(field_2d * scale * resonance) * np.cos(Theta * 4)
        
    elif solid_type == 'octahedron':
        # Octahedron: Air element - mobility and phase fluidity
        phase = field_2d * scale * resonance * TAU * PHI_INV
        result = np.sin(phase) * np.cos(phase * PHI)
        
    elif solid_type == 'cube':
        # Cube: Earth element - stability and structure
        result = 1.0 / (1.0 + np.exp(-field_2d * scale * resonance))
        result = result * np.cos(R * 5 * PHI_INV) * 0.2 + result * 0.8
        
    elif solid_type == 'icosahedron':
        # Icosahedron: Water element - flow and adaptive coherence
        phase1 = field_2d * scale * resonance
        phase2 = field_2d * scale * resonance * PHI
        phase3 = 5 * Theta
        
        result = (np.sin(phase1) + np.sin(phase2 + phase3) * PHI_INV) / (1 + PHI_INV)
        
    elif solid_type == 'dodecahedron':
        # Dodecahedron: Aether/spirit element - harmonic synthesis
        phase = field_2d * scale * resonance
        h1 = np.sin(phase)
        h2 = np.sin(phase * PHI) * PHI_INV 
        h3 = np.sin(phase * PHI * PHI) * PHI_INV * PHI_INV
        
        result = (h1 + h2 + h3) / (1 + PHI_INV + PHI_INV * PHI_INV)
        
    else:  # 'all' - blend different geometries
        # Create base field for blending
        t = np.exp(-3 * R**2) * (1 + 0.3 * np.sin(5 * PHI * Theta))
        
        # Tetrahedron component (red)
        tetra = np.tanh(field_2d * scale * resonance) * np.cos(Theta * 4)
        
        # Octahedron component (air/cyan)
        phase_octa = field_2d * scale * resonance * TAU * PHI_INV
        octa = np.sin(phase_octa) * np.cos(phase_octa * PHI)
        
        # Cube component (earth/structure)
        cube = 1.0 / (1.0 + np.exp(-field_2d * scale * resonance))
        cube = cube * np.cos(R * 5 * PHI_INV) * 0.2 + cube * 0.8
        
        # Icosahedron component (water/fluidity)
        phase1_ico = field_2d * scale * resonance
        phase2_ico = field_2d * scale * resonance * PHI
        phase3_ico = 5 * Theta
        ico = (np.sin(phase1_ico) + np.sin(phase2_ico + phase3_ico) * PHI_INV) / (1 + PHI_INV)
        
        # Dodecahedron component (aether/harmonic)
        phase_dod = field_2d * scale * resonance
        h1 = np.sin(phase_dod)
        h2 = np.sin(phase_dod * PHI) * PHI_INV 
        h3 = np.sin(phase_dod * PHI * PHI) * PHI_INV * PHI_INV
        dod = (h1 + h2 + h3) / (1 + PHI_INV + PHI_INV * PHI_INV)
        
        # Blend components with phi-weighted harmonics
        result = tetra + octa * PHI_INV + cube * PHI_INV * PHI_INV + ico * PHI + dod
        result = result / (1 + PHI_INV + PHI_INV * PHI_INV + PHI + 1)
    
    # Normalize to -1 to 1 range for consistent visualization
    result_min, result_max = np.min(result), np.max(result)
    if result_min != result_max:  # Avoid division by zero
        result = 2 * (result - result_min) / (result_max - result_min) - 1
        
    return result

# The following function generates the file holographic_encoding_enhanced.pdf.
# Modifications to colormap, alpha, and blending will be performed here to fix the red tint and doubling issue.
def figure11_holographic_encoding():
    """
    Generate enhanced holographic encoding visualization (Figure 11)
    This function generates the holographic_encoding_enhanced.pdf file.
    Modifications to colormap, alpha, and blending will be performed here to fix the red tint and doubling issue.
    """
    print("Generating enhanced holographic encoding visualization (Figure 11)...")

    # Generate the base grid
    X, Y, R, Theta = generate_grid()

    # Create a base field for holographic encoding
    base_field = np.exp(-2 * R**2) * (1 + 0.5 * np.sin(5 * PHI * Theta))

    try:
        # Apply holographic resonance
        holographic_field = apply_resonance(base_field, intensity=1.0, resonance_type='holographic')

        # Create visualization
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(holographic_field, cmap='gray',  # Set colormap to gray to address red tint
                      interpolation='bilinear', extent=[-1, 1, -1, 1],
                      vmin=-1, vmax=1)

        ax.set_title("Enhanced Holographic Encoding", fontsize=14)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        # Save the figure
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, "holographic_encoding_enhanced.pdf")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved enhanced holographic encoding visualization to {output_path}")
        plt.close(fig)

    except Exception as e:
        print(f"Error generating holographic encoding figure: {e}")
        import traceback
        traceback.print_exc()

# ==== Figure Generation Functions ====
def generate_figure2_geometric_basis():
    """
    Generate enhanced visualization for Figure 2 (geometric basis representation)
    with improved symmetry mapping and explicit guides
    """
    print("Generating enhanced geometric basis visualization (Figure 2)...")
    
    # Generate the base grid
    X, Y, R, Theta = generate_grid()
    
    # Create a harmonic field with phi modulation
    field = np.exp(-3 * R**2) * (1 + 0.3 * np.sin(Theta * 5 * PHI))
    
    # Apply geometric activations for each Platonic solid
    solids = ['tetrahedron', 'cube', 'dodecahedron', 'icosahedron', 'all']
    transformed_fields = {}
    
    for solid in solids:
        try:
            # Apply geometric activation
            transformed = geometric_activation(field, solid, scale=2.0, resonance=1.2)
            transformed_fields[solid] = transformed
        except Exception as e:
            print(f"Error processing {solid} field: {e}")
            # Fallback - create an empty field with the right shape
            transformed_fields[solid] = np.zeros((GRID_SIZE, GRID_SIZE))
    
    # Create enhanced visualization with proper gridspec
    from matplotlib import gridspec
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.4)
    
    # Create individual axes
    ax1 = fig.add_subplot(gs[0, 0])  # Original visualization
    ax2 = fig.add_subplot(gs[0, 1])  # Enhanced difference mapping
    ax3 = fig.add_subplot(gs[0, 2])  # Edge detection
    ax_combined = fig.add_subplot(gs[1, :])  # Spanning all columns in second row
    
    # Original visualization 
    im = ax1.imshow(transformed_fields['all'], cmap=CRYSTAL_CMAP, 
                  interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax1.set_title("Original Geometric Basis", fontsize=12)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Enhanced visualization with color-coded difference mapping
    # Create difference field that highlights transitions between geometric patterns
    diff_field = (transformed_fields['tetrahedron'] - transformed_fields['cube'] + 
                 transformed_fields['dodecahedron'] - transformed_fields['icosahedron'])
    diff_field = diff_field / 2.0  # Normalize
    
    im = ax2.imshow(diff_field, cmap=PHASE_CMAP, 
                  interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax2.set_title("Enhanced Difference Mapping", fontsize=12)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Edge detection highlighting key structural features
    # Compute gradient magnitude for edge detection
    gx, gy = np.gradient(transformed_fields['all'])
    edges = np.sqrt(gx**2 + gy**2)
    edges_max = np.max(edges)
    if edges_max > 0:
        edges = edges / edges_max  # Normalize
    
    im = ax3.imshow(edges, cmap='bone', 
                  interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax3.set_title("Edge Detection", fontsize=12)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Combined view with explicit geometric guides is created using gs[1, :]
    # and stored in ax_combined variable
    
    # Create blended visualization with all features
    combined = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    # Base image (geometric field)
    base = CRYSTAL_CMAP(0.5 + 0.5 * transformed_fields['all'])[:, :, :3]
    # Edge overlay
    edge_overlay = np.stack([edges, edges, edges], axis=-1) * np.array([0.8, 0.3, 0.3]).reshape(1, 1, 3)
    # Difference field highlight
    diff_normalized = (diff_field - np.min(diff_field)) / (np.max(diff_field) - np.min(diff_field))
    diff_overlay = PHASE_CMAP(diff_normalized)[:, :, :3] * 0.3
    
    # Combine everything
    combined = base * 0.7 + edge_overlay * 0.5 + diff_overlay * 0.5
    combined = np.clip(combined, 0, 1)
    
    im = ax_combined.imshow(combined, interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax_combined.set_title("Composite View with Explicit Geometric Guides", fontsize=14)
    
    # Add geometric guides
    # Golden ratio circles
    for i in range(1, 5):
        radius = i * 0.2 * PHI_INV
        circle = plt.Circle((0, 0), radius, fill=False, color='gold', linestyle='--', 
                            alpha=0.7, linewidth=0.8)
        ax_combined.add_patch(circle)
        
        # Add radius label with phi notation
        if i == 2 or i == 4:
            label = f"r = {i}φ⁻¹/5"
            txt = ax_combined.text(radius*0.7, radius*0.7, label, color='white', fontsize=9,
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
        ax_combined.plot(x, y, 'o', color='red', markersize=6)
        txt = ax_combined.text(x, y+0.05, f"T{i+1}", color='white', fontsize=9,
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
        ax_combined.plot(x, y, 'o', color='cyan', markersize=6)
        txt = ax_combined.text(x, y+0.05, f"O{i+1}", color='white', fontsize=9,
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
        ax_combined.plot(x, y, 'o', color='magenta', markersize=6)
        txt = ax_combined.text(x, y+0.05, f"I{i+1}", color='white', fontsize=9,
                    ha='center', va='center')
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
    
    # Add legend for geometric elements
    elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Tetrahedron'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=8, label='Octahedron'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='magenta', markersize=8, label='Icosahedron'),
        plt.Line2D([0], [0], color='gold', linestyle='--', label='φ-ratio circles')
        ]
    ax_combined.legend(handles=elements, loc='upper right', fontsize=10)
    
    # Save the figure
    fig.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "geometric_basis_enhanced.pdf")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved enhanced geometric basis visualization to {output_path}")
    plt.close(fig)

def figure7_field_evolution():
    """
    Generate enhanced visualization for Figure 7 (resonant field time evolution)
    with improved phase coherence visualization
    """
    print("Generating enhanced field evolution visualization (Figure 7)...")
    
    # Generate the base grid
    X, Y, R, Theta = generate_grid()
    
    # Create three evolutionary stages of the field
    fields = []
    
    # Stage 1: Initial field formation (simple structure)
    field1 = np.exp(-2 * R**2) * (1 + 0.1 * np.sin(Theta * 3))
    
    try:
        # Apply geometric activation with NumPy implementation
        transformed1 = geometric_activation(field1, 'all', scale=1.0, resonance=0.8)
        fields.append(transformed1)
    except Exception as e:
        print(f"Error processing field1: {e}")
        # Fallback - create a simple field with the right shape
        simple_field = np.exp(-2 * (X**2 + Y**2)) * (1 + 0.1 * np.sin(3 * Theta))
        fields.append(simple_field)
    
    # Stage 2: Complex mid-evolution phase with maximum information density
    # Create more complex field with multiple resonance components
    field2 = np.exp(-1.5 * R**2) * (1 + 
                                  0.3 * np.sin(Theta * 5 * PHI) + 
                                  0.2 * np.cos(R * 8 * PHI_INV))
    
    try:
        # Apply resonance and geometric activation
        field2_resonant = apply_resonance(field2, intensity=1.2, resonance_type='quantum')
        transformed2 = geometric_activation(field2_resonant, 'all', scale=1.5, resonance=1.2)
        fields.append(transformed2)
    except Exception as e:
        print(f"Error processing field2: {e}")
        # Fallback - create a more complex field
        complex_field = np.exp(-1.5 * (X**2 + Y**2)) * (1 + 
                                                     0.3 * np.sin(5 * PHI * Theta) + 
                                                     0.2 * np.cos(8 * PHI_INV * R))
        fields.append(complex_field)
    
    # Stage 3: Stabilized resonance state with harmonic patterns
    field3 = np.exp(-R**2) * (1 + 
                             0.4 * np.sin(Theta * 5 * PHI) + 
                             0.3 * np.cos(R * 8 * PHI_INV) +
                             0.2 * np.sin(R * 10 * Theta))
    
    try:
        # Apply multiple resonance layers for complexity
        field3_resonant1 = apply_resonance(field3, intensity=1.5, resonance_type='quantum')
        field3_resonant2 = apply_resonance(field3_resonant1, intensity=0.8, resonance_type='holographic')
        transformed3 = geometric_activation(field3_resonant2, 'all', scale=2.0, resonance=1.5)
        fields.append(transformed3)
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
        for y in range(window_size, GRID_SIZE-window_size, window_size):
            for x in range(window_size, GRID_SIZE-window_size, window_size):
                directions = flow_direction[y-window_size:y+window_size, x-window_size:x+window_size]
                # Circular mean of angles using complex representation
                z = np.mean(np.exp(1j * directions))
                coherence[y-window_size:y+window_size, x-window_size:x+window_size] = np.abs(z)  # 1=perfect coherence, 0=random
                
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
            gx_subset = gx_subset / max_mag * 0.05
            gy_subset = gy_subset / max_mag * 0.05
            
            # Add flow arrows
            ax.quiver(X_subset, Y_subset, gx_subset, gy_subset, 
                      color='cyan', alpha=0.7, scale=1.0, scale_units='inches')
            
        # Add phase coherence contour lines for all visualizations
        coherence = coherence_maps[i]
        # Smooth coherence map for better contours
        coherence_smooth = ndimage.gaussian_filter(coherence, sigma=2.0)
        
        # Plot phase coherence contours
        levels = np.linspace(0.2, 0.9, 5)  # Phase coherence levels
        contour = ax.contour(np.linspace(-1, 1, GRID_SIZE), np.linspace(-1, 1, GRID_SIZE),
                           coherence_smooth, levels=levels, 
                           colors='white', alpha=0.5, linewidths=0.5)
        
        # Add high-coherence region annotations for the final state
        if i == 2:
            # Find regions of high coherence
            high_coherence = coherence_smooth > 0.8
            labeled_regions, num_regions = ndimage.label(high_coherence)
            
            # Add annotation for up to 3 high-coherence regions
            for region_idx in range(1, min(num_regions + 1, 4)):  # cap at 3 annotations
                # Find centroid for each region
                region_coords = np.array(ndimage.center_of_mass(high_coherence, labeled_regions, region_idx)).astype(int)
                center_x, center_y = region_coords[1], region_coords[0]
                
                # Convert to display coordinates
                display_x = np.linspace(-1, 1, GRID_SIZE)[center_x]
                display_y = np.linspace(-1, 1, GRID_SIZE)[center_y]
                
                # Add annotation
                ax.annotate(f"High Coherence Region", xy=(display_x, display_y), xytext=(display_x + 0.2, display_y + 0.2),
                            arrowprops=dict(facecolor='black', shrink=0.05),
                            fontsize=8, ha='center', color='white',
                            path_effects=[PathEffects.withStroke(linewidth=1.5, foreground="black")])
        
        ax.set_title(titles[i], fontsize=12)
        ax.axis('off')  # Hide axes
    
    # Save the figure
    fig.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "field_evolution_combined_enhanced.pdf")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved enhanced field evolution visualization to {output_path}")
    plt.close(fig)

def figure9_geometric_transformation():
    """
    Enhanced visualization for Figure 9: Geometric transformations
    demonstrating transitions between Platonic forms through harmonic resonance
    """
    print("Generating enhanced geometric transformation visualizations (Figure 9)...")
    
    # Define parameter ranges
    num_frames = 20  # Number of transition frames
    morph_range = np.linspace(0, 1, num_frames)
    
    # Generate base grid
    X, Y, R, Theta = generate_grid()
    
    # Create base field with initial phi modulation
    field_base = np.exp(-1.5 * R**2) * (1 + 0.3 * np.sin(Theta * 5 * PHI))
    
    # Create different geometric fields
    solids = ['tetrahedron', 'cube', 'dodecahedron', 'icosahedron']
    transformed_fields = {solid: geometric_activation(field_base, solid, scale=2.0, resonance=1.2) for solid in solids}
    
    # Perform interpolation between two distinct geometric forms
    def geometric_morph(morph_value):
        """Morph between two geometric forms"""
        # Define start/end geometries
        geom1 = transformed_fields['tetrahedron']  
        geom2 = transformed_fields['icosahedron']   
        
        # Ensure shapes are compatible
        if geom1.shape != geom2.shape:
            print("Incompatible array dimensions")
            return np.zeros_like(geom1)
        
        # Use morph_value to blend
        morphed_field = geom1 * (1 - morph_value) + geom2 * morph_value
        return morphed_field

#!/usr/bin/env python3
"""
Enhanced Figure Generator for Resonant Field Theory Paper

This script creates high-quality, nuanced visualizations specifically for 
figures 2, 7, and 9 of the Resonant Field Theory paper. It uses pure NumPy
for computational stability and adds detailed geometric elements based on
phi-harmonic resonance for better visual representation.

Author: Crystalline Consciousness Research Group
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon, Circle, Rectangle
import matplotlib.patheffects as PathEffects
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage
import time
from datetime import datetime

# ==== Constants and Settings ====
# Mathematical constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio (φ ≈ 1.618033988749895)
PHI_INV = 1 / PHI          # Inverse golden ratio (φ⁻¹ ≈ 0.618033988749895)
TAU = 2 * np.pi            # Full circle in radians (τ = 2π)

# Visualization settings
GRID_SIZE = 512  # Higher resolution for improved details
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                         'figures_enhanced_20250503_161506')

# ==== Color Maps ====
def create_custom_colormaps():
    """Create custom phi-harmonic colormaps for resonant field visualization"""
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

# ==== Helper Functions ====
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

def geometric_activation(field, solid_type='all', scale=1.0, resonance=1.0):
    """
    Apply geometric activation using pure NumPy.
    This function provides a stable implementation of the geometric transforms.
    
    Args:
        field: 2D NumPy array - the field to transform
        solid_type: string - which Platonic solid to use for transformation
        scale: float - intensity scaling
        resonance: float - resonance factor
        
    Returns:
        2D NumPy array - transformed field
    """
    # Reshape field for 2D grid if needed
    field_2d = field
    if len(field.shape) == 1:
        field_2d = field.reshape(GRID_SIZE, GRID_SIZE)
    elif field.shape != (GRID_SIZE, GRID_SIZE):
        # If field needs reshaping but can't be done directly, create default field
        if field.size != GRID_SIZE * GRID_SIZE:
            print(f"Warning: Cannot reshape field of size {field.size} to {(GRID_SIZE, GRID_SIZE)}")
            field_2d = np.zeros((GRID_SIZE, GRID_SIZE))
        else:
            field_2d = field.reshape(GRID_SIZE, GRID_SIZE)
    
    # Generate coordinates
    X, Y, R, Theta = generate_grid()
    
    # Apply transformation based on solid type
    if solid_type == 'tetrahedron':
        # Tetrahedron: Fire element - sharp, directed energy
        result = np.tanh(field_2d * scale * resonance) * np.cos(Theta * 4)
        
    elif solid_type == 'octahedron':
        # Octahedron: Air element - mobility and phase fluidity
        phase = field_2d * scale * resonance * TAU * PHI_INV
        result = np.sin(phase) * np.cos(phase * PHI)
        
    elif solid_type == 'cube':
        # Cube: Earth element - stability and structure
        result = 1.0 / (1.0 + np.exp(-field_2d * scale * resonance))
        result = result * np.cos(R * 5 * PHI_INV) * 0.2 + result * 0.8
        
    elif solid_type == 'icosahedron':
        # Icosahedron: Water element - flow and adaptive coherence
        phase1 = field_2d * scale * resonance
        phase2 = field_2d * scale * resonance * PHI
        phase3 = 5 * Theta
        
        result = (np.sin(phase1) + np.sin(phase2 + phase3) * PHI_INV) / (1 + PHI_INV)
        
    elif solid_type == 'dodecahedron':
        # Dodecahedron: Aether/spirit element - harmonic synthesis
        phase = field_2d * scale * resonance
        h1 = np.sin(phase)
        h2 = np.sin(phase * PHI) * PHI_INV 
        h3 = np.sin(phase * PHI * PHI) * PHI_INV * PHI_INV
        
        result = (h1 + h2 + h3) / (1 + PHI_INV + PHI_INV * PHI_INV)
        
    else:  # 'all' - blend different geometries
        # Create base field for blending
        t = np.exp(-3 * R**2) * (1 + 0.3 * np.sin(5 * PHI * Theta))
        
        # Tetrahedron component (red)
        tetra = np.tanh(field_2d * scale * resonance) * np.cos(Theta * 4)
        
        # Octahedron component (air/cyan)
        phase_octa = field_2d * scale * resonance * TAU * PHI_INV
        octa = np.sin(phase_octa) * np.cos(phase_octa * PHI)
        
        # Cube component (earth/structure)
        cube = 1.0 / (1.0 + np.exp(-field_2d * scale * resonance))
        cube = cube * np.cos(R * 5 * PHI_INV) * 0.2 + cube * 0.8
        
        # Icosahedron component (water/fluidity)
        phase1_ico = field_2d * scale * resonance
        phase2_ico = field_2d * scale * resonance * PHI
        phase3_ico = 5 * Theta
        ico = (np.sin(phase1_ico) + np.sin(phase2_ico + phase3_ico) * PHI_INV) / (1 + PHI_INV)
        
        # Dodecahedron component (aether/harmonic)
        phase_dod = field_2d * scale * resonance
        h1 = np.sin(phase_dod)
        h2 = np.sin(phase_dod * PHI) * PHI_INV 
        h3 = np.sin(phase_dod * PHI * PHI) * PHI_INV * PHI_INV
        dod = (h1 + h2 + h3) / (1 + PHI_INV + PHI_INV * PHI_INV)
        
        # Blend components with phi-weighted harmonics
        result = tetra + octa * PHI_INV + cube * PHI_INV * PHI_INV + ico * PHI + dod
        result = result / (1 + PHI_INV + PHI_INV * PHI_INV + PHI + 1)
    
    # Normalize to -1 to 1 range for consistent visualization
    result_min, result_max = np.min(result), np.max(result)
    if result_min != result_max:  # Avoid division by zero
        result = 2 * (result - result_min) / (result_max - result_min) - 1
        
    return result

# The following function generates the file holographic_encoding_enhanced.pdf.
# Modifications to colormap, alpha, and blending will be performed here to fix the red tint and doubling issue.
def figure11_holographic_encoding():
    """
    Generate enhanced holographic encoding visualization (Figure 11)
    This function generates the holographic_encoding_enhanced.pdf file.
    Modifications to colormap, alpha, and blending will be performed here to fix the red tint and doubling issue.
    """
    print("Generating enhanced holographic encoding visualization (Figure 11)...")

    # Generate the base grid
    X, Y, R, Theta = generate_grid()

    # Create a base field for holographic encoding
    base_field = np.exp(-2 * R**2) * (1 + 0.5 * np.sin(5 * PHI * Theta))

    try:
        # Apply holographic resonance
        holographic_field = apply_resonance(base_field, intensity=1.0, resonance_type='holographic')

        # Create visualization
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(holographic_field, cmap='gray',  # Set colormap to gray to address red tint
                      interpolation='bilinear', extent=[-1, 1, -1, 1],
                      vmin=-1, vmax=1)

        ax.set_title("Enhanced Holographic Encoding", fontsize=14)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        # Save the figure
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, "holographic_encoding_enhanced.pdf")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved enhanced holographic encoding visualization to {output_path}")
        plt.close(fig)

    except Exception as e:
        print(f"Error generating holographic encoding figure: {e}")
        import traceback
        traceback.print_exc()

# ==== Figure Generation Functions ====
def generate_figure2_geometric_basis():
    """
    Generate enhanced visualization for Figure 2 (geometric basis representation)
    with improved symmetry mapping and explicit guides
    """
    print("Generating enhanced geometric basis visualization (Figure 2)...")
    
    # Generate the base grid
    X, Y, R, Theta = generate_grid()
    
    # Create a harmonic field with phi modulation
    field = np.exp(-3 * R**2) * (1 + 0.3 * np.sin(Theta * 5 * PHI))
    
    # Apply geometric activations for each Platonic solid
    solids = ['tetrahedron', 'cube', 'dodecahedron', 'icosahedron', 'all']
    transformed_fields = {}
    
    for solid in solids:
        try:
            # Apply geometric activation
            transformed = geometric_activation(field, solid, scale=2.0, resonance=1.2)
            transformed_fields[solid] = transformed
        except Exception as e:
            print(f"Error processing {solid} field: {e}")
            # Fallback - create an empty field with the right shape
            transformed_fields[solid] = np.zeros((GRID_SIZE, GRID_SIZE))
    
    # Create enhanced visualization with proper gridspec
    from matplotlib import gridspec
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.4)
    
    # Create individual axes
    ax1 = fig.add_subplot(gs[0, 0])  # Original visualization
    ax2 = fig.add_subplot(gs[0, 1])  # Enhanced difference mapping
    ax3 = fig.add_subplot(gs[0, 2])  # Edge detection
    ax_combined = fig.add_subplot(gs[1, :])  # Spanning all columns in second row
    
    # Original visualization 
    im = ax1.imshow(transformed_fields['all'], cmap=CRYSTAL_CMAP, 
                  interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax1.set_title("Original Geometric Basis", fontsize=12)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Enhanced visualization with color-coded difference mapping
    # Create difference field that highlights transitions between geometric patterns
    diff_field = (transformed_fields['tetrahedron'] - transformed_fields['cube'] + 
                 transformed_fields['dodecahedron'] - transformed_fields['icosahedron'])
    diff_field = diff_field / 2.0  # Normalize
    
    im = ax2.imshow(diff_field, cmap=PHASE_CMAP, 
                  interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax2.set_title("Enhanced Difference Mapping", fontsize=12)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Edge detection highlighting key structural features
    # Compute gradient magnitude for edge detection
    gx, gy = np.gradient(transformed_fields['all'])
    edges = np.sqrt(gx**2 + gy**2)
    edges_max = np.max(edges)
    if edges_max > 0:
        edges = edges / edges_max  # Normalize
    
    im = ax3.imshow(edges, cmap='bone', 
                  interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax3.set_title("Edge Detection", fontsize=12)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Combined view with explicit geometric guides is created using gs[1, :]
    # and stored in ax_combined variable
    
    # Create blended visualization with all features
    combined = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    # Base image (geometric field)
    base = CRYSTAL_CMAP(0.5 + 0.5 * transformed_fields['all'])[:, :, :3]
    # Edge overlay
    edge_overlay = np.stack([edges, edges, edges], axis=-1) * np.array([0.8, 0.3, 0.3]).reshape(1, 1, 3)
    # Difference field highlight
    diff_normalized = (diff_field - np.min(diff_field)) / (np.max(diff_field) - np.min(diff_field))
    diff_overlay = PHASE_CMAP(diff_normalized)[:, :, :3] * 0.3
    
    # Combine everything
    combined = base * 0.7 + edge_overlay * 0.5 + diff_overlay * 0.5
    combined = np.clip(combined, 0, 1)
    
    im = ax_combined.imshow(combined, interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax_combined.set_title("Composite View with Explicit Geometric Guides", fontsize=14)
    
    # Add geometric guides
    # Golden ratio circles
    for i in range(1, 5):
        radius = i * 0.2 * PHI_INV
        circle = plt.Circle((0, 0), radius, fill=False, color='gold', linestyle='--', 
                            alpha=0.7, linewidth=0.8)
        ax_combined.add_patch(circle)
        
        # Add radius label with phi notation
        if i == 2 or i == 4:
            label = f"r = {i}φ⁻¹/5"
            txt = ax_combined.text(radius*0.7, radius*0.7, label, color='white', fontsize=9,
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
        ax_combined.plot(x, y, 'o', color='red', markersize=6)
        txt = ax_combined.text(x, y+0.05, f"T{i+1}", color='white', fontsize=9,
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
        ax_combined.plot(x, y, 'o', color='cyan', markersize=6)
        txt = ax_combined.text(x, y+0.05, f"O{i+1}", color='white', fontsize=9,
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
        ax_combined.plot(x, y, 'o', color='magenta', markersize=6)
        txt = ax_combined.text(x, y+0.05, f"I{i+1}", color='white', fontsize=9,
                    ha='center', va='center')
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
    
    # Add legend for geometric elements
    elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Tetrahedron'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=8, label='Octahedron'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='magenta', markersize=8, label='Icosahedron'),
        plt.Line2D([0], [0], color='gold', linestyle='--', label='φ-ratio circles')
    ]
    ax_combined.legend(handles=elements, loc='upper right', fontsize=10)
    
    # Save the figure
    fig.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "geometric_basis_enhanced.pdf")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved enhanced geometric basis visualization to {output_path}")
    plt.close(fig)

def figure7_field_evolution():
    """
    Generate enhanced visualization for Figure 7 (resonant field time evolution)
    with improved phase coherence visualization
    """
    print("Generating enhanced field evolution visualization (Figure 7)...")
    
    # Generate the base grid
    X, Y, R, Theta = generate_grid()
    
    # Create three evolutionary stages of the field
    fields = []
    
    # Stage 1: Initial field formation (simple structure)
    field1 = np.exp(-2 * R**2) * (1 + 0.1 * np.sin(Theta * 3))
    
    try:
        # Apply geometric activation with NumPy implementation
        transformed1 = geometric_activation(field1, 'all', scale=1.0, resonance=0.8)
        fields.append(transformed1)
    except Exception as e:
        print(f"Error processing field1: {e}")
        # Fallback - create a simple field with the right shape
        simple_field = np.exp(-2 * (X**2 + Y**2)) * (1 + 0.1 * np.sin(3 * Theta))
        fields.append(simple_field)
    
    # Stage 2: Complex mid-evolution phase with maximum information density
    # Create more complex field with multiple resonance components
    field2 = np.exp(-1.5 * R**2) * (1 + 
                                  0.3 * np.sin(Theta * 5 * PHI) + 
                                  0.2 * np.cos(R * 8 * PHI_INV))
    
    try:
        # Apply resonance and geometric activation
        field2_resonant = apply_resonance(field2, intensity=1.2, resonance_type='quantum')
        transformed2 = geometric_activation(field2_resonant, 'all', scale=1.5, resonance=1.2)
        fields.append(transformed2)
    except Exception as e:
        print(f"Error processing field2: {e}")
        # Fallback - create a more complex field
        complex_field = np.exp(-1.5 * (X**2 + Y**2)) * (1 + 
                                                     0.3 * np.sin(5 * PHI * Theta) + 
                                                     0.2 * np.cos(8 * PHI_INV * R))
        fields.append(complex_field)
    
    # Stage 3: Stabilized resonance state with harmonic patterns
    field3 = np.exp(-R**2) * (1 + 
                             0.4 * np.sin(Theta * 5 * PHI) + 
                             0.3 * np.cos(R * 8 * PHI_INV) +
                             0.2 * np.sin(R * 10 * Theta))
    
    try:
        # Apply multiple resonance layers for complexity
        field3_resonant1 = apply_resonance(field3, intensity=1.5, resonance_type='quantum')
        field3_resonant2 = apply_resonance(field3_resonant1, intensity=0.8, resonance_type='holographic')
        transformed3 = geometric_activation(field3_resonant2, 'all', scale=2.0, resonance=1.5)
        fields.append(transformed3)
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
        for y in range(window_size, GRID_SIZE-window_size, window_size):
            for x in range(window_size, GRID_SIZE-window_size, window_size):
                directions = flow_direction[y-window_size:y+window_size, x-window_size:x+window_size]
                # Circular mean of angles using complex representation
                z = np.mean(np.exp(1j * directions))
                coherence[y-window_size:y+window_size, x-window_size:x+window_size] = np.abs(z)  # 1=perfect coherence, 0=random
                
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
            gx_subset = gx_subset / max_mag * 0.05
            gy_subset = gy_subset / max_mag * 0.05
            
            # Add flow arrows
            ax.quiver(X_subset, Y_subset, gx_subset, gy_subset, 
                      color='cyan', alpha=0.7, scale=1.0, scale_units='inches')
            
        # Add phase coherence contour lines for all visualizations
        coherence = coherence_maps[i]
        # Smooth coherence map for better contours
        coherence_smooth = ndimage.gaussian_filter(coherence, sigma=2.0)
        
        # Plot phase coherence contours
        levels = np.linspace(0.2, 0.9, 5)  # Phase coherence levels
        contour = ax.contour(np.linspace(-1, 1, GRID_SIZE), np.linspace(-1, 1, GRID_SIZE),
                           coherence_smooth, levels=levels, 
                           colors='white', alpha=0.5, linewidths=0.5)
        
        # Add high-coherence region annotations for the final state
        if i == 2:
            # Find regions of high coherence
            high_coherence = coherence_smooth > 0.8
            labeled_regions, num_regions = ndimage.label(high_coherence)
            
            # Add annotation for up to 3 high-coherence regions
            for region_idx in range(1, min(num_regions + 1, 4)):  # cap at 3 annotations
                # Find centroid for each region
                region_coords = np.array(ndimage.center_of_mass(high_coherence, labeled_regions, region_idx)).astype(int)
                center_x, center_y = region_coords[1], region_coords[0]
                
                # Convert to display coordinates
                display_x = np.linspace(-1, 1, GRID_SIZE)[center_x]
                display_y = np.linspace(-1, 1, GRID_SIZE)[center_y]
                
                # Add annotation
                ax.annotate(f"High Coherence Region", xy=(display_x, display_y), xytext=(display_x + 0.2, display_y + 0.2),
                            arrowprops=dict(facecolor='black', shrink=0.05),
                            fontsize=8, ha='center', color='white',
                            path_effects=[PathEffects.withStroke(linewidth=1.5, foreground="black")])
        
        ax.set_title(titles[i], fontsize=12)
        ax.axis('off')  # Hide axes
    
    # Save the figure
    fig.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "field_evolution_combined_enhanced.pdf")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved enhanced field evolution visualization to {output_path}")
    plt.close(fig)

def figure9_geometric_transformation():
    """
    Enhanced visualization for Figure 9: Geometric transformations
    demonstrating transitions between Platonic forms through harmonic resonance
    """
    print("Generating enhanced geometric transformation visualizations (Figure 9)...")
    
    # Define parameter ranges
    num_frames = 20  # Number of transition frames
    morph_range = np.linspace(0, 1, num_frames)
    
    # Generate base grid
    X, Y, R, Theta = generate_grid()
    
    # Create base field with initial phi modulation
    field_base = np.exp(-1.5 * R**2) * (1 + 0.3 * np.sin(Theta * 5 * PHI))
    
    # Create different geometric fields
    solids = ['tetrahedron', 'cube', 'dodecahedron', 'icosahedron']
    transformed_fields = {solid: geometric_activation(field_base, solid, scale=2.0, resonance=1.2) for solid in solids}
    
    # Perform interpolation between two distinct geometric forms
    def geometric_morph(morph_value):
        """Morph between two geometric forms"""
        # Define start/end geometries
        geom1 = transformed_fields['tetrahedron']  
        geom2 = transformed_fields['icosahedron']   
        
        # Ensure shapes are compatible
        if geom1.shape != geom2.shape:
            print("Incompatible array dimensions")
            return np.zeros_like(geom1)
        
        # Use morph_value to blend
        morphed_field = geom1 * (1 - morph_value) + geom2 * morph_value
        return morphed_field
    
    # Generate each transition frame
    for i, morph_value in enumerate(morph_range):
        # Generate blended field
        blended_field = geometric_morph(morph_value)
        
        # Setup figure
        fig, ax = plt.subplots()

#!/usr/bin/env python3
"""
Enhanced Figure Generator for Resonant Field Theory Paper

This script creates high-quality, nuanced visualizations specifically for 
figures 2, 7, and 9 of the Resonant Field Theory paper. It uses pure NumPy
for computational stability and adds detailed geometric elements based on
phi-harmonic resonance for better visual representation.

Author: Crystalline Consciousness Research Group
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon, Circle, Rectangle
import matplotlib.patheffects as PathEffects
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage
import time
from datetime import datetime

# ==== Constants and Settings ====
# Mathematical constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio (φ ≈ 1.618033988749895)
PHI_INV = 1 / PHI          # Inverse golden ratio (φ⁻¹ ≈ 0.618033988749895)
TAU = 2 * np.pi            # Full circle in radians (τ = 2π)

# Visualization settings
GRID_SIZE = 512  # Higher resolution for improved details
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                         'figures_enhanced_20250503_161506')

# ==== Color Maps ====
def create_custom_colormaps():
    """Create custom phi-harmonic colormaps for resonant field visualization"""
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

# ==== Helper Functions ====
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

def geometric_activation(field, solid_type='all', scale=1.0, resonance=1.0):
    """
    Apply geometric activation using pure NumPy.
    This function provides a stable implementation of the geometric transforms.
    
    Args:
        field: 2D NumPy array - the field to transform
        solid_type: string - which Platonic solid to use for transformation
        scale: float - intensity scaling
        resonance: float - resonance factor
        
    Returns:
        2D NumPy array - transformed field
    """
    # Reshape field for 2D grid if needed
    field_2d = field
    if len(field.shape) == 1:
        field_2d = field.reshape(GRID_SIZE, GRID_SIZE)
    elif field.shape != (GRID_SIZE, GRID_SIZE):
        # If field needs reshaping but can't be done directly, create default field
        if field.size != GRID_SIZE * GRID_SIZE:
            print(f"Warning: Cannot reshape field of size {field.size} to {(GRID_SIZE, GRID_SIZE)}")
            field_2d = np.zeros((GRID_SIZE, GRID_SIZE))
        else:
            field_2d = field.reshape(GRID_SIZE, GRID_SIZE)
    
    # Generate coordinates
    X, Y, R, Theta = generate_grid()
    
    # Apply transformation based on solid type
    if solid_type == 'tetrahedron':
        # Tetrahedron: Fire element - sharp, directed energy
        result = np.tanh(field_2d * scale * resonance) * np.cos(Theta * 4)
        
    elif solid_type == 'octahedron':
        # Octahedron: Air element - mobility and phase fluidity
        phase = field_2d * scale * resonance * TAU * PHI_INV
        result = np.sin(phase) * np.cos(phase * PHI)
        
    elif solid_type == 'cube':
        # Cube: Earth element - stability and structure
        result = 1.0 / (1.0 + np.exp(-field_2d * scale * resonance))
        result = result * np.cos(R * 5 * PHI_INV) * 0.2 + result * 0.8
        
    elif solid_type == 'icosahedron':
        # Icosahedron: Water element - flow and adaptive coherence
        phase1 = field_2d * scale * resonance
        phase2 = field_2d * scale * resonance * PHI
        phase3 = 5 * Theta
        
        result = (np.sin(phase1) + np.sin(phase2 + phase3) * PHI_INV) / (1 + PHI_INV)
        
    elif solid_type == 'dodecahedron':
        # Dodecahedron: Aether/spirit element - harmonic synthesis
        phase = field_2d * scale * resonance
        h1 = np.sin(phase)
        h2 = np.sin(phase * PHI) * PHI_INV 
        h3 = np.sin(phase * PHI * PHI) * PHI_INV * PHI_INV
        
        result = (h1 + h2 + h3) / (1 + PHI_INV + PHI_INV * PHI_INV)
        
    else:  # 'all' - blend different geometries
        # Create base field for blending
        t = np.exp(-3 * R**2) * (1 + 0.3 * np.sin(5 * PHI * Theta))
        
        # Tetrahedron component (red)
        tetra = np.tanh(field_2d * scale * resonance) * np.cos(Theta * 4)
        
        # Octahedron component (air/cyan)
        phase_octa = field_2d * scale * resonance * TAU * PHI_INV
        octa = np.sin(phase_octa) * np.cos(phase_octa * PHI)
        
        # Cube component (earth/structure)
        cube = 1.0 / (1.0 + np.exp(-field_2d * scale * resonance))
        cube = cube * np.cos(R * 5 * PHI_INV) * 0.2 + cube * 0.8
        
        # Icosahedron component (water/fluidity)
        phase1_ico = field_2d * scale * resonance
        phase2_ico = field_2d * scale * resonance * PHI
        phase3_ico = 5 * Theta
        ico = (np.sin(phase1_ico) + np.sin(phase2_ico + phase3_ico) * PHI_INV) / (1 + PHI_INV)
        
        # Dodecahedron component (aether/harmonic)
        phase_dod = field_2d * scale * resonance
        h1 = np.sin(phase_dod)
        h2 = np.sin(phase_dod * PHI) * PHI_INV 
        h3 = np.sin(phase_dod * PHI * PHI) * PHI_INV * PHI_INV
        dod = (h1 + h2 + h3) / (1 + PHI_INV + PHI_INV * PHI_INV)
        
        # Blend components with phi-weighted harmonics
        result = tetra + octa * PHI_INV + cube * PHI_INV * PHI_INV + ico * PHI + dod
        result = result / (1 + PHI_INV + PHI_INV * PHI_INV + PHI + 1)
    
    # Normalize to -1 to 1 range for consistent visualization
    result_min, result_max = np.min(result), np.max(result)
    if result_min != result_max:  # Avoid division by zero
        result = 2 * (result - result_min) / (result_max - result_min) - 1
        
    return result

def apply_resonance(data, intensity=1.0, resonance_type='quantum'):
    """
    Apply resonance patterns to data using phi-harmonic principles.
    
    Args:
        data: 2D NumPy array - field to transform
        intensity: float - scaling factor
        resonance_type: string - type of resonance pattern
        
    Returns:
        2D NumPy array - resonated field
    """
    # Reshape if needed
    data_2d = data
    if len(data.shape) == 1:
        if data.size == GRID_SIZE * GRID_SIZE:
            data_2d = data.reshape(GRID_SIZE, GRID_SIZE)
        else:
            # Generate grid for computation if incompatible shape
            X, Y, R, Theta = generate_grid()
            data_2d = np.exp(-2 * R**2) * (1 + 0.2 * np.sin(5 * PHI * Theta))
    
    # Generate coordinates
    X, Y, R, Theta = generate_grid()
    
    if resonance_type == 'quantum':
        # Create quantum-like wave interference
        phase1 = data_2d * intensity * TAU
        phase2 = data_2d * intensity * TAU * PHI
        
        # Apply constructive/destructive interference
        result = np.cos(phase1) * np.sin(phase2 * PHI_INV)
        result = result * np.exp(-0.5 * R**2) + data_2d * (1 - intensity * 0.2)
        
    elif resonance_type == 'holographic':
        # Create holographic-like encoding with phi-based scaling
        # Use positional encoding for distributed representation
        pos_x = np.linspace(0, 1, GRID_SIZE).reshape(1, -1)
        pos_y = np.linspace(0, 1, GRID_SIZE).reshape(-1, 1)
        
        # Create holographic masks with phi-harmonics
        mask1 = np.sin(pos_x * TAU * PHI) * np.cos(pos_y * TAU * PHI_INV)
        mask2 = np.cos(pos_x * TAU * PHI_INV) * np.sin(pos_y * TAU * PHI)
        
        # Apply holographic encoding
        holographic = data_2d * mask1 + data_2d * mask2 * PHI_INV
        
        # Blend with original data
        result = data_2d * (1 - intensity * 0.3) + holographic * intensity * 0.3
        
    else:  # default/fallback
        # Simple resonance effect with phi-based scaling
        modulation = np.sin(5 * PHI * Theta) * PHI_INV + np.cos(3 * R * PHI) * PHI_INV * PHI_INV
        result = data_2d * (1 - intensity * 0.2) + data_2d * modulation * intensity * 0.2
    
    # Normalize to -1 to 1 range
    result_min, result_max = np.min(result), np.max(result)
    if result_min != result_max:  # Avoid division by zero
        result = 2 * (result - result_min) / (result_max - result_min) - 1
        
    return result

# The following function generates the file holographic_encoding_enhanced.pdf.
# Modifications to colormap, alpha, and blending will be performed here to fix the red tint and doubling issue.
def figure11_holographic_encoding():
    """
    Generate enhanced holographic encoding visualization (Figure 11)
    This function generates the holographic_encoding_enhanced.pdf file.
    Modifications to colormap, alpha, and blending will be performed here to fix the red tint and doubling issue.
    """
    print("Generating enhanced holographic encoding visualization (Figure 11)...")

    # Generate the base grid
    X, Y, R, Theta = generate_grid()

    # Create a base field for holographic encoding
    base_field = np.exp(-2 * R**2) * (1 + 0.5 * np.sin(5 * PHI * Theta))

    try:
        # Apply holographic resonance
        holographic_field = apply_resonance(base_field, intensity=1.0, resonance_type='holographic')

        # Create visualization
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(holographic_field, cmap='gray',  # Set colormap to gray to address red tint
                      interpolation='bilinear', extent=[-1, 1, -1, 1],
                      vmin=-1, vmax=1)

        ax.set_title("Enhanced Holographic Encoding", fontsize=14)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        # Save the figure
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, "holographic_encoding_enhanced.pdf")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved enhanced holographic encoding visualization to {output_path}")
        plt.close(fig)

    except Exception as e:
        print(f"Error generating holographic encoding figure: {e}")
        import traceback
        traceback.print_exc()

# ==== Figure Generation Functions ====
def generate_figure2_geometric_basis():
    """
    Generate enhanced visualization for Figure 2 (geometric basis representation)
    with improved symmetry mapping and explicit guides
    """
    print("Generating enhanced geometric basis visualization (Figure 2)...")
    
    # Generate the base grid
    X, Y, R, Theta = generate_grid()
    
    # Create a harmonic field with phi modulation
    field = np.exp(-3 * R**2) * (1 + 0.3 * np.sin(Theta * 5 * PHI))
    
    # Apply geometric activations for each Platonic solid
    solids = ['tetrahedron', 'cube', 'dodecahedron', 'icosahedron', 'all']
    transformed_fields = {}
    
    for solid in solids:
        try:
            # Apply geometric activation
            transformed = geometric_activation(field, solid, scale=2.0, resonance=1.2)
            transformed_fields[solid] = transformed
        except Exception as e:
            print(f"Error processing {solid} field: {e}")
            # Fallback - create an empty field with the right shape
            transformed_fields[solid] = np.zeros((GRID_SIZE, GRID_SIZE))
    
    # Create enhanced visualization with proper gridspec
    from matplotlib import gridspec
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.4)
    
    # Create individual axes
    ax1 = fig.add_subplot(gs[0, 0])  # Original visualization
    ax2 = fig.add_subplot(gs[0, 1])  # Enhanced difference mapping
    ax3 = fig.add_subplot(gs[0, 2])  # Edge detection
    ax_combined = fig.add_subplot(gs[1, :])  # Spanning all columns in second row
    
    # Original visualization 
    im = ax1.imshow(transformed_fields['all'], cmap=CRYSTAL_CMAP, 
                  interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax1.set_title("Original Geometric Basis", fontsize=12)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Enhanced visualization with color-coded difference mapping
    # Create difference field that highlights transitions between geometric patterns
    diff_field = (transformed_fields['tetrahedron'] - transformed_fields['cube'] + 
                 transformed_fields['dodecahedron'] - transformed_fields['icosahedron'])
    diff_field = diff_field / 2.0  # Normalize
    
    im = ax2.imshow(diff_field, cmap=PHASE_CMAP, 
                  interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax2.set_title("Enhanced Difference Mapping", fontsize=12)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Edge detection highlighting key structural features
    # Compute gradient magnitude for edge detection
    gx, gy = np.gradient(transformed_fields['all'])
    edges = np.sqrt(gx**2 + gy**2)
    edges_max = np.max(edges)
    if edges_max > 0:
        edges = edges / edges_max  # Normalize
    
    im = ax3.imshow(edges, cmap='bone', 
                  interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax3.set_title("Edge Detection", fontsize=12)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Combined view with explicit geometric guides is created using gs[1, :]
    # and stored in ax_combined variable
    
    # Create blended visualization with all features
    combined = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    # Base image (geometric field)
    base = CRYSTAL_CMAP(0.5 + 0.5 * transformed_fields['all'])[:, :, :3]
    # Edge overlay
    edge_overlay = np.stack([edges, edges, edges], axis=-1) * np.array([0.8, 0.3, 0.3]).reshape(1, 1, 3)
    # Difference field highlight
    diff_normalized = (diff_field - np.min(diff_field)) / (np.max(diff_field) - np.min(diff_field))
    diff_overlay = PHASE_CMAP(diff_normalized)[:, :, :3] * 0.3
    
    # Combine everything
    combined = base * 0.7 + edge_overlay * 0.5 + diff_overlay * 0.5
    combined = np.clip(combined, 0, 1)
    
    im = ax_combined.imshow(combined, interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax_combined.set_title("Composite View with Explicit Geometric Guides", fontsize=14)
    
    # Add geometric guides
    # Golden ratio circles
    for i in range(1, 5):
        radius = i * 0.2 * PHI_INV
        circle = plt.Circle((0, 0), radius, fill=False, color='gold', linestyle='--', 
                            alpha=0.7, linewidth=0.8)
        ax_combined.add_patch(circle)
        
        # Add radius label with phi notation
        if i == 2 or i == 4:
            label = f"r = {i}φ⁻¹/5"
            txt = ax_combined.text(radius*0.7, radius*0.7, label, color='white', fontsize=9,
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
        ax_combined.plot(x, y, 'o', color='red', markersize=6)
        txt = ax_combined.text(x, y+0.05, f"T{i+1}", color='white', fontsize=9,
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
        ax_combined.plot(x, y, 'o', color='cyan', markersize=6)
        txt = ax_combined.text(x, y+0.05, f"O{i+1}", color='white', fontsize=9,
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
        ax_combined.plot(x, y, 'o', color='magenta', markersize=6)
        txt = ax_combined.text(x, y+0.05, f"I{i+1}", color='white', fontsize=9,
                    ha='center', va='center')
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
    
    # Add legend for geometric elements
    elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Tetrahedron'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=8, label='Octahedron'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='magenta', markersize=8, label='Icosahedron'),
        plt.Line2D([0], [0], color='gold', linestyle='--', label='φ-ratio circles')
    ]
    ax_combined.legend(handles=elements, loc='upper right', fontsize=10)
    
    # Save the figure
    fig.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "geometric_basis_enhanced.pdf")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved enhanced geometric basis visualization to {output_path}")
    plt.close(fig)

def generate_figure7_field_evolution():
    """
    Generate enhanced visualization for Figure 7 (resonant field time evolution)
    with improved phase coherence visualization
    """
    print("Generating enhanced field evolution visualization (Figure 7)...")
    
    # Generate the base grid
    X, Y, R, Theta = generate_grid()
    
    # Create three evolutionary stages of the field
    fields = []
    
    # Stage 1: Initial field formation (simple structure)
    field1 = np.exp(-2 * R**2) * (1 + 0.1 * np.sin(Theta * 3))
    
    try:
        # Apply geometric activation with NumPy implementation
        transformed1 = geometric_activation(field1, 'all', scale=1.0, resonance=0.8)
        fields.append(transformed1)
    except Exception as e:
        print(f"Error processing field1: {e}")
        # Fallback - create a simple field with the right shape
        simple_field = np.exp(-2 * (X**2 + Y**2)) * (1 + 0.1 * np.sin(3 * Theta))
        fields.append(simple_field)
    
    # Stage 2: Complex mid-evolution phase with maximum information density
    # Create more complex field with multiple resonance components
    field2 = np.exp(-1.5 * R**2) * (1 + 
                                  0.3 * np.sin(Theta * 5 * PHI) + 
                                  0.2 * np.cos(R * 8 * PHI_INV))
    
    try:
        # Apply resonance and geometric activation
        field2_resonant = apply_resonance(field2, intensity=1.2, resonance_type='quantum')
        transformed2 = geometric_activation(field2_resonant, 'all', scale=1.5, resonance=1.2)
        fields.append(transformed2)
    except Exception as e:
        print(f"Error processing field2: {e}")
        # Fallback - create a more complex field
        complex_field = np.exp(-1.5 * (X**2 + Y**2)) * (1 + 
                                                     0.3 * np.sin(5 * PHI * Theta) + 
                                                     0.2 * np.cos(8 * PHI_INV * R))
        fields.append(complex_field)
    
    # Stage 3: Stabilized resonance state with harmonic patterns
    field3 = np.exp(-R**2) * (1 + 
                             0.4 * np.sin(Theta * 5 * PHI) + 
                             0.3 * np.cos(R * 8 * PHI_INV) +
                             0.2 * np.sin(R * 10 * Theta))
    
    try:
        # Apply multiple resonance layers for complexity
        field3_resonant1 = apply_resonance(field3, intensity=1.5, resonance_type='quantum')
        field3_resonant2 = apply_resonance(field3_resonant1, intensity=0.8, resonance_type='holographic')
        transformed3 = geometric_activation(field3_resonant2, 'all', scale=2.0, resonance=1.5)
        fields.append(transformed3)
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
        for y in range(window_size, GRID_SIZE-window_size, window_size):
            for x in range(window_size, GRID_SIZE-window_size, window_size):
                directions = flow_direction[y-window_size:y+window_size, x-window_size:x+window_size]
                # Circular mean of angles using complex representation
                z = np.mean(np.exp(1j * directions))
                coherence[y-window_size:y+window_size, x-window_size:x+window_size] = np.abs(z)  # 1=perfect coherence, 0=random
                
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
                gx_subset = gx_subset / max_mag * 0.05
                gy_subset = gy_subset / max_mag * 0.05
                
            # Add flow arrows
            ax.quiver(X_subset, Y_subset, gx_subset, gy_subset, 
                      color='cyan', alpha=0.7, scale=1.0, scale_units='inches')
            
        # Add phase coherence contour lines for all visualizations
        coherence = coherence_maps[i]
        # Smooth coherence map for better contours
        coherence_smooth = ndimage.gaussian_filter(coherence, sigma=2.0)
        
        # Plot phase coherence contours
        levels = np.linspace(0.2, 0.9, 5)  # Phase coherence levels
        contour = ax.contour(np.linspace(-1, 1, GRID_SIZE), np.linspace(-1, 1, GRID_SIZE),
                           coherence_smooth, levels=levels, 
                           colors='white', alpha=0.5, linewidths=0.5)
        
        # Add high-coherence region annotations for the final state
        if i == 2:
            # Find regions of high coherence
            high_coherence = coherence_smooth > 0.8
            labeled_regions, num_regions = ndimage.label(high_coherence)
            
            # Add annotation for up to 3 high-coherence regions
            for region_idx in range(1, min(num_regions + 1, 3)):
                pass

    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Enhanced visualization with color-coded difference mapping
    # Create difference field that highlights transitions between geometric patterns
    diff_field = (transformed_fields['tetrahedron'] - transformed_fields['cube'] + 
                 transformed_fields['dodecahedron'] - transformed_fields['icosahedron'])
    diff_field = diff_field / 2.0  # Normalize
    
    im = ax2.imshow(diff_field, cmap=PHASE_CMAP, 
                  interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax2.set_title("Enhanced Difference Mapping", fontsize=12)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Edge detection highlighting key structural features
    # Compute gradient magnitude for edge detection
    gx, gy = np.gradient(transformed_fields['all'])
    edges = np.sqrt(gx**2 + gy**2)
    edges_max = np.max(edges)
    if edges_max > 0:
        edges = edges / edges_max  # Normalize
    
    im = ax3.imshow(edges, cmap='bone', 
                  interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax3.set_title("Edge Detection", fontsize=12)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Combined view with explicit geometric guides is created using gs[1, :]
    # and stored in ax_combined variable
    
    # Create blended visualization with all features
    combined = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    # Base image (geometric field)
    base = CRYSTAL_CMAP(0.5 + 0.5 * transformed_fields['all'])[:, :, :3]
    # Edge overlay
    edge_overlay = np.stack([edges, edges, edges], axis=-1) * np.array([0.8, 0.3, 0.3]).reshape(1, 1, 3)
    # Difference field highlight
    diff_normalized = (diff_field - np.min(diff_field)) / (np.max(diff_field) - np.min(diff_field))
    diff_overlay = PHASE_CMAP(diff_normalized)[:, :, :3] * 0.3
    
    # Combine everything
    combined = base * 0.7 + edge_overlay * 0.5 + diff_overlay * 0.5
    combined = np.clip(combined, 0, 1)
    
    im = ax_combined.imshow(combined, interpolation='bilinear', extent=[-1, 1, -1, 1])
    ax_combined.set_title("Composite View with Explicit Geometric Guides", fontsize=14)
    
    # Add geometric guides
    # Golden ratio circles
    for i in range(1, 5):
        radius = i * 0.2 * PHI_INV
        circle = plt.Circle((0, 0), radius, fill=False, color='gold', linestyle='--', 
                            alpha=0.7, linewidth=0.8)
        ax_combined.add_patch(circle)
        
        # Add radius label with phi notation
        if i == 2 or i == 4:
            label = f"r = {i}φ⁻¹/5"
            txt = ax_combined.text(radius*0.7, radius*0.7, label, color='white', fontsize=9,
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
        ax_combined.plot(x, y, 'o', color='red', markersize=6)
        txt = ax_combined.text(x, y+0.05, f"T{i+1}", color='white', fontsize=9,
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
        ax_combined.plot(x, y, 'o', color='cyan', markersize=6)
        txt = ax_combined.text(x, y+0.05, f"O{i+1}", color='white', fontsize=9,
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
        ax_combined.plot(x, y, 'o', color='magenta', markersize=6)
        txt = ax_combined.text(x, y+0.05, f"I{i+1}", color='white', fontsize=9,
                    ha='center', va='center')
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
    
    # Add legend for geometric elements
    elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Tetrahedron'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=8, label='Octahedron'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='magenta', markersize=8, label='Icosahedron'),
        plt.Line2D([0], [0], color='gold', linestyle='--', label='φ-ratio circles')
    ]
    ax_combined.legend(handles=elements, loc='upper right', fontsize=10)
    
    # Save the figure
    fig.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "geometric_basis_enhanced.pdf")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved enhanced geometric basis visualization to {output_path}")
    plt.close(fig)

def generate_figure7_field_evolution():
    """
    Generate enhanced visualization for Figure 7 (resonant field time evolution)
    with improved phase coherence visualization
    """
    print("Generating enhanced field evolution visualization (Figure 7)...")
    
    # Generate the base grid
    X, Y, R, Theta = generate_grid()
    
    # Create three evolutionary stages of the field
    fields = []
    
    # Stage 1: Initial field formation (simple structure)
    field1 = np.exp(-2 * R**2) * (1 + 0.1 * np.sin(Theta * 3))
    
    try:
        # Apply geometric activation with NumPy implementation
        transformed1 = geometric_activation(field1, 'all', scale=1.0, resonance=0.8)
        fields.append(transformed1)
    except Exception as e:
        print(f"Error processing field1: {e}")
        # Fallback - create a simple field with the right shape
        simple_field = np.exp(-2 * (X**2 + Y**2)) * (1 + 0.1 * np.sin(3 * Theta))
        fields.append(simple_field)
    
    # Stage 2: Complex mid-evolution phase with maximum information density
    # Create more complex field with multiple resonance components
    field2 = np.exp(-1.5 * R**2) * (1 + 
                                  0.3 * np.sin(Theta * 5 * PHI) + 
                                  0.2 * np.cos(R * 8 * PHI_INV))
    
    try:
        # Apply resonance and geometric activation
        field2_resonant = apply_resonance(field2, intensity=1.2, resonance_type='quantum')
        transformed2 = geometric_activation(field2_resonant, 'all', scale=1.5, resonance=1.2)
        fields.append(transformed2)
    except Exception as e:
        print(f"Error processing field2: {e}")
        # Fallback - create a more complex field
        complex_field = np.exp(-1.5 * (X**2 + Y**2)) * (1 + 
                                                     0.3 * np.sin(5 * PHI * Theta) + 
                                                     0.2 * np.cos(8 * PHI_INV * R))
        fields.append(complex_field)
    
    # Stage 3: Stabilized resonance state with harmonic patterns
    field3 = np.exp(-R**2) * (1 + 
                             0.4 * np.sin(Theta * 5 * PHI) + 
                             0.3 * np.cos(R * 8 * PHI_INV) +
                             0.2 * np.sin(R * 10 * Theta))
    
    try:
        # Apply multiple resonance layers for complexity
        field3_resonant1 = apply_resonance(field3, intensity=1.5, resonance_type='quantum')
        field3_resonant2 = apply_resonance(field3_resonant1, intensity=0.8, resonance_type='holographic')
        transformed3 = geometric_activation(field3_resonant2, 'all', scale=2.0, resonance=1.5)
        fields.append(transformed3)
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
        for y in range(window_size, GRID_SIZE-window_size, window_size):
            for x in range(window_size, GRID_SIZE-window_size, window_size):
                directions = flow_direction[y-window_size:y+window_size, x-window_size:x+window_size]
                # Circular mean of angles using complex representation
                z = np.mean(np.exp(1j * directions))
                coherence[y-window_size:y+window_size, x-window_size:x+window_size] = np.abs(z)  # 1=perfect coherence, 0=random
                
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
                gx_subset = gx_subset / max_mag * 0.05
                gy_subset = gy_subset / max_mag * 0.05
                
            # Add flow arrows
            ax.quiver(X_subset, Y_subset, gx_subset, gy_subset, 
                      color='cyan', alpha=0.7, scale=1.0, scale_units='inches')
            
        # Add phase coherence contour lines for all visualizations
        coherence = coherence_maps[i]
        # Smooth coherence map for better contours
        coherence_smooth = ndimage.gaussian_filter(coherence, sigma=2.0)
        
        # Plot phase coherence contours
        levels = np.linspace(0.2, 0.9, 5)  # Phase coherence levels
        contour = ax.contour(np.linspace(-1, 1, GRID_SIZE), np.linspace(-1, 1, GRID_SIZE),
                           coherence_smooth, levels=levels, 
                           colors='white', alpha=0.5, linewidths=0.5)
        
        # Add high-coherence region annotations for the final state
        if i == 2:
            # Find regions of high coherence
            high_coherence = coherence_smooth > 0.8
            labeled_regions, num_regions = ndimage.label(high_coherence)
            
            # Add annotation for up to 3 high-coherence regions
            for region_idx in range(1, min(num_regions + 1, 4)):
                region_mask = labeled_regions == region_idx
                if np.sum(region_mask) > 100:  # Minimum region size
                    # Get centroid of region
                    y_indices, x_indices = np.where(region_mask)
                    centroid_y = np.mean(y_indices)
                    centroid_x = np.mean(x_indices)
                    
                    # Convert to (-1,1) coordinate space
                    x_pos = (centroid_x / GRID_SIZE) * 2 - 1
                    y_pos = (centroid_y / GRID_SIZE) * 2 - 1
                    
                    # Add annotation
                    txt = ax.text(x_pos, y_pos, f"HC-{region_idx}", color='yellow', fontsize=10,
                               ha='center', va='center', weight='bold')
                    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
        
        # Add geometric resonance metrics for the final state
        if i == 2:
            # Calculate phi-harmonic structural metrics
            resonance_metric = np.abs(np.fft.fft2(fields[i]))
            resonance_metric = np.fft.fftshift(resonance_metric)
            
            # Overlay resonance nodes as circles
            phi_nodes = [
                (0.3 * PHI, 0.3 * PHI_INV),
                (-0.3 * PHI, 0.3 * PHI),
                (0.3 * PHI_INV, -0.3 * PHI),
                (-0.3 * PHI_INV, -0.3 * PHI_INV)
            ]
            
            for x, y in phi_nodes:
                circle = plt.Circle((x, y), 0.1, fill=False, color='gold', 
                                   linestyle='--', alpha=0.8, linewidth=1.0)
                ax.add_patch(circle)
                txt = ax.text(x, y, "φ", color='gold', fontsize=10,
                           ha='center', va='center')
                txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
        
        # Add title and finalize
        ax.set_title(titles[i], fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
    
    # Add a unified title for the figure
    plt.suptitle('Enhanced Field Evolution with Phase Coherence Analysis', fontsize=16, y=0.98)
    
    # Save the combined figure
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "field_evolution_combined_enhanced.pdf")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved enhanced field evolution visualization to {output_path}")
    plt.close(fig)

def figure9_geometric_transformation():
    """
    Generate enhanced visualization for Figure 9 (geometric transformation sequence)
    with detailed transition mapping and explicit geometric guides
    """
    print("Generating enhanced geometric transformation visualizations (Figure 9)...")
    
    # Generate base grid
    X, Y, R, Theta = generate_grid()
    
    # Create a sequence of 20 transformation steps
    num_steps = 20
    fields = []
    
    # Generate base field with phi-harmonic modulation
    base_field = np.exp(-2 * R**2) * (1 + 0.3 * np.sin(5 * PHI * Theta))
    
    # Define a transition sequence across geometric forms
    # We'll morph through: tetrahedron -> octahedron -> cube -> icosahedron -> dodecahedron
    solid_sequence = [
        'tetrahedron',  # Fire element
        'octahedron',   # Air element
        'cube',         # Earth element
        'icosahedron',  # Water element
        'dodecahedron'  # Aether/spirit element
    ]
    
    # Generate transformation fields
    for step in range(num_steps):
        # Determine which solids to blend and their weights
        phase = step / (num_steps - 1)  # 0 to 1
        idx1 = int(phase * (len(solid_sequence) - 1))
        idx2 = min(idx1 + 1, len(solid_sequence) - 1)
        weight = phase * (len(solid_sequence) - 1) - idx1  # 0 to 1 for interpolation
         
        # Get the two solid types to blend
        solid1 = solid_sequence[idx1]
        solid2 = solid_sequence[idx2]
        
        try:
            # Apply geometric activations for the two solids
            transformed1 = geometric_activation(base_field, solid1, scale=1.5, resonance=1.0)
            transformed2 = geometric_activation(base_field, solid2, scale=1.5, resonance=1.0)
            
            # Blend the two transformations based on weight
            blended = transformed1 * (1 - weight) + transformed2 * weight
            
            # Apply resonance with increasing intensity as the sequence progresses
            resonance_intensity = 0.5 + 0.5 * (step / (num_steps - 1))
            resonated = apply_resonance(blended, intensity=resonance_intensity)
            
            fields.append(resonated)
            
        except Exception as e:
            print(f"Error processing transformation step {step}: {e}")
            # Fallback - create a simple field
            simple_field = np.exp(-2 * R**2) * (1 + 0.2 * np.sin(3 * Theta * phase))
            fields.append(simple_field)
    
    # Create visualizations for select frames (0, 10, 19)
    key_frames = [0, 10, 19]
    
    for idx in key_frames:
        field = fields[idx]
        
        # Create figure with enhanced visualization
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))
        
        # Left: Standard visualization
        ax = axs[0]
        im = ax.imshow(field, cmap=CRYSTAL_CMAP, 
                      interpolation='bilinear', extent=[-1, 1, -1, 1],
                      vmin=-1, vmax=1)
        
        # Add title and labels
        phase = idx / (num_steps - 1)  # 0 to 1
        idx1 = int(phase * (len(solid_sequence) - 1))
        idx2 = min(idx1 + 1, len(solid_sequence) - 1)
        solid1 = solid_sequence[idx1]
        solid2 = solid_sequence[idx2]
        weight = phase * (len(solid_sequence) - 1) - idx1  # 0 to 1
        
        # Format title with phase information
        if solid1 != solid2:
            transition_title = f"{solid1.capitalize()} → {solid2.capitalize()} Transition"
            weight_str = f"({weight:.2f})"
        else:
            transition_title = f"{solid1.capitalize()} State"
            weight_str = ""
            
        ax.set_title(f"Frame {idx}: {transition_title} {weight_str}", fontsize=14)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        # Right: Enhanced visualization with geometric guides
        ax = axs[1]
        
        # Compute gradient magnitude for edge detection
        gx, gy = np.gradient(field)
        edges = np.sqrt(gx**2 + gy**2)
        edges_max = np.max(edges)
        if edges_max > 0:
            edges = edges / edges_max  # Normalize
        
        # Create combined visualization
        base = CRYSTAL_CMAP(0.5 + 0.5 * field)[:, :, :3]
        edge_overlay = np.stack([edges, edges, edges], axis=-1) * np.array([0.8, 0.3, 0.3]).reshape(1, 1, 3)
        combined = base * 0.7 + edge_overlay * 0.5
        combined = np.clip(combined, 0, 1)
        
        im = ax.imshow(combined, interpolation='bilinear', extent=[-1, 1, -1, 1])
        ax.set_title(f"Enhanced Visualization with Geometric Guides", fontsize=14)
        
        # Add geometric guides based on current transition state
        # Add visual elements based on which solids are being blended
        
        # Fire/Tetrahedron elements (red)
        if 'tetrahedron' in [solid1, solid2]:
            tetra_weight = 1.0 if solid1 == 'tetrahedron' else (
                            1.0 - weight if solid2 == 'tetrahedron' else 0.0)
            
            if tetra_weight > 0.2:
                # Add tetrahedral vertices
                tetra_vertices = [
                    [0.7, 0.7],  # Normalized for visibility
                    [0.7, -0.7],
                    [-0.7, 0.7],
                    [-0.7, -0.7]
                ]
                
                # Connect vertices to form tetrahedron edges
                for i in range(len(tetra_vertices)):
                    for j in range(i+1, len(tetra_vertices)):
                        x1, y1 = tetra_vertices[i]
                        x2, y2 = tetra_vertices[j]
                        ax.plot([x1, x2], [y1, y2], '-', color='red', 
                               alpha=tetra_weight * 0.7, linewidth=1.5)
                
                # Add vertices
                for i, (x, y) in enumerate(tetra_vertices):
                    ax.plot(x, y, 'o', color='red', alpha=tetra_weight, markersize=6)
        
        # Air/Octahedron elements (cyan)
        if 'octahedron' in [solid1, solid2]:
            octa_weight = 1.0 if solid1 == 'octahedron' else (
                          1.0 - weight if solid2 == 'octahedron' else 0.0)
            
            if octa_weight > 0.2:
                # Add octahedral vertices
                octa_vertices = [
                    [0.8, 0],
                    [-0.8, 0],
                    [0, 0.8],
                    [0, -0.8],
                    [0, 0],  # Center point for visibility
                ]
                
                # Connect vertices to form octahedron edges
                for i in range(len(octa_vertices)-1):  # Exclude center for edges
                    # Connect to center
                    x1, y1 = octa_vertices[i]
                    x2, y2 = octa_vertices[-1]  # Center point
                    ax.plot([x1, x2], [y1, y2], '-', color='cyan', 
                           alpha=octa_weight * 0.7, linewidth=1.5)
                
                # Add vertices
                for i, (x, y) in enumerate(octa_vertices[:-1]):  # Exclude center for dots
                    ax.plot(x, y, 'o', color='cyan', alpha=octa_weight, markersize=6)
        
        # Earth/Cube elements (green)
        if 'cube' in [solid1, solid2]:
            cube_weight = 1.0 if solid1 == 'cube' else (
                         1.0 - weight if solid2 == 'cube' else 0.0)
            
            if cube_weight > 0.2:
                # Add cube vertices (2D projection)
                cube_vertices = [
                    [0.6, 0.6],
                    [0.6, -0.6],
                    [-0.6, 0.6],
                    [-0.6, -0.6]
                ]
                
                # Draw cube as rectangle
                rect = Rectangle((-0.6, -0.6), 1.2, 1.2, fill=False, 
                                edgecolor='green', alpha=cube_weight * 0.7, 
                                linewidth=1.5)
                ax.add_patch(rect)
                
                # Add vertices
                for i, (x, y) in enumerate(cube_vertices):
                    ax.plot(x, y, 'o', color='green', alpha=cube_weight, markersize=6)
        
        # Water/Icosahedron elements (blue)
        if 'icosahedron' in [solid1, solid2]:
            ico_weight = 1.0 if solid1 == 'icosahedron' else (
                        1.0 - weight if solid2 == 'icosahedron' else 0.0)
            
            if ico_weight > 0.2:
                # Create pentagonal representation for icosahedron
                ico_vertices = []
                for i in range(5):
                    angle = i * TAU / 5
                    r = 0.7
                    x = r * np.cos(angle)
                    y = r * np.sin(angle)
                    ico_vertices.append([x, y])
                
                # Connect vertices to form pentagon
                for i in range(len(ico_vertices)):
                    j = (i + 1) % len(ico_vertices)
                    x1, y1 = ico_vertices[i]
                    x2, y2 = ico_vertices[j]
                    ax.plot([x1, x2], [y1, y2], '-', color='blue', 
                           alpha=ico_weight * 0.7, linewidth=1.5)
                
                # Add second inner pentagon with phi ratio
                inner_ico_vertices = []
                for i in range(5):
                    angle = (i * TAU / 5) + (TAU / 10)  # Offset
                    r = 0.7 * PHI_INV  # Phi ratio
                    x = r * np.cos(angle)
                    y = r * np.sin(angle)
                    inner_ico_vertices.append([x, y])
                
                # Connect inner vertices
                for i in range(len(inner_ico_vertices)):
                    j = (i + 1) % len(inner_ico_vertices)
                    x1, y1 = inner_ico_vertices[i]
                    x2, y2 = inner_ico_vertices[j]
                    ax.plot([x1, x2], [y1, y2], '-', color='blue', 
                           alpha=ico_weight * 0.7, linewidth=1.5)
                
                # Connect outer to inner
                for i in range(len(ico_vertices)):
                    x1, y1 = ico_vertices[i]
                    x2, y2 = inner_ico_vertices[i]
                    ax.plot([x1, x2], [y1, y2], '-', color='blue', 
                           alpha=ico_weight * 0.5, linewidth=1.0)
                
                # Add vertices
                for x, y in ico_vertices + inner_ico_vertices:
                    ax.plot(x, y, 'o', color='blue', alpha=ico_weight, markersize=4)
        
        # Spirit/Dodecahedron elements (purple)
        if 'dodecahedron' in [solid1, solid2]:
            dod_weight = 1.0 if solid1 == 'dodecahedron' else (
                         1.0 - weight if solid2 == 'dodecahedron' else 0.0)
            
            if dod_weight > 0.2:
                # Create simple visual representation for dodecahedron
                # Use a pentagon and five triangles
                
                # Central pentagon vertices
                dod_vertices = []
                for i in range(5):
                    angle = i * TAU / 5
                    r = 0.4
                    x = r * np.cos(angle)
                    y = r * np.sin(angle)
                    dod_vertices.append([x, y])
                
                # Connect pentagon vertices
                for i in range(len(dod_vertices)):
                    j = (i + 1) % len(dod_vertices)
                    x1, y1 = dod_vertices[i]
                    x2, y2 = dod_vertices[j]
                    ax.plot([x1, x2], [y1, y2], '-', color='purple', 
                           alpha=dod_weight * 0.7, linewidth=1.5)
                
                # Draw outer points at golden ratio distance
                outer_dod_vertices = []
                for i in range(5):
                    angle = i * TAU / 5
                    r = 0.4 * PHI  # Golden ratio
                    x = r * np.cos(angle)
                    y = r * np.sin(angle)
                    outer_dod_vertices.append([x, y])
                    
                    # Connect to pentagon
                    ax.plot([x, dod_vertices[i][0]], [y, dod_vertices[i][1]], '-', 
                           color='purple', alpha=dod_weight * 0.7, linewidth=1.5)
                
                # Add vertices
                for x, y in dod_vertices:
                    ax.plot(x, y, 'o', color='purple', alpha=dod_weight, markersize=5)
                for x, y in outer_dod_vertices:
                    ax.plot(x, y, 'o', color='purple', alpha=dod_weight, markersize=5)
        
        # Add a legend showing current transformation
        elements = []
        if 'tetrahedron' in [solid1, solid2]:
            elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                    markersize=8, label='Tetrahedron (Fire)'))
        if 'octahedron' in [solid1, solid2]:
            elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', 
                                    markersize=8, label='Octahedron (Air)'))
        if 'cube' in [solid1, solid2]:
            elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                                    markersize=8, label='Cube (Earth)'))
        if 'icosahedron' in [solid1, solid2]:
            elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                                    markersize=8, label='Icosahedron (Water)'))
        if 'dodecahedron' in [solid1, solid2]:
            elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
                                    markersize=8, label='Dodecahedron (Aether)'))
            
        if elements:
            ax.legend(handles=elements, loc='upper right', fontsize=10)
        
        # Save figure
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f"geometric_transformation_enhanced_{idx}.pdf")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved geometric transformation visualization frame {idx} to {output_path}")
        plt.close(fig)

def generate_visualization_legend():
    """Generate a reference legend explaining the visualizations used in the paper"""
    print("Generating visualization legend...")
    
    # Create figure for legend
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Make the axis invisible
    ax.axis('off')
    
    # Add title
    ax.text(0.5, 0.98, "Resonant Field Theory Visualization Guide", 
           ha='center', va='top', fontsize=16, fontweight='bold')
    
    # Add explanation sections
    y_pos = 0.92
    
    # 1. Colormaps
    ax.text(0.5, y_pos, "1. Visualization Colormaps", 
           ha='center', va='top', fontsize=14, fontweight='bold')
    y_pos -= 0.05
    
    # Display color gradients
    cmap_length = 0.7
    cmap_height = 0.03
    cmap_x = 0.15
    
    # PHI colormap
    ax.text(0.1, y_pos, "Phi-Harmonic:", ha='left', va='center', fontsize=12)
    
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    
    ax.imshow(gradient, aspect='auto', cmap=PHI_CMAP, 
             extent=[cmap_x, cmap_x + cmap_length, y_pos - cmap_height/2, y_pos + cmap_height/2])
    y_pos -= 0.06
    
    # Crystal colormap
    ax.text(0.1, y_pos, "Crystal:", ha='left', va='center', fontsize=12)
    ax.imshow(gradient, aspect='auto', cmap=CRYSTAL_CMAP, 
             extent=[cmap_x, cmap_x + cmap_length, y_pos - cmap_height/2, y_pos + cmap_height/2])
    y_pos -= 0.06
    
    # Phase colormap
    ax.text(0.1, y_pos, "Phase:", ha='left', va='center', fontsize=12)
    ax.imshow(gradient, aspect='auto', cmap=PHASE_CMAP, 
             extent=[cmap_x, cmap_x + cmap_length, y_pos - cmap_height/2, y_pos + cmap_height/2])
    y_pos -= 0.08
    
    # 2. Geometric Elements
    ax.text(0.5, y_pos, "2. Geometric Elements", 
           ha='center', va='top', fontsize=14, fontweight='bold')
    y_pos -= 0.05
    
    # Create list of geometric elements
    geometries = [
        ("Tetrahedron (Fire)", "red", "Directed energy, transformation"),
        ("Octahedron (Air)", "cyan", "Mobility, phase fluidity"),
        ("Cube (Earth)", "green", "Stability, structure"),
        ("Icosahedron (Water)", "blue", "Flow, adaptive coherence"),
        ("Dodecahedron (Aether)", "purple", "Harmonic synthesis, integration")
    ]
    
    for i, (name, color, desc) in enumerate(geometries):
        ax.text(0.1, y_pos, f"• {name}:", ha='left', va='center', 
               fontsize=12, color=color, fontweight='bold')
        ax.text(0.45, y_pos, desc, ha='left', va='center', fontsize=11)
        y_pos -= 0.05
    
    y_pos -= 0.03
    
    # 3. Mathematical Constants
    ax.text(0.5, y_pos, "3. Key Mathematical Constants", 
           ha='center', va='top', fontsize=14, fontweight='bold')
    y_pos -= 0.05
    
    # List key mathematical constants
    constants = [
        ("φ (phi)", f"{PHI:.10f}", "Golden ratio - basis for harmonic resonance"),
        ("φ⁻¹", f"{PHI_INV:.10f}", "Inverse golden ratio - resonance damping factor"),
        ("τ (tau)", f"{TAU:.10f}", "2π - full cycle in oscillatory systems")
    ]
    
    for i, (name, val, desc) in enumerate(constants):
        ax.text(0.1, y_pos, f"• {name}:", ha='left', va='center', 
               fontsize=12, fontweight='bold')
        ax.text(0.3, y_pos, val, ha='left', va='center', fontsize=11, 
               family='monospace')
        ax.text(0.5, y_pos, desc, ha='left', va='center', fontsize=11)
        y_pos -= 0.05
    
    y_pos -= 0.03
    
    # 4. Visualization Elements
    ax.text(0.5, y_pos, "4. Visualization Annotations", 
           ha='center', va='top', fontsize=14, fontweight='bold')
    y_pos -= 0.05
    
    # List visualization elements and their meanings
    elements = [
        ("HC-1, HC-2, ...", "High Coherence Regions", "Areas of strong phase alignment"),
        ("→ (arrows)", "Field Flow", "Direction of resonant energy movement"),
        ("--- (contour lines)", "Phase Coherence", "Isoclines of similar phase alignment"),
        ("⚪ (colored dots)", "Geometric Vertices", "Platonic solid projection points"),
        ("⚪ with φ", "φ-Resonance Node", "Locations of golden ratio resonance")
    ]
    
    for i, (sym, name, desc) in enumerate(elements):
        ax.text(0.1, y_pos, sym, ha='left', va='center', fontsize=12, fontweight='bold')
        ax.text(0.3, y_pos, name, ha='left', va='center', fontsize=11, fontweight='bold')
        ax.text(0.55, y_pos, desc, ha='left', va='center', fontsize=11)
        y_pos -= 0.05
    
    # Save legend
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "visualization_legend.pdf")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization legend to {output_path}")
    plt.close(fig)
d
def main():
    """Execute all enhanced figure generation functions"""
    print(f"Generating enhanced figures for Resonant Field Theory paper...")

    # Ensure output directory exists
    ensure_output_dir()

    # Generate enhanced figures
    start_time = time.time()

    try:
        # Generate Figure 11 (Holographic Encoding)
        figure11_holographic_encoding()

        # Generate Figure 2 (Geometric Basis)
        generate_figure2_geometric_basis()

        # Generate Figure 7 (Field Evolution)
        generate_figure7_field_evolution()

        # Generate Figure 9 (Geometric Transformation)
        figure9_geometric_transformation()

        # Generate visualization legend
        generate_visualization_legend()
    except Exception as e:
        print(f"Error generating figures: {e}")
        import traceback
        traceback.print_exc()
        return 1
        return 0

if __name__ == "__main__":
    try:
        exit_code = main()

