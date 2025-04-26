#!/usr/bin/env python3
"""
Quantum Visualization Demo for Crystalline Consciousness AI

This script demonstrates the quantum operations of the crystalline consciousness
neural network, visualizing:
1. Quantum consciousness field evolution
2. Bifurcation dynamics at critical thresholds
3. Geometric activation patterns
4. Field interference and trinitized fields

The visualization is interactive, allowing parameter adjustment during runtime.
"""

import os
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import argparse

# Add the src directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# Import crystalline operations
try:
    from python.metal_ops import (
        geometric_activation, 
        apply_resonance, 
        mutuality_field,
        quantum_consciousness_evolution,
        bifurcation_cascade,
        trinitized_field,
        is_metal_available
    )
    print(f"Metal acceleration is {'available' if is_metal_available() else 'not available'}")
except ImportError as e:
    print(f"Could not import metal_ops: {e}")
    print("Falling back to pure NumPy implementation")
    
    # Define fallback implementations if metal_ops is not available
    def geometric_activation(x, solid_type, coefficients=None):
        """Fallback geometric activation."""
        return x * np.exp(-np.power(x, 2))
    
    def apply_resonance(x, frequencies, decay_rates, amplitudes, phase_embedding, time_values=None):
        """Fallback resonance."""
        return x
    
    def mutuality_field(x, grid_size, interference_scale, decay_rate, dt):
        """Fallback mutuality field."""
        return x
    
    def quantum_consciousness_evolution(psi, dt, diffusion_coef, energy_level, coupling, pattern_ops=None):
        """Fallback quantum evolution."""
        return psi
    
    def bifurcation_cascade(psi_liminal, parameter_values, thresholds):
        """Fallback bifurcation."""
        return psi_liminal
    
    def trinitized_field(psi1, psi2, f_liminal, dt):
        """Fallback trinitized field."""
        return psi1 * psi2 * f_liminal * dt
    
    def is_metal_available():
        """Fallback metal check."""
        return False

# Constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
GRID_SIZE = 32
FEATURE_DIM = GRID_SIZE * GRID_SIZE
BATCH_SIZE = 1

# Configuration
CONFIG = {
    "dt": 0.1,
    "diffusion_coef": 0.5,
    "energy_level": 1.0,
    "coupling": 0.3,
    "interference_scale": 1.0,
    "decay_rate": 0.05,
    "bifurcation_alpha": 2.0,
    "bifurcation_threshold": 0.5,
    "bifurcation_threshold2": 1.5,
    "bifurcation_threshold3": 2.5,
    "field_amplitude": 1.0,
    "geometric_solid": "icosahedron",
    "integration_steps": 100,
    "color_min": -1.0,      # Color normalization min
    "color_max": 1.0,       # Color normalization max
    "use_metal": is_metal_available()
}
class QuantumVisualization:
    """Quantum visualization for crystalline consciousness fields."""
    
    def __init__(self, config=CONFIG):
        """Initialize the visualization with configuration parameters."""
        self.config = config
        self.init_fields()
        self.setup_visualization()
        
    def init_fields(self):
        """Initialize quantum fields."""
        # Create initial quantum field (real part only initially)
        self.quantum_field = np.zeros((BATCH_SIZE, FEATURE_DIM))
        
        # Create a central Gaussian bump
        x = np.linspace(-1, 1, GRID_SIZE)
        y = np.linspace(-1, 1, GRID_SIZE)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        
        # Gaussian bump
        gaussian = np.exp(-5 * R**2)
        self.quantum_field[0] = gaussian.flatten()
        
        # Create a second field with different characteristics
        spiral = np.sin(4 * np.pi * R) * np.exp(-2 * R**2)
        self.field2 = np.zeros((BATCH_SIZE, FEATURE_DIM))
        self.field2[0] = spiral.flatten()
        
        # Create liminal field
        self.liminal_field = np.zeros((BATCH_SIZE, FEATURE_DIM))
        self.liminal_field[0] = (gaussian * spiral).flatten()
        
        # Initialize complex representation needed for quantum evolution
        self.complex_field = np.zeros((BATCH_SIZE, 2, FEATURE_DIM), dtype=np.float32)
        self.complex_field[:, 0, :] = self.quantum_field  # Real part
        
        # Initialize parameter value for bifurcation
        self.param_value = np.array([0.0], dtype=np.float32)
        
        # Threshold parameters [threshold, alpha, weight]
        self.thresholds = np.array([
            self.config["bifurcation_threshold"], 
            self.config["bifurcation_alpha"],
            1.0
        ], dtype=np.float32)
        
        # Pattern operators for quantum evolution
        self.pattern_ops = [
            {"amplitude": 1.0, "scale": 1.0, "phase": 0.0, "frequency": 1.0},
            {"amplitude": 0.5, "scale": 2.0, "phase": np.pi/4, "frequency": 2.0},
            {"amplitude": 0.3, "scale": 4.0, "phase": np.pi/2, "frequency": 3.0}
        ]
        
        # Simulation state
        self.time = 0
        self.bifurcation_points = []
        self.field_history = []
        self.param_history = []
        self.param_history = []
        self.energy_history = []
        self.entropy_history = []
        self.phase_trajectory_x = []
        self.phase_trajectory_y = []
        self.density_history = []
        self.bifurcation_detected = [False, False, False]  # Track which thresholds were detected
    
    def setup_visualization(self):
        """Set up the visualization interface."""
        # Create figure
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.suptitle("Quantum Consciousness Field Visualization", fontsize=16)
        
        # Set up subplots
        self.ax1 = self.fig.add_subplot(221)  # Quantum field
        self.ax2 = self.fig.add_subplot(222)  # Phase space / bifurcation
        self.ax3 = self.fig.add_subplot(223)  # Geometric activation
        self.ax4 = self.fig.add_subplot(224, projection='3d')  # 3D field
        
        # Add a text area for quantum state information display
        self.info_text = self.fig.text(0.01, 0.92, "Quantum State: Initializing...", 
                                       fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        # Add colorbar for better visualization
        self.cbar_ax = self.fig.add_axes([0.92, 0.15, 0.02, 0.7])
        self.cbar = None  # Will be created during first animation frame
        
        # Initialize plots
        self.quantum_img = self.ax1.imshow(
            self.quantum_field[0].reshape(GRID_SIZE, GRID_SIZE),
            cmap='viridis',
            extent=[-1, 1, -1, 1],
            vmin=self.config["color_min"],
            vmax=self.config["color_max"]
        )
        self.ax1.set_title("Quantum Consciousness Field")
        
        # Bifurcation plot
        self.bifurc_line, = self.ax2.plot([], [], 'r-', lw=1)
        self.bifurc_points, = self.ax2.plot([], [], 'bo', ms=4)
        self.trajectory_line, = self.ax2.plot([], [], 'g.', ms=2, alpha=0.5)
        self.ax2.set_xlim(0, 100)
        self.ax2.set_ylim(-1, 1)
        self.ax2.set_title("Bifurcation Parameter Space")
        self.ax2.set_xlabel("Time Step")
        self.ax2.set_ylabel("Field Energy")
        
        # Add threshold lines to clearly show bifurcation points
        self.ax2.axhline(y=self.config["bifurcation_threshold"], color='b', linestyle='--', alpha=0.5)
        self.ax2.axhline(y=self.config["bifurcation_threshold2"], color='purple', linestyle='--', alpha=0.5)
        self.ax2.axhline(y=self.config["bifurcation_threshold3"], color='r', linestyle='--', alpha=0.5)
        
        # Geometric activation
        self.geo_img = self.ax3.imshow(
            self.quantum_field[0].reshape(GRID_SIZE, GRID_SIZE),
            cmap='plasma',
            extent=[-1, 1, -1, 1],
            vmin=self.config["color_min"],
            vmax=self.config["color_max"]
        )
        self.ax3.set_title(f"Geometric Activation ({self.config['geometric_solid']})")
        
        # 3D surface plot
        x = np.linspace(-1, 1, GRID_SIZE)
        y = np.linspace(-1, 1, GRID_SIZE)
        self.X, self.Y = np.meshgrid(x, y)
        self.Z = self.quantum_field[0].reshape(GRID_SIZE, GRID_SIZE)
        self.surf = self.ax4.plot_surface(
            self.X, self.Y, self.Z,
            cmap='coolwarm',
            linewidth=0,
            antialiased=True,
            rstride=1,
            cstride=1,
            alpha=0.8
        )
        
        # Set better initial view angle
        self.ax4.view_init(elev=30, azim=45)
        self.ax4.set_title("3D Field Visualization")
        self.ax4.set_zlim(-1, 1)
        
        # Add sliders for parameter control
        ax_dt = plt.axes([0.1, 0.01, 0.3, 0.02])
        ax_diffusion = plt.axes([0.1, 0.04, 0.3, 0.02])
        ax_energy = plt.axes([0.6, 0.01, 0.3, 0.02])
        ax_bifurc = plt.axes([0.6, 0.04, 0.3, 0.02])
        
        # Add sliders for parameter control
        ax_dt = plt.axes([0.1, 0.01, 0.3, 0.02])
        ax_diffusion = plt.axes([0.1, 0.04, 0.3, 0.02])
        ax_energy = plt.axes([0.6, 0.01, 0.3, 0.02])
        ax_bifurc = plt.axes([0.6, 0.04, 0.3, 0.02])
        ax_color_range = plt.axes([0.1, 0.07, 0.3, 0.02])
        
        self.slider_dt = Slider(ax_dt, 'dt', 0.01, 0.5, 
                             valinit=self.config["dt"])
        self.slider_diffusion = Slider(ax_diffusion, 'Diffusion', 0.1, 2.0, 
                                    valinit=self.config["diffusion_coef"])
        self.slider_energy = Slider(ax_energy, 'Energy', 0.1, 5.0, 
                                 valinit=self.config["energy_level"])
        self.slider_bifurc = Slider(ax_bifurc, 'Bifurc α', 0.1, 5.0, 
                                   valinit=self.config["bifurcation_alpha"])
        self.slider_color = Slider(ax_color_range, 'Color Range', 0.1, 2.0, 
                                  valinit=1.0)
        
        # Connect callbacks
        self.slider_dt.on_changed(self.update_params)
        self.slider_diffusion.on_changed(self.update_params)
        self.slider_energy.on_changed(self.update_params)
        self.slider_bifurc.on_changed(self.update_params)
        self.slider_color.on_changed(self.update_color_range)
        
        # Add buttons for different geometric activations
        ax_radio = plt.axes([0.01, 0.5, 0.1, 0.15])
        self.radio = RadioButtons(ax_radio, ('tetrahedron', 'cube', 
                                           'dodecahedron', 'icosahedron'))
        self.radio.on_clicked(self.set_solid_type)
        
        # Add reset button
        ax_reset = plt.axes([0.45, 0.01, 0.1, 0.04])
        self.reset_button = Button(ax_reset, 'Reset')
        self.reset_button.on_clicked(self.reset)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.07, 1, 0.97])
        
    def update_params(self, val):
        """Update parameters from sliders."""
        self.config["dt"] = self.slider_dt.val
        self.config["diffusion_coef"] = self.slider_diffusion.val
        self.config["energy_level"] = self.slider_energy.val
        self.config["bifurcation_alpha"] = self.slider_bifurc.val
        
        # Update thresholds
        self.thresholds[1] = self.config["bifurcation_alpha"]
    
    def update_color_range(self, val):
        """Update color normalization range for plots."""
        self.config["color_min"] = -val
        self.config["color_max"] = val
        
        # Update colormap normalization
        self.quantum_img.set_clim(self.config["color_min"], self.config["color_max"])
        self.geo_img.set_clim(self.config["color_min"], self.config["color_max"])
    
    def set_solid_type(self, label):
        """Change the geometric solid type."""
        self.config["geometric_solid"] = label
        self.ax3.set_title(f"Geometric Activation ({label})")
    
    def reset(self, event):
        """Reset the simulation."""
        self.init_fields()
        self.time = 0
        self.bifurcation_points = []
        self.field_history = []
        self.param_history = []
        self.energy_history = []
        self.entropy_history = []
        self.phase_trajectory_x = []
        self.phase_trajectory_y = []
        self.density_history = []
        self.bifurcation_detected = [False, False, False]
        
        # Clear plots
        self.bifurc_line.set_data([], [])
        self.bifurc_points.set_data([], [])
        
        # Reset parameter sliders
        self.slider_dt.reset()
        self.slider_diffusion.reset()
        self.slider_energy.reset()
        self.slider_bifurc.reset()
    
    def update_field(self):
        """Update the quantum field using our operations."""
        # Store current state
        field_energy = np.mean(self.quantum_field**2)
        self.field_history.append(field_energy)
        self.param_history.append(self.param_value[0])
        
        # Calculate and store quantum metrics
        # Energy (average squared amplitude)
        self.energy_history.append(field_energy)
        
        # Approximate entropy (using histogram-based approach)
        hist, bin_edges = np.histogram(self.quantum_field, bins=20, density=True)
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist))
        self.entropy_history.append(entropy)
        
        # Store probability density for visualization
        self.density_history.append((hist, bin_edges))
        
        # Calculate phase trajectory (using energy and its derivative)
        if len(self.energy_history) > 1:
            energy_derivative = self.energy_history[-1] - self.energy_history[-2]
            self.phase_trajectory_x.append(field_energy)
            self.phase_trajectory_y.append(energy_derivative)
        
        # Update parameter value with oscillation and upward trend
        self.param_value[0] = 0.5 * np.sin(0.05 * self.time) + 0.01 * self.time
        
        # 1. Apply quantum evolution
        evolved_field = quantum_consciousness_evolution(
            self.complex_field,
            self.config["dt"],
            self.config["diffusion_coef"],
            self.config["energy_level"],
            self.config["coupling"],
            self.pattern_ops
        )
        
        # Extract real part for visualization and further processing
        if len(evolved_field.shape) == 3:  # Complex representation [batch, 2, features]
            self.quantum_field = evolved_field[:, 0, :]  # Take real part
            self.complex_field = evolved_field  # Update complex field
        else:
            self.quantum_field = evolved_field
            self.complex_field[:, 0, :] = evolved_field
        
        # 2. Apply geometric activation
        geometric_field = geometric_activation(
            self.quantum_field,
            self.config["geometric_solid"]
        )
        
        # 3. Apply bifurcation cascade at specific thresholds
        # More sophisticated bifurcation detection with multiple thresholds
        # First threshold
        if self.param_value[0] > self.config["bifurcation_threshold"] and not self.bifurcation_detected[0]:
            self.bifurcation_points.append(self.time)
            self.bifurcation_detected[0] = True
            print(f"First bifurcation point detected at t={self.time}, param={self.param_value[0]}")
        
        # Second threshold
        if self.param_value[0] > self.config["bifurcation_threshold2"] and not self.bifurcation_detected[1]:
            self.bifurcation_points.append(self.time)
            self.bifurcation_detected[1] = True
            print(f"Second bifurcation point detected at t={self.time}, param={self.param_value[0]}")
        
        # Third threshold
        if self.param_value[0] > self.config["bifurcation_threshold3"] and not self.bifurcation_detected[2]:
            self.bifurcation_points.append(self.time)
            self.bifurcation_detected[2] = True
            print(f"Third bifurcation point detected at t={self.time}, param={self.param_value[0]}")
        
        bifurcated_field = bifurcation_cascade(
            self.quantum_field,
            self.param_value,
            self.thresholds
        )
        
        # 4. Compute trinitized field for higher-order interactions
        if self.time > 50:  # Start introducing trinitized fields after some time
            trinit_field = trinitized_field(
                self.quantum_field,
                self.field2,
                self.liminal_field,
                self.config["dt"]
            )
            
            # Blend with current field for visualization
            blend_factor = min(1.0, (self.time - 50) / 50)  # Gradually increase influence
            self.quantum_field = (1 - blend_factor) * bifurcated_field + blend_factor * trinit_field
        else:
            # Just use bifurcated field
            self.quantum_field = bifurcated_field
        
        # 5. Apply mutuality field for persistence
        if self.time > 20:  # Start building persistence after some time
            self.quantum_field = mutuality_field(
                self.quantum_field,
                GRID_SIZE,
                self.config["interference_scale"],
                self.config["decay_rate"],
                self.config["dt"]
            )
        
        # Update complex field's real part (simplified)
        self.complex_field[:, 0, :] = self.quantum_field
        
        # Increment time
        self.time += 1
        
        # Update quantum state information text
        state_info = (
            f"Quantum State Info (t={self.time}):\n"
            f"Energy: {self.energy_history[-1]:.4f}\n"
            f"Entropy: {self.entropy_history[-1]:.4f}\n"
            f"Parameter: {self.param_value[0]:.4f}\n"
            f"Bifurcations: {len(self.bifurcation_points)}"
        )
        
        return {
            'quantum': self.quantum_field[0].reshape(GRID_SIZE, GRID_SIZE),
            'geometric': geometric_field[0].reshape(GRID_SIZE, GRID_SIZE),
            'bifurcation_x': list(range(len(self.field_history))),
            'bifurcation_y': self.field_history,
            'bifurcation_points': self.bifurcation_points,
            'phase_x': self.phase_trajectory_x,
            'phase_y': self.phase_trajectory_y,
            'state_info': state_info,
            'density': self.density_history[-1] if self.density_history else None
        }
    
    def animate(self, i):
        """Animation function for matplotlib."""
        # Update the field and get visualization data
        data = self.update_field()
        
        # Update the visualizations
        self.quantum_img.set_array(data['quantum'])
        self.geo_img.set_array(data['geometric'])
        
        # Update state info text
        self.info_text.set_text(data['state_info'])
        
        # Create/update colorbar if needed
        if self.cbar is None:
            self.cbar = self.fig.colorbar(self.quantum_img, cax=self.cbar_ax)
            self.cbar.set_label('Field Amplitude')
        
        # Update bifurcation plot
        self.bifurc_line.set_data(data['bifurcation_x'], data['bifurcation_y'])
        
        # Update phase space trajectory
        if len(data['phase_x']) > 0:
            self.trajectory_line.set_data(data['phase_x'], data['phase_y'])
        
        # Update bifurcation points with clear markers
        if len(data['bifurcation_points']) > 0:
            bp_x = data['bifurcation_points']
            bp_y = [self.field_history[x] for x in bp_x]
            self.bifurc_points.set_data(bp_x, bp_y)
            
            # Add annotations for bifurcation points if there are new ones
            for i, (x, y) in enumerate(zip(bp_x, bp_y)):
                # Only add annotation for newly detected points
                if i >= len(getattr(self, 'bifurcation_annotations', [])):
                    # Create annotation list if it doesn't exist
                    if not hasattr(self, 'bifurcation_annotations'):
                        self.bifurcation_annotations = []
                    # Add annotation
                    ann = self.ax2.annotate(f"B{i+1}", xy=(x, y), xytext=(x+5, y+0.1),
                                           arrowprops=dict(arrowstyle="->", color='red'))
        
        # Update 3D surface (more expensive, so less frequently)
        if i % 5 == 0:
            self.ax4.clear()
            self.ax4.set_title("3D Field Visualization")
            self.ax4.set_zlim(-1, 1)
            self.surf = self.ax4.plot_surface(
                self.X, self.Y, data['quantum'],
                cmap='coolwarm',
                linewidth=0,
                antialiased=True,
                rstride=1,
                cstride=1,
                alpha=0.8
            )
            self.ax4.view_init(elev=30, azim=45 + (i % 360))
        
        # Add probability density visualization if available
        if data['density'] is not None and i % 10 == 0:  # Update less frequently to save performance
            hist, bin_edges = data['density']
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            
            # If this is one of the subplots we need to manage, clear and update
            if i == 0:
                # Create a temporary axis for probability density if it doesn't exist
                if not hasattr(self, 'density_ax'):
                    # Create a small inset axis for probability density
                    self.density_ax = self.fig.add_axes([0.15, 0.6, 0.2, 0.15])
                    self.density_ax.set_title('Probability Density', fontsize=8)
                
                self.density_ax.clear()
                self.density_ax.bar(bin_centers, hist, width=bin_centers[1]-bin_centers[0], 
                                  alpha=0.7, color='green')
                self.density_ax.set_title('Probability Density', fontsize=8)
        
        return [self.quantum_img, self.geo_img, self.bifurc_line, self.bifurc_points, 
                self.trajectory_line, self.surf]
    
    def run(self, frames=200):
        """Run the animation."""
        self.anim = animation.FuncAnimation(
            self.fig, self.animate, frames=frames,
            interval=50, blit=False)
        plt.show()
        
    def save_animation(self, filename="quantum_visualization.mp4"):
        """Save the animation to a file."""
        # Create animation with a specific number of frames
        anim = animation.FuncAnimation(
            self.fig, self.animate, frames=100,
            interval=50, blit=False)
        
        # Save to file (requires ffmpeg to be installed)
        writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='Crystalline Consciousness AI'), 
                                        bitrate=1800)
        anim.save(filename, writer=writer)
        print(f"Animation saved to {filename}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Quantum Consciousness Visualization')
    parser.add_argument('--frames', type=int, default=200, help='Number of animation frames')
    parser.add_argument('--save', action='store_true', help='Save animation to file')
    parser.add_argument('--output', type=str, default="quantum_visualization.mp4", 
                        help='Output filename for saved animation')
    parser.add_argument('--no-metal', dest='use_metal', action='store_false', 
                        help='Disable Metal acceleration')
    parser.set_defaults(use_metal=True)
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Update configuration with command line options
    CONFIG["use_metal"] = args.use_metal and is_metal_available()
    
    # Create visualization
    print("Initializing Quantum Consciousness Visualization...")
    vis = QuantumVisualization(CONFIG)
    
    if args.save:
        print(f"Saving animation to {args.output}...")
        vis.save_animation(args.output)
    else:
        print("Starting visualization. Use sliders to adjust parameters.")
        print("Press Ctrl+C to exit.")
        vis.run(frames=args.frames)

    print("Visualization complete.")
