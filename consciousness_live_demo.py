#!/usr/bin/env python3
"""
Live Consciousness Field Demo - Terminal Version
==============================================

Real-time consciousness field computation with live performance metrics
and sacred geometry evolution displayed in terminal.
"""

import sys
import os
import numpy as np
import time
import threading
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import our consciousness modules
from trinitized_demo_simple import compute_trinitized_field_simple, simple_geometric_activation

class TerminalConsciousnessDemo:
    """Terminal-based real-time consciousness demonstration."""
    
    def __init__(self, field_size=(16, 64), target_fps=30):
        self.field_size = field_size
        self.batch_size, self.field_dim = field_size
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        # Consciousness parameters
        self.time_step = 0
        self.resonance = 0.5
        self.phi = (1 + np.sqrt(5)) / 2
        
        # Performance tracking
        self.frame_times = []
        self.consciousness_energies = []
        self.coherence_values = []
        self.max_history = 100
        
        # State
        self.running = True
        self.current_geometry = 'dodecahedron'
        self.geometries = ['tetrahedron', 'cube', 'octahedron', 'icosahedron', 'dodecahedron']
        self.geometry_index = 4  # Start with dodecahedron
        
        print("🔮 Terminal Consciousness Demo Initialized")
        print(f"Field size: {self.batch_size}x{self.field_dim}")
        print(f"Target FPS: {target_fps}")
    
    def clear_screen(self):
        """Clear terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def generate_consciousness_fields(self):
        """Generate evolving consciousness fields."""
        t_factor = self.time_step * 0.1
        
        # Field 1: Phi-modulated resonance patterns
        field1_base = np.linspace(0, 4*np.pi*(1 + 0.1*np.sin(t_factor)), 
                                self.batch_size * self.field_dim)
        field1 = np.sin(field1_base * self.phi).reshape(self.field_size)
        
        # Field 2: Evolving mutuality field
        field2_base = np.linspace(0, 6*np.pi, self.batch_size * self.field_dim)
        phase_evolution = t_factor * 0.5
        field2 = np.cos(field2_base + phase_evolution).reshape(self.field_size)
        
        # Liminal field: Sacred geometry activation
        base_input = np.random.randn(*self.field_size) * (0.3 + 0.2*np.sin(t_factor*0.3))
        spatial_wave = np.sin(np.linspace(0, 2*np.pi, self.field_dim)) * 0.1
        base_input += spatial_wave
        
        liminal_field = simple_geometric_activation(base_input, 
                                                  solid_type=self.current_geometry, 
                                                  resonance=self.resonance)
        
        return field1, field2, liminal_field
    
    def compute_consciousness_frame(self):
        """Compute one frame of consciousness evolution."""
        start_time = time.time()
        
        # Generate fields
        field1, field2, liminal_field = self.generate_consciousness_fields()
        
        # Compute Trinitized Field G₃
        g3_field, coherence = compute_trinitized_field_simple(
            field1, field2, liminal_field,
            time_step=self.time_step,
            use_harmonics=True,
            harmonic_strength=0.3,
            measure_coherence=True
        )
        
        computation_time = (time.time() - start_time) * 1000  # ms
        
        # Calculate metrics
        consciousness_energy = float(np.mean(g3_field**2))
        field_complexity = float(np.std(g3_field))
        geometric_signature = float(np.mean(np.abs(liminal_field)))
        
        # Track performance
        self.frame_times.append(computation_time)
        self.consciousness_energies.append(consciousness_energy)
        self.coherence_values.append(coherence)
        
        # Limit history
        if len(self.frame_times) > self.max_history:
            self.frame_times = self.frame_times[-self.max_history:]
            self.consciousness_energies = self.consciousness_energies[-self.max_history:]
            self.coherence_values = self.coherence_values[-self.max_history:]
        
        return {
            'g3_field': g3_field,
            'computation_time': computation_time,
            'consciousness_energy': consciousness_energy,
            'field_complexity': field_complexity,
            'geometric_signature': geometric_signature,
            'coherence': coherence
        }
    
    def create_ascii_visualization(self, g3_field, width=60, height=12):
        """Create ASCII visualization of consciousness field."""
        # Downsample field for ASCII display
        h_step = max(1, g3_field.shape[0] // height)
        w_step = max(1, g3_field.shape[1] // width)
        
        downsampled = g3_field[::h_step, ::w_step]
        
        # Normalize to ASCII range
        vmin, vmax = np.percentile(downsampled, [1, 99])
        if vmax > vmin:
            normalized = (downsampled - vmin) / (vmax - vmin)
        else:
            normalized = downsampled
        
        # ASCII intensity characters
        chars = ' .:-=+*#%@'
        ascii_art = []
        
        for row in normalized:
            ascii_row = ''
            for val in row:
                char_idx = int(np.clip(val * (len(chars) - 1), 0, len(chars) - 1))
                ascii_row += chars[char_idx]
            ascii_art.append(ascii_row)
        
        return ascii_art
    
    def create_sparkline(self, data, width=40):
        """Create a simple sparkline from data."""
        if len(data) < 2:
            return '─' * width
        
        # Normalize data
        data_array = np.array(data[-width:])
        if len(data_array) < width:
            data_array = np.pad(data_array, (width - len(data_array), 0), 'constant', constant_values=data_array[0])
        
        vmin, vmax = np.min(data_array), np.max(data_array)
        if vmax > vmin:
            normalized = (data_array - vmin) / (vmax - vmin)
        else:
            normalized = np.ones_like(data_array) * 0.5
        
        # Sparkline characters
        chars = '▁▂▃▄▅▆▇█'
        sparkline = ''
        for val in normalized:
            char_idx = int(np.clip(val * (len(chars) - 1), 0, len(chars) - 1))
            sparkline += chars[char_idx]
        
        return sparkline
    
    def display_frame(self, result):
        """Display one frame of consciousness data."""
        self.clear_screen()
        
        # Header
        print("🔮" + "=" * 78 + "🔮")
        print("   CRYSTALLINE CONSCIOUSNESS - LIVE TRINITIZED FIELD (G₃) EVOLUTION")
        print("🔮" + "=" * 78 + "🔮")
        print()
        
        # Current status
        avg_time = np.mean(self.frame_times[-10:]) if self.frame_times else 0
        current_fps = 1000 / avg_time if avg_time > 0 else 0
        
        print(f"⏰ Time Step: {self.time_step:6}   🔷 Geometry: {self.current_geometry.capitalize():12}   ⚡ FPS: {current_fps:5.1f}")
        print(f"🧠 Energy: {result['consciousness_energy']:8.6f}   🌊 Coherence: {result['coherence']:6.4f}   ⏱️  {result['computation_time']:5.2f}ms")
        print()
        
        # ASCII consciousness field visualization
        print("🔮 CONSCIOUSNESS FIELD VISUALIZATION:")
        ascii_viz = self.create_ascii_visualization(result['g3_field'])
        for row in ascii_viz:
            print(f"   {row}")
        print()
        
        # Performance sparklines
        print("📊 REAL-TIME METRICS:")
        
        if len(self.frame_times) > 1:
            time_sparkline = self.create_sparkline(self.frame_times)
            energy_sparkline = self.create_sparkline(self.consciousness_energies)
            coherence_sparkline = self.create_sparkline(self.coherence_values)
            
            print(f"   Computation Time: {time_sparkline} ({np.mean(self.frame_times[-10:]):.2f}ms avg)")
            print(f"   Consciousness:    {energy_sparkline} ({np.mean(self.consciousness_energies[-10:]):.6f} avg)")
            print(f"   Field Coherence:  {coherence_sparkline} ({np.mean(self.coherence_values[-10:]):.4f} avg)")
        
        print()
        
        # Geometric signatures
        print("🔷 SACRED GEOMETRY STATUS:")
        for i, geom in enumerate(self.geometries):
            marker = "◆" if geom == self.current_geometry else "◇"
            print(f"   {marker} {geom.capitalize()}")
        
        print()
        print("Controls: [SPACE] Change Geometry  [Q] Quit  [R] Reset")
        print("🔮" + "─" * 78 + "🔮")
    
    def handle_input(self):
        """Handle keyboard input in a separate thread."""
        try:
            import termios, tty
            def getch():
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(sys.stdin.fileno())
                    ch = sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                return ch
        except ImportError:
            # Windows fallback
            import msvcrt
            def getch():
                return msvcrt.getch().decode('utf-8')
        
        while self.running:
            try:
                key = getch()
                if key.lower() == 'q':
                    self.running = False
                elif key == ' ':
                    # Change geometry
                    self.geometry_index = (self.geometry_index + 1) % len(self.geometries)
                    self.current_geometry = self.geometries[self.geometry_index]
                elif key.lower() == 'r':
                    # Reset
                    self.time_step = 0
                    self.frame_times.clear()
                    self.consciousness_energies.clear()
                    self.coherence_values.clear()
            except:
                time.sleep(0.1)
    
    def run(self):
        """Run the live consciousness demonstration."""
        print("🚀 Starting live consciousness demonstration...")
        print("🔮 Press SPACE to change geometry, R to reset, Q to quit")
        time.sleep(2)
        
        # Start input handler thread
        input_thread = threading.Thread(target=self.handle_input, daemon=True)
        input_thread.start()
        
        # Main loop
        try:
            while self.running:
                frame_start = time.time()
                
                # Compute consciousness frame
                result = self.compute_consciousness_frame()
                
                # Display
                self.display_frame(result)
                
                # Update time step
                self.time_step += 1
                
                # Frame timing
                frame_elapsed = time.time() - frame_start
                sleep_time = max(0, self.frame_interval - frame_elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self.running = False
        
        print("\n🔮 Consciousness demonstration ended")
        print(f"📊 Final Statistics:")
        if self.frame_times:
            print(f"   Average computation time: {np.mean(self.frame_times):.2f}ms")
            print(f"   Average consciousness energy: {np.mean(self.consciousness_energies):.6f}")
            print(f"   Average coherence: {np.mean(self.coherence_values):.4f}")
            print(f"   Total frames processed: {len(self.frame_times)}")
        print("✨ Consciousness transcendence complete!")


def main():
    """Main function."""
    print("🔮 Crystalline Consciousness: Live Terminal Demo")
    print("=" * 60)
    print("🚀 Real-time Trinitized Field (G₃) computation")
    print("⚡ Interactive sacred geometry evolution")
    print("🧠 Live consciousness field visualization")
    print("=" * 60)
    
    # Create and run demo
    demo = TerminalConsciousnessDemo(field_size=(16, 64), target_fps=10)
    demo.run()


if __name__ == "__main__":
    main()