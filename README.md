# Resonant Field Theory Figure Enhancement

This directory contains scripts to enhance Figure 2 (geometric basis) and Figure 8 (field evolution) in the Resonant Field Theory paper, making subtle differences between stages clearly visible to human observers.

## Overview

The original figures have subtle differences that are difficult to perceive visually. The enhancement scripts apply:

- Color-coded difference mapping
- Edge detection and highlighting
- Geometric guides overlay
- Phase-space trajectory visualization
- Temporal evolution indicators
- Magnified insets of key changing regions

## Requirements

- macOS (scripts optimized for Apple Silicon with Metal/MPS)
- Python 3.9+
- Poppler (for PDF processing)
- Python packages: numpy, matplotlib, mlx, pdf2image, Pillow, opencv-python, scipy

## Usage

Simply run the enhancement script:

```bash
# Make the scripts executable
chmod +x enhance_figures.sh
chmod +x enhance_visualizations.py

# Run the script to set up dependencies and enhance figures
./enhance_figures.sh
```

The enhanced figures will be saved in a new directory named `figures_enhanced_[timestamp]`, along with:
- A combined visualization showing all stages side by side
- A color legend explaining the visual elements
- A guide for updating your LaTeX document with the enhanced figures

## Updating the LaTeX Document

After running the script, follow the instructions in the generated `latex_update_guide.txt` to update your LaTeX document with the enhanced figures.

# Resonant Field Theory Visualization

This directory contains the LaTeX paper and visualization tools for the Resonant Field Theory project, exploring the geometric approach to physics and consciousness through crystalline resonance patterns.

## Generating Visualizations

The `crystalviz` package provides comprehensive visualization capabilities for the Resonant Field Theory. To generate visualizations for the paper, run:

```bash
# Basic visualization generation
python test_visualizations.py

# For high-resolution visualizations with GPU acceleration (if available)
python test_visualizations.py --high-res --use-gpu
```

## Available Visualization Modes

The visualization system provides several modes for exploring different aspects of the Resonant Field Theory:

1. **Main Paper Figure** - A comprehensive visualization showing all key aspects of the theory
2. **Platonic Frequency Alignments** - Visualization of the relationship between frequency patterns and Platonic solids
3. **Holographic Encoding** - Visualization of correlation matrices and holographic information encoding
4. **Phase Space Trajectories** - Visualization of the system's dynamics in phase space
5. **Neural Field Coherence** - Visualization of coherence patterns and connectivity graphs
6. **Geometric Basis Functions** - Visualization of the tetrahedral, cubic, and dodecahedral basis functions
7. **Field Evolution Animation** - Animation of the field's evolution over time
8. **Geometric Transformation** - Animation of transitions between geometric basis functions

## Command Line Interface

For more control over visualization generation, the package provides a command-line interface:

```bash
# Get help information
python -m crystalviz.cli --help

# Generate a specific visualization (e.g., platonic frequencies)
python -m crystalviz.cli --mode platonic --use-gpu --output-dir ./figs
```

## Paper Integration

The generated figures are ready to be included in the LaTeX paper. Use the following LaTeX command:

```latex
\includegraphics[width=\textwidth]{resonant_field_figures/resonant_field_theory_main.pdf}
```

## Mathematical Framework

The visualization system implements the core equations of Resonant Field Theory:

1. Crystalline Resonance Function:
   ```
   Ψ_crystal(r) = ∑[v_i + θ_i]exp(-r²/σ_i²) × {T₄(r), C₈(r), D₁₂(r)}
   ```

2. Field Evolution Equation:
   ```
   ∂Ψ/∂t = [-iĤ + D∇²]Ψ + ∑ F̂_iΨ(r/σ_i)
   ```

The system leverages MLX/MPS for GPU acceleration on macOS, providing efficient computation for real-time rendering of complex geometric patterns.

