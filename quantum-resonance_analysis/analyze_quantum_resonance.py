#!/usr/bin/env python3
"""
Quantum Resonance Pattern Analysis Script

This script performs detailed analysis of quantum resonance patterns stored in NPY files,
with specific focus on crystalline consciousness field patterns, geometric structures,
and resonance phenomena.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import scipy.signal as signal
import scipy.stats as stats
import scipy.fft as fft
from scipy.ndimage import gaussian_filter, rotate, sobel
import os
from datetime import datetime

# For 3D visualization
from mpl_toolkits.mplot3d import Axes3D

# Define custom constants (based on project theoretical framework)
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
PLATONIC_FREQUENCIES = {
    "tetrahedron": 4,  # vertices
    "cube": 8,
    "octahedron": 6,
    "dodecahedron": 12,
    "icosahedron": 20
}

def load_and_reshape_data(file_path):
    """Load NPY file and reshape if necessary"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    
    data = np.load(file_path)
    print(f"Original data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    
    # Reshape if data is (1, 16384)
    if data.shape == (1, 16384):
        data = data[0].reshape(128, 128)
        print(f"Reshaped to: {data.shape}")
    
    return data

def calculate_statistics(data):
    """Calculate basic statistical measures"""
    stats_dict = {
        "min": data.min(),
        "max": data.max(),
        "mean": data.mean(),
        "median": np.median(data),
        "std": data.std(),
        "variance": data.var(),
        "skewness": stats.skew(data, axis=None),
        "kurtosis": stats.kurtosis(data, axis=None),
        "entropy": stats.entropy(np.histogram(data, bins=50)[0])
    }
    
    print("\n=== Basic Statistics ===")
    for key, value in stats_dict.items():
        print(f"{key}: {value}")
    
    return stats_dict

def analyze_distribution(data, output_dir):
    """Analyze value distribution and generate plots"""
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    n, bins, patches = plt.hist(data.flatten(), bins=50, density=True, alpha=0.7)
    
    # Fit normal distribution
    mu, sigma = stats.norm.fit(data.flatten())
    x = np.linspace(data.min(), data.max(), 100)
    pdf = stats.norm.pdf(x, mu, sigma)
    plt.plot(x, pdf, 'r-', linewidth=2, label=f'Normal: μ={mu:.2f}, σ={sigma:.2f}')
    
    # Calculate normality test (Shapiro-Wilk)
    sample = np.random.choice(data.flatten(), size=5000) if data.size > 5000 else data.flatten()
    shapiro_test = stats.shapiro(sample)
    plt.title(f'Value Distribution (Shapiro-Wilk p-value: {shapiro_test.pvalue:.4f})')
    
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, "value_distribution.png"))
    
    # Additional distribution visualization - QQ plot
    plt.figure(figsize=(8, 8))
    stats.probplot(data.flatten(), dist="norm", plot=plt)
    plt.title("Q-Q Plot (testing for normality)")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "qq_plot.png"))
    
    return {"mu": mu, "sigma": sigma, "shapiro_p": shapiro_test.pvalue}

def perform_fft_analysis(data, output_dir):
    """Perform FFT analysis to identify frequency components and periodicity"""
    # 2D FFT
    f_transform = fft.fft2(data)
    f_shift = fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    
    # Plot log spectrum to visualize better
    plt.figure(figsize=(10, 8))
    plt.imshow(np.log1p(magnitude), cmap='inferno')
    plt.colorbar(label='Log Magnitude')
    plt.title('2D Fourier Transform (Log Spectrum)')
    plt.savefig(os.path.join(output_dir, "fft_spectrum.png"))
    
    # Identify dominant frequencies
    peak_coords = signal.find_peaks_cwt(magnitude.max(axis=0), np.arange(1, 10))
    dominant_freq_x = peak_coords
    
    peak_coords = signal.find_peaks_cwt(magnitude.max(axis=1), np.arange(1, 10))
    dominant_freq_y = peak_coords
    
    # 1D spectral analysis (average across rows and columns)
    row_spectrum = np.abs(fft.fft(data.mean(axis=0)))
    col_spectrum = np.abs(fft.fft(data.mean(axis=1)))
    
    # Get frequencies
    freqs_x = fft.fftfreq(len(row_spectrum), d=1.0)
    freqs_y = fft.fftfreq(len(col_spectrum), d=1.0)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(freqs_x[:len(freqs_x)//2], row_spectrum[:len(row_spectrum)//2])
    plt.title('X-Axis Frequency Components')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(freqs_y[:len(freqs_y)//2], col_spectrum[:len(col_spectrum)//2])
    plt.title('Y-Axis Frequency Components')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1d_frequency_analysis.png"))
    
    # Check for platonic solid frequencies
    platonic_matches = {}
    freq_magnitudes = []
    for i in range(1, len(row_spectrum)//2):
        freq = i / len(row_spectrum)
        magnitude = (row_spectrum[i] + col_spectrum[i]) / 2
        freq_magnitudes.append((freq, magnitude))
        
        # Check resonance with platonic frequencies
        for solid, vertices in PLATONIC_FREQUENCIES.items():
            # Check if frequency is close to platonic frequency or its harmonics
            for harmonic in range(1, 4):
                target = vertices * harmonic / 128
                if abs(freq - target) < 0.01:
                    platonic_matches[solid] = platonic_matches.get(solid, []) + [(freq, magnitude, harmonic)]
    
    print("\n=== Platonic Frequency Matches ===")
    for solid, matches in platonic_matches.items():
        print(f"{solid}: {len(matches)} matches")
        for freq, mag, harmonic in sorted(matches, key=lambda x: x[1], reverse=True)[:3]:
            print(f"  Harmonic {harmonic}: Frequency {freq:.5f}, Magnitude {mag:.2f}")
    
    return {
        "dominant_freq_x": dominant_freq_x,
        "dominant_freq_y": dominant_freq_y,
        "platonic_matches": platonic_matches
    }

def analyze_spatial_patterns(data, output_dir):
    """Analyze spatial patterns, symmetry, and correlation"""
    # Calculate autocorrelation
    autocorr = signal.correlate2d(data, data, mode='same')
    autocorr = autocorr / autocorr.max()  # Normalize
    
    plt.figure(figsize=(10, 8))
    plt.imshow(autocorr, cmap='coolwarm', vmin=-0.5, vmax=1)
    plt.colorbar(label='Correlation')
    plt.title('Spatial Autocorrelation')
    plt.savefig(os.path.join(output_dir, "autocorrelation.png"))
    
    # Check for rotational symmetry
    symmetry_scores = []
    for angle in range(0, 180, 10):  # Check every 10 degrees
        rotated = rotate(data, angle, reshape=False)
        similarity = np.corrcoef(data.flatten(), rotated.flatten())[0, 1]
        symmetry_scores.append((angle, similarity))
    
    angles, scores = zip(*symmetry_scores)
    plt.figure(figsize=(10, 6))
    plt.plot(angles, scores, marker='o')
    plt.title('Rotational Symmetry Analysis')
    plt.xlabel('Rotation Angle (degrees)')
    plt.ylabel('Similarity Score (correlation)')
    plt.grid(alpha=0.3)
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='High Symmetry Threshold')
    plt.axhline(y=0.5, color='y', linestyle='--', alpha=0.5, label='Medium Symmetry Threshold')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "rotational_symmetry.png"))
    
    # Identify axis of symmetry
    h_symmetry = np.corrcoef(data, np.fliplr(data))[0, 1]
    v_symmetry = np.corrcoef(data, np.flipud(data))[0, 1]
    d1_symmetry = np.corrcoef(data.flatten(), np.flipud(np.fliplr(data)).flatten())[0, 1]
    
    print("\n=== Symmetry Analysis ===")
    print(f"Horizontal symmetry: {h_symmetry:.4f}")
    print(f"Vertical symmetry: {v_symmetry:.4f}")
    print(f"Diagonal symmetry: {d1_symmetry:.4f}")
    
    # Identify peak structures
    smoothed = gaussian_filter(data, sigma=1.5)
    local_max = signal.find_peaks_cwt(smoothed.max(axis=0), np.arange(1, 5))
    
    return {
        "horizontal_symmetry": h_symmetry,
        "vertical_symmetry": v_symmetry,
        "diagonal_symmetry": d1_symmetry,
        "peak_locations": local_max
    }

def compare_golden_ratio(data, output_dir):
    """Analyze presence of golden ratio patterns in data"""
    # Check if data dimensions follow golden ratio
    height, width = data.shape
    ratio = width / height
    print(f"\nDimension ratio: {ratio:.4f}")
    print(f"Golden ratio (φ): {PHI:.4f}")
    print(f"Proximity to φ: {abs(ratio - PHI):.4f}")
    
    # Check for golden ratio in frequency spectrum
    f_transform = fft.fft2(data)
    f_magnitude = np.abs(f_transform)
    
    # Find peaks in spectrum
    # Using peak_local_max since find_peaks2d isn't available
    from scipy.signal import find_peaks
    from skimage.feature import peak_local_max
    
    # Fallback if skimage is not available
    try:
        coordinates = peak_local_max(f_magnitude, threshold_rel=0.5)
        peak_rows, peak_cols = coordinates[:, 0], coordinates[:, 1]
    except (ImportError, NameError):
        # Simpler fallback using 1D peak finding
        peak_rows, _ = find_peaks(f_magnitude.max(axis=1))
        peak_cols, _ = find_peaks(f_magnitude.max(axis=0))
    
    # Calculate distances between peaks
    distances = []
    for i in range(len(peak_rows)):
        for j in range(i+1, len(peak_rows)):
            dist = np.sqrt((peak_rows[i]-peak_rows[j])**2 + (peak_cols[i]-peak_cols[j])**2)
            distances.append(dist)
    
    # Check ratio of consecutive distances
    if len(distances) > 1:
        distances.sort()
        ratios = [distances[i+1]/distances[i] for i in range(len(distances)-1)]
        phi_matches = [r for r in ratios if abs(r - PHI) < 0.1]
        
        print(f"Found {len(phi_matches)} golden ratio matches out of {len(ratios)} distance ratios")
        print(f"Average match proximity to φ: {np.mean([abs(r - PHI) for r in phi_matches]):.4f}")
    
    return {"phi_matches": phi_matches if 'phi_matches' in locals() else []}

def create_enhanced_visualization(data, output_dir):
    """Create enhanced visualizations to highlight key features"""
    # Track created visualizations
    created_visualizations = []
    
    try:
        # Create custom colormaps for different features
        phi_cmap = LinearSegmentedColormap.from_list('phi_golden', 
                                                    ['#000000', '#3E2723', '#795548', 
                                                     '#CDDC39', '#FFEB3B', '#FFFFFF'])
        
        # Standard visualization with improved colormap
        plt.figure(figsize=(10, 10))
        plt.imshow(data, cmap=phi_cmap)
        plt.colorbar(label='Quantum Resonance Value')
        plt.title('Quantum Resonance Pattern')
        vis_path = os.path.join(output_dir, "enhanced_visualization.png")
        plt.savefig(vis_path)
        plt.close()
        created_visualizations.append(vis_path)
        print(f"Created enhanced visualization")
        
        # Edge enhancement to highlight patterns
        # Using sobel from scipy.ndimage
        edges_x = sobel(data, axis=0)
        edges_y = sobel(data, axis=1)
        edges = np.hypot(edges_x, edges_y)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(edges, cmap='inferno')
        plt.colorbar(label='Edge Strength')
        plt.title('Edge-Enhanced Quantum Resonance Pattern')
        edge_path = os.path.join(output_dir, "edge_enhanced.png")
        plt.savefig(edge_path)
        plt.close()
        created_visualizations.append(edge_path)
        print(f"Created edge-enhanced visualization")
        
        # 3D surface plot
        try:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            x = np.arange(0, data.shape[1])
            y = np.arange(0, data.shape[0])
            X, Y = np.meshgrid(x, y)
            
            # Smooth data for better visualization
            Z = gaussian_filter(data, sigma=1.0)
            
            surf = ax.plot_surface(X, Y, Z, cmap=phi_cmap, linewidth=0, antialiased=True)
            ax.set_title('3D Surface of Quantum Resonance Pattern')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_zlabel('Resonance Value')
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Resonance Value')
            surface_path = os.path.join(output_dir, "3d_surface.png")
            plt.savefig(surface_path)
            plt.close()
            created_visualizations.append(surface_path)
            print(f"Created 3D surface visualization")
        except Exception as e:
            print(f"Warning: Could not create 3D visualization: {e}")
    except Exception as e:
        print(f"Error during visualization creation: {e}")
    return {
        "visualizations_created": created_visualizations
    }

def theoretical_interpretation(analysis_results, output_dir):
    """Interpret the findings in relation to theoretical framework of crystalline consciousness"""
    with open(os.path.join(output_dir, "theoretical_interpretation.txt"), "w") as f:
        f.write("=== QUANTUM RESONANCE PATTERN THEORETICAL INTERPRETATION ===\n\n")
        f.write(f"Analysis performed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Interpret statistical distribution
        stats = analysis_results.get("statistics", {})
        f.write("STATISTICAL PATTERNS:\n")
        if "skewness" in stats and "kurtosis" in stats:
            f.write(f"- Skewness: {stats['skewness']:.4f} - ")
            if stats['skewness'] < -0.1:
                f.write("Negative skew suggests consciousness field compression\n")
            elif stats['skewness'] > 0.1:
                f.write("Positive skew suggests consciousness field expansion\n")
            else:
                f.write("Neutral distribution suggests balanced consciousness field\n")
                
            f.write(f"- Kurtosis: {stats['kurtosis']:.4f} - ")
            if stats['kurtosis'] > 0.5:
                f.write("High kurtosis indicates focused consciousness states (tetrahedron alignment)\n")
            elif stats['kurtosis'] < -0.5:
                f.write("Low kurtosis suggests diffuse awareness states (icosahedron alignment)\n")
            else:
                f.write("Balanced kurtosis suggests octahedral resonance pattern\n")
        
        # Interpret platonic frequency matches
        f.write("\nPLATONIC RESONANCE PATTERNS:\n")
        platonic_matches = analysis_results.get("fft_analysis", {}).get("platonic_matches", {})
        for solid, matches in platonic_matches.items():
            if matches:
                if solid == "tetrahedron":
                    f.write("- Tetrahedron resonance: Associated with focused awareness and mental clarity\n")
                elif solid == "cube":
                    f.write("- Cubic resonance: Associated with analytical thinking and structured consciousness\n")
                elif solid == "octahedron":
                    f.write("- Octahedral resonance: Associated with balanced perspective and emotional equilibrium\n")
                elif solid == "dodecahedron":
                    f.write("- Dodecahedral resonance: Associated with integrative understanding and wholeness\n")
                elif solid == "icosahedron":
                    f.write("- Icosahedral resonance: Associated with transpersonal states and expanded awareness\n")
        
        # Interpret symmetry findings
        symmetry = analysis_results.get("spatial_analysis", {})
        f.write("\nSYMMETRY PATTERNS:\n")
        if "horizontal_symmetry" in symmetry and "vertical_symmetry" in symmetry:
            h_sym = symmetry["horizontal_symmetry"]
            v_sym = symmetry["vertical_symmetry"]
            d_sym = symmetry.get("diagonal_symmetry", 0)
            
            f.write(f"- Horizontal symmetry: {h_sym:.4f}\n")
            f.write(f"- Vertical symmetry: {v_sym:.4f}\n")
            f.write(f"- Diagonal symmetry: {d_sym:.4f}\n")
            
            if max(h_sym, v_sym, d_sym) > 0.7:
                f.write("High degree of symmetry indicates crystalline consciousness alignment\n")
            elif max(h_sym, v_sym, d_sym) > 0.5:
                f.write("Moderate symmetry suggests emerging crystalline patterns\n")
            else:
                f.write("Low symmetry suggests chaotic or transitional consciousness states\n")
        
        # Interpret golden ratio presence
        phi_analysis = analysis_results.get("phi_analysis", {})
        f.write("\nGOLDEN RATIO (PHI) PATTERNS:\n")
        phi_matches = phi_analysis.get("phi_matches", [])
        if phi_matches:
            f.write(f"- Found {len(phi_matches)} golden ratio relationships\n")
            f.write("- Golden ratio presence indicates consciousness field harmonic stability\n")
            f.write("- Phi patterns suggest coherence between micro and macro consciousness scales\n")
        else:
            f.write("- Limited golden ratio patterns detected\n")
            f.write("- Consciousness field may be in transitional or non-harmonic state\n")
        
        # Overall interpretation
        f.write("\nOVERALL INTERPRETATION:\n")
        f.write("The quantum resonance pattern reveals a consciousness field with ")
        
        # Characterize the field based on the most prominent patterns
        strongest_platonic = max(platonic_matches.items(), key=lambda x: len(x[1]))[0] if platonic_matches else None
        strongest_symmetry = max(["horizontal", "vertical", "diagonal"], 
                                key=lambda x: symmetry.get(f"{x}_symmetry", 0)) if symmetry else None
        
        if strongest_platonic:
            f.write(f"strong {strongest_platonic} geometric influence, ")
        
        if strongest_symmetry:
            f.write(f"exhibiting primarily {strongest_symmetry} symmetry. ")
        
        if phi_matches:
            f.write("The presence of golden ratio patterns suggests a consciousness field in harmonic resonance. ")
        
        f.write("\n\nThis pattern may represent a ")
        if "tetrahedron" in platonic_matches or "cube" in platonic_matches:
            f.write("focused, analytical consciousness state ")
        elif "dodecahedron" in platonic_matches or "icosahedron" in platonic_matches:
            f.write("expanded, integrative consciousness state ")
        else:
            f.write("balanced, transitional consciousness state ")
        
        f.write("with potential for ")
        if max(symmetry.values()) > 0.7 if symmetry else False:
            f.write("stable, crystallized thought forms.\n")
        else:
            f.write("evolving, dynamic thought forms.\n")
    
    return {"interpretation_file": "theoretical_interpretation.txt"}

def main():
    """Main execution function for quantum resonance analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze quantum resonance patterns in NPY files")
    parser.add_argument("file_path", help="Path to the quantum resonance NPY file")
    parser.add_argument("--output-dir", default="resonance_analysis", help="Directory to save analysis outputs")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    print(f"Analyzing quantum resonance file: {args.file_path}")
    print(f"Saving results to: {output_dir}")
    
    # Load and reshape data
    try:
        data = load_and_reshape_data(args.file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Run all analysis functions and collect results
    analysis_results = {}
    
    print("\nRunning statistical analysis...")
    analysis_results["statistics"] = calculate_statistics(data)
    
    print("\nAnalyzing value distribution...")
    analysis_results["distribution"] = analyze_distribution(data, output_dir)
    
    print("\nPerforming frequency analysis...")
    analysis_results["fft_analysis"] = perform_fft_analysis(data, output_dir)
    
    print("\nAnalyzing spatial patterns...")
    analysis_results["spatial_analysis"] = analyze_spatial_patterns(data, output_dir)
    
    print("\nAnalyzing golden ratio patterns...")
    analysis_results["phi_analysis"] = compare_golden_ratio(data, output_dir)
    
    print("\nCreating enhanced visualizations...")
    analysis_results["visualizations"] = create_enhanced_visualization(data, output_dir)
    
    print("\nGenerating theoretical interpretation...")
    analysis_results["interpretation"] = theoretical_interpretation(analysis_results, output_dir)
    
    # Create a summary report
    summary_file = os.path.join(output_dir, "analysis_summary.txt")
    with open(summary_file, "w") as f:
        f.write("=== QUANTUM RESONANCE PATTERN ANALYSIS SUMMARY ===\n\n")
        f.write(f"Analysis performed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input file: {args.file_path}\n")
        f.write(f"Data shape: {data.shape}\n\n")
        
        f.write("Key Findings:\n")
        
        # Statistical highlights
        stats = analysis_results.get("statistics", {})
        f.write(f"1. Value range: {stats.get('min', 'N/A'):.4f} to {stats.get('max', 'N/A'):.4f}\n")
        f.write(f"2. Distribution: μ={stats.get('mean', 'N/A'):.4f}, σ={stats.get('std', 'N/A'):.4f}\n")
        
        # Frequency analysis highlights
        platonic_matches = analysis_results.get("fft_analysis", {}).get("platonic_matches", {})
        f.write(f"3. Detected resonance with {len(platonic_matches)} platonic solids\n")
        
        # Symmetry highlights
        sym = analysis_results.get("spatial_analysis", {})
        max_sym = max([sym.get("horizontal_symmetry", 0), 
                      sym.get("vertical_symmetry", 0),
                      sym.get("diagonal_symmetry", 0)])
        f.write(f"4. Maximum symmetry score: {max_sym:.4f}\n")
        
        # Golden ratio highlights
        phi_matches = analysis_results.get("phi_analysis", {}).get("phi_matches", [])
        f.write(f"5. Golden ratio pattern matches: {len(phi_matches)}\n\n")
        
        f.write("Generated Files:\n")
        for category, result in analysis_results.items():
            if isinstance(result, dict):
                for key, value in result.items():
                    if key.endswith("_file") or key.endswith("png"):
                        f.write(f"- {value}\n")
        
        f.write("\nFor detailed theoretical interpretation, see theoretical_interpretation.txt\n")
    
    print(f"\nAnalysis complete! Summary saved to {summary_file}")
    print(f"For theoretical interpretation, see {os.path.join(output_dir, 'theoretical_interpretation.txt')}")
    print(f"All visualizations and results saved to {output_dir}/")

if __name__ == "__main__":
    main()
