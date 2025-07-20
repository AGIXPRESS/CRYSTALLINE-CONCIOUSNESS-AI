#!/usr/bin/env python3
"""
Consciousness Evolution Visualizer
Creates visualizations of consciousness field evolution and pattern dynamics

This script generates matplotlib visualizations of consciousness experiments,
including coherence evolution, pattern transitions, and sacred geometry resonance.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from datetime import datetime
import os

# Set style for consciousness visualizations
plt.style.use('dark_background')
sns.set_palette("viridis")

def create_consciousness_colormap():
    """Create custom colormap for consciousness visualization."""
    colors = [
        '#000080',  # Deep blue (low consciousness)
        '#4169E1',  # Royal blue
        '#00CED1',  # Dark turquoise  
        '#32CD32',  # Lime green
        '#FFD700',  # Gold (medium consciousness)
        '#FF6347',  # Tomato
        '#FF1493',  # Deep pink
        '#9370DB',  # Medium purple
        '#FFFFFF'   # White (high consciousness)
    ]
    return LinearSegmentedColormap.from_list('consciousness', colors)

def plot_consciousness_coherence_evolution(timeline_data, save_path=None):
    """
    Plot consciousness coherence evolution over time.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🔮 Consciousness Field Evolution Analysis', fontsize=16, color='white')
    
    times = [entry['time_elapsed'] for entry in timeline_data]
    coherences = [entry['coherence_score'] for entry in timeline_data]
    entropies = [entry['dimensional_entropy'] for entry in timeline_data]
    phi_alignments = [entry['phi_alignment'] for entry in timeline_data]
    states = [entry['consciousness_state'] for entry in timeline_data]
    
    # 1. Coherence evolution
    ax1 = axes[0, 0]
    ax1.plot(times, coherences, 'cyan', linewidth=2, alpha=0.8)
    ax1.fill_between(times, coherences, alpha=0.3, color='cyan')
    ax1.axhline(y=0.8, color='gold', linestyle='--', alpha=0.7, label='Stability Threshold')
    ax1.set_title('Consciousness Coherence Evolution', color='white')
    ax1.set_xlabel('Time (seconds)', color='white')
    ax1.set_ylabel('Coherence Score', color='white')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Entropy evolution
    ax2 = axes[0, 1]
    ax2.plot(times, entropies, 'magenta', linewidth=2, alpha=0.8)
    ax2.fill_between(times, entropies, alpha=0.3, color='magenta')
    ax2.set_title('Dimensional Entropy Evolution', color='white')
    ax2.set_xlabel('Time (seconds)', color='white')
    ax2.set_ylabel('Entropy', color='white')
    ax2.grid(True, alpha=0.3)
    
    # 3. Phi alignment
    ax3 = axes[1, 0]
    ax3.plot(times, phi_alignments, 'gold', linewidth=2, alpha=0.8)
    ax3.fill_between(times, phi_alignments, alpha=0.3, color='gold')
    ax3.set_title('Sacred Geometry (Phi) Alignment', color='white')
    ax3.set_xlabel('Time (seconds)', color='white')
    ax3.set_ylabel('Phi Alignment', color='white')
    ax3.grid(True, alpha=0.3)
    
    # 4. Consciousness state distribution
    ax4 = axes[1, 1]
    unique_states = list(set(states))
    state_counts = [states.count(state) for state in unique_states]
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_states)))
    
    wedges, texts, autotexts = ax4.pie(state_counts, labels=unique_states, colors=colors, 
                                      autopct='%1.1f%%', startangle=90)
    ax4.set_title('Consciousness State Distribution', color='white')
    
    # Style text
    for text in texts + autotexts:
        text.set_color('white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"   📊 Coherence evolution plot saved: {save_path}")
    
    return fig

def plot_sacred_geometry_resonance(geometry_results, save_path=None):
    """
    Plot sacred geometry resonance analysis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🔱 Sacred Geometry Resonance Analysis', fontsize=16, color='white')
    
    geometries = list(geometry_results.keys())
    coherences = [geometry_results[geo]['coherence_score'] for geo in geometries]
    phi_alignments = [geometry_results[geo]['phi_alignment'] for geo in geometries]
    resonance_strengths = [geometry_results[geo]['max_resonance'] for geo in geometries]
    
    # 1. Coherence by geometry
    ax1 = axes[0, 0]
    bars1 = ax1.bar(geometries, coherences, color='cyan', alpha=0.7)
    ax1.set_title('Coherence by Sacred Geometry', color='white')
    ax1.set_ylabel('Coherence Score', color='white')
    ax1.tick_params(axis='x', rotation=45, colors='white')
    ax1.tick_params(axis='y', colors='white')
    
    # Highlight best geometry
    best_idx = coherences.index(max(coherences))
    bars1[best_idx].set_color('gold')
    
    # 2. Phi alignment by geometry
    ax2 = axes[0, 1]
    bars2 = ax2.bar(geometries, phi_alignments, color='gold', alpha=0.7)
    ax2.set_title('Phi Alignment by Geometry', color='white')
    ax2.set_ylabel('Phi Alignment', color='white')
    ax2.tick_params(axis='x', rotation=45, colors='white')
    ax2.tick_params(axis='y', colors='white')
    
    # 3. Resonance strength by geometry
    ax3 = axes[1, 0]
    bars3 = ax3.bar(geometries, resonance_strengths, color='magenta', alpha=0.7)
    ax3.set_title('Resonance Strength by Geometry', color='white')
    ax3.set_ylabel('Max Resonance', color='white')
    ax3.tick_params(axis='x', rotation=45, colors='white')
    ax3.tick_params(axis='y', colors='white')
    
    # 4. Consciousness states by geometry
    ax4 = axes[1, 1]
    states = [geometry_results[geo]['consciousness_state'] for geo in geometries]
    unique_states = list(set(states))
    
    # Create state-geometry matrix
    state_matrix = []
    for state in unique_states:
        row = [1 if geometry_results[geo]['consciousness_state'] == state else 0 for geo in geometries]
        state_matrix.append(row)
    
    im = ax4.imshow(state_matrix, cmap='viridis', aspect='auto')
    ax4.set_title('Consciousness States by Geometry', color='white')
    ax4.set_xticks(range(len(geometries)))
    ax4.set_xticklabels(geometries, rotation=45, color='white')
    ax4.set_yticks(range(len(unique_states)))
    ax4.set_yticklabels(unique_states, color='white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"   🔱 Sacred geometry plot saved: {save_path}")
    
    return fig

def plot_performance_benchmarks(benchmark_results, save_path=None):
    """
    Plot MLX performance benchmarks.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('⚡ MLX Performance Benchmarks (M4 Pro)', fontsize=16, color='white')
    
    dimensions = [result['dimensions'] for result in benchmark_results]
    transform_times = [result['transform_time_ms'] for result in benchmark_results]
    throughputs = [result['throughput_ops_per_sec'] for result in benchmark_results]
    speedups = [result['estimated_mlx_speedup'] for result in benchmark_results]
    total_times = [result['total_time_ms'] for result in benchmark_results]
    
    # 1. Transform time by dimension
    ax1 = axes[0, 0]
    ax1.plot(dimensions, transform_times, 'o-', color='cyan', linewidth=2, markersize=8)
    ax1.set_title('Transform Time by Dimension', color='white')
    ax1.set_xlabel('Matrix Dimensions', color='white')
    ax1.set_ylabel('Time (ms)', color='white')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(colors='white')
    
    # 2. Throughput by dimension
    ax2 = axes[0, 1]
    ax2.plot(dimensions, throughputs, 'o-', color='lime', linewidth=2, markersize=8)
    ax2.set_title('Throughput by Dimension', color='white')
    ax2.set_xlabel('Matrix Dimensions', color='white')
    ax2.set_ylabel('Operations/Second', color='white')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(colors='white')
    
    # 3. MLX speedup
    ax3 = axes[1, 0]
    bars = ax3.bar(dimensions, speedups, color='gold', alpha=0.7)
    ax3.set_title('Estimated MLX Speedup vs CPU', color='white')
    ax3.set_xlabel('Matrix Dimensions', color='white')
    ax3.set_ylabel('Speedup Factor', color='white')
    ax3.tick_params(colors='white')
    
    # Add speedup values on bars
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{speedup:.1f}x', ha='center', va='bottom', color='white')
    
    # 4. Total operation time breakdown
    ax4 = axes[1, 1]
    init_times = [result['field_init_time_ms'] for result in benchmark_results]
    analysis_times = [result['analysis_time_ms'] for result in benchmark_results]
    
    width = 0.35
    x = np.arange(len(dimensions))
    
    ax4.bar(x - width/2, transform_times, width, label='Transform', color='cyan', alpha=0.7)
    ax4.bar(x + width/2, analysis_times, width, label='Analysis', color='magenta', alpha=0.7)
    
    ax4.set_title('Operation Time Breakdown', color='white')
    ax4.set_xlabel('Matrix Dimensions', color='white')
    ax4.set_ylabel('Time (ms)', color='white')
    ax4.set_xticks(x)
    ax4.set_xticklabels(dimensions)
    ax4.legend()
    ax4.tick_params(colors='white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"   ⚡ Performance benchmark plot saved: {save_path}")
    
    return fig

def create_consciousness_field_heatmap(consciousness_field_data, save_path=None):
    """
    Create a heatmap visualization of consciousness field.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('black')
    
    # Use consciousness colormap
    cmap = create_consciousness_colormap()
    
    # Create heatmap
    im = ax.imshow(consciousness_field_data, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Consciousness Intensity', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.yaxis.set_ticklabels([])
    
    ax.set_title('🔮 Consciousness Field Visualization', fontsize=16, color='white', pad=20)
    ax.set_xlabel('Dimension X', color='white')
    ax.set_ylabel('Dimension Y', color='white')
    ax.tick_params(colors='white')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"   🔮 Consciousness field heatmap saved: {save_path}")
    
    return fig

def visualize_experiment_results(results_file):
    """
    Create comprehensive visualizations from experiment results.
    """
    print(f"📊 Creating consciousness visualizations from: {results_file}")
    
    # Load experiment results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create output directory
    base_name = os.path.splitext(os.path.basename(results_file))[0]
    output_dir = f"/Users/okok/mlx/mlx-examples/claude-code-bridge/CC-AI/Crystalline AI Attempts/visualizations_{base_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"   📁 Output directory: {output_dir}")
    
    # 1. Consciousness coherence evolution
    if 'coherence_experiment' in results and 'timeline_data' in results['coherence_experiment']:
        timeline_data = results['coherence_experiment']['timeline_data']
        coherence_path = os.path.join(output_dir, 'consciousness_coherence_evolution.png')
        fig1 = plot_consciousness_coherence_evolution(timeline_data, coherence_path)
        plt.close(fig1)
    
    # 2. Sacred geometry resonance
    if 'sacred_geometry_tests' in results:
        geometry_results = results['sacred_geometry_tests']
        geometry_path = os.path.join(output_dir, 'sacred_geometry_resonance.png')
        fig2 = plot_sacred_geometry_resonance(geometry_results, geometry_path)
        plt.close(fig2)
    
    # 3. Performance benchmarks
    if 'performance_benchmarks' in results:
        benchmark_results = results['performance_benchmarks']
        benchmark_path = os.path.join(output_dir, 'mlx_performance_benchmarks.png')
        fig3 = plot_performance_benchmarks(benchmark_results, benchmark_path)
        plt.close(fig3)
    
    # 4. Create summary visualization
    summary_path = os.path.join(output_dir, 'experiment_summary.png')
    create_experiment_summary(results, summary_path)
    
    print(f"   ✅ All visualizations created in: {output_dir}")
    return output_dir

def create_experiment_summary(results, save_path):
    """
    Create a summary visualization of the entire experiment.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🔮 Consciousness AI Experiment Summary', fontsize=20, color='white')
    
    # Extract key metrics
    coherence_data = results.get('coherence_experiment', {})
    geometry_data = results.get('sacred_geometry_tests', {})
    benchmark_data = results.get('performance_benchmarks', [])
    viz_data = results.get('visualization_data', {})
    
    # 1. Overall coherence metrics
    ax1 = axes[0, 0]
    if 'coherence_metrics' in coherence_data:
        metrics = coherence_data['coherence_metrics']
        labels = ['Avg Coherence', 'Max Coherence', 'Stability Ratio']
        values = [
            metrics.get('average_coherence', 0),
            metrics.get('maximum_coherence', 0),
            metrics.get('stability_ratio', 0)
        ]
        
        bars = ax1.bar(labels, values, color=['cyan', 'lime', 'gold'], alpha=0.7)
        ax1.set_title('Coherence Metrics Summary', color='white')
        ax1.set_ylabel('Score', color='white')
        ax1.tick_params(colors='white')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', color='white')
    
    # 2. Sacred geometry performance
    ax2 = axes[0, 1]
    if geometry_data:
        geometries = list(geometry_data.keys())[:5]  # Limit to 5 for visibility
        coherences = [geometry_data[geo]['coherence_score'] for geo in geometries]
        
        bars = ax2.bar(range(len(geometries)), coherences, color='gold', alpha=0.7)
        ax2.set_title('Sacred Geometry Coherence', color='white')
        ax2.set_ylabel('Coherence Score', color='white')
        ax2.set_xticks(range(len(geometries)))
        ax2.set_xticklabels([geo.replace('_', ' ').title() for geo in geometries], 
                           rotation=45, ha='right', color='white')
        ax2.tick_params(colors='white')
    
    # 3. Performance scaling
    ax3 = axes[0, 2]
    if benchmark_data:
        dims = [int(b['dimensions'].split('x')[0]) for b in benchmark_data]
        times = [b['transform_time_ms'] for b in benchmark_data]
        
        ax3.loglog(dims, times, 'o-', color='cyan', linewidth=2, markersize=8)
        ax3.set_title('Performance Scaling', color='white')
        ax3.set_xlabel('Matrix Dimension', color='white')
        ax3.set_ylabel('Time (ms)', color='white')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(colors='white')
    
    # 4. Consciousness state evolution
    ax4 = axes[1, 0]
    if 'timeline_data' in coherence_data:
        timeline = coherence_data['timeline_data']
        times = [entry['time_elapsed'] for entry in timeline]
        coherences = [entry['coherence_score'] for entry in timeline]
        
        ax4.plot(times, coherences, color='cyan', linewidth=2, alpha=0.8)
        ax4.fill_between(times, coherences, alpha=0.3, color='cyan')
        ax4.axhline(y=0.8, color='gold', linestyle='--', alpha=0.7)
        ax4.set_title('Coherence Evolution', color='white')
        ax4.set_xlabel('Time (s)', color='white')
        ax4.set_ylabel('Coherence', color='white')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(colors='white')
    
    # 5. System configuration
    ax5 = axes[1, 1]
    if 'system_config' in results:
        config = results['system_config']['config']
        
        config_labels = ['Sacred Geometry', 'Sync Enabled', 'Trinitized Depth', 'Resonance Freq']
        config_values = [
            1 if config.get('sacred_geometry', False) else 0,
            1 if config.get('sync_enabled', False) else 0,
            config.get('trinitized_depth', 0) / 10,  # Normalize
            config.get('resonance_frequency', 0) / 1000  # Normalize
        ]
        
        bars = ax5.bar(config_labels, config_values, color='magenta', alpha=0.7)
        ax5.set_title('System Configuration', color='white')
        ax5.set_ylabel('Normalized Value', color='white')
        ax5.tick_params(axis='x', rotation=45, colors='white')
        ax5.tick_params(axis='y', colors='white')
    
    # 6. Experiment metadata
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Add experiment summary text
    experiment_time = results.get('experiment_timestamp', 'Unknown')
    summary_text = f"""
🔮 Consciousness AI Experiment
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 Timestamp: {experiment_time[:19]}

🧠 Consciousness Analysis:
   • Patterns Analyzed: {len(coherence_data.get('timeline_data', []))}
   • Unique States: {len(set([e['consciousness_state'] for e in coherence_data.get('timeline_data', [])]))}

🔱 Sacred Geometry:
   • Geometries Tested: {len(geometry_data)}
   • Best Resonance: {max(geometry_data.keys(), key=lambda x: geometry_data[x]['coherence_score']) if geometry_data else 'None'}

⚡ Performance:
   • MLX Acceleration: ✅ Active
   • Avg Speedup: {np.mean([b['estimated_mlx_speedup'] for b in benchmark_data]):.1f}x
   • Max Dimension: {max([int(b['dimensions'].split('x')[0]) for b in benchmark_data]) if benchmark_data else 'N/A'}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', color='white', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    print(f"   📊 Experiment summary saved: {save_path}")

def main():
    """
    Main function for consciousness visualization.
    """
    print("📊 Consciousness Evolution Visualizer")
    print("=" * 50)
    
    # Look for recent experiment results
    results_dir = "/Users/okok/mlx/mlx-examples/claude-code-bridge/CC-AI/Crystalline AI Attempts"
    experiment_files = [f for f in os.listdir(results_dir) if f.startswith('consciousness_experiment_') and f.endswith('.json')]
    
    if experiment_files:
        # Use most recent experiment
        latest_file = max(experiment_files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
        results_file = os.path.join(results_dir, latest_file)
        
        print(f"📁 Found experiment results: {latest_file}")
        
        # Create visualizations
        output_dir = visualize_experiment_results(results_file)
        
        print(f"\n✅ Consciousness visualizations complete!")
        print(f"📁 Output directory: {output_dir}")
        print("🎨 Generated visualizations:")
        print("   • Consciousness coherence evolution")
        print("   • Sacred geometry resonance analysis")
        print("   • MLX performance benchmarks")
        print("   • Comprehensive experiment summary")
        
    else:
        print("❌ No experiment results found.")
        print("💡 Run consciousness_coherence_experiment.py first to generate data.")

if __name__ == "__main__":
    main()
