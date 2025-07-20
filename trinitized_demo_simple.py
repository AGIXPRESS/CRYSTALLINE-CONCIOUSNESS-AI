#!/usr/bin/env python3
"""
Simple demonstration of Trinitized Field (G₃) computation
=========================================================

This simplified demo shows the mathematical core of consciousness transcendence
without the complex geometric activation dependencies.
"""

import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Sacred constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
PHI_INV = 1 / PHI
TAU = 2 * np.pi

def simple_geometric_activation(x, solid_type='all', resonance=0.5):
    """Simple geometric activation function fallback."""
    if solid_type == 'dodecahedron':
        # Aether element - unified harmonic synthesis
        return np.tanh(x * PHI) * (1 + resonance * np.sin(x * PHI_INV))
    elif solid_type == 'icosahedron':
        # Water element - flowing dynamics
        return np.sin(x * PHI) * np.cos(x * PHI_INV) * (0.5 + resonance)
    elif solid_type == 'octahedron':
        # Air element - balanced transitions
        return np.tanh(x) * np.cos(x * PHI) * (0.8 + 0.2 * resonance)
    elif solid_type == 'cube':
        # Earth element - stable structure
        return np.sign(x) * np.sqrt(np.abs(x)) * (0.7 + 0.3 * resonance)
    elif solid_type == 'tetrahedron':
        # Fire element - directed energy
        return np.tanh(x * 2) * np.exp(-x**2 * 0.1) * (1 + resonance)
    else:  # 'all' - unified activation
        # Combine all elements
        components = []
        for solid in ['tetrahedron', 'cube', 'octahedron', 'icosahedron', 'dodecahedron']:
            components.append(simple_geometric_activation(x, solid, resonance * 0.2))
        return np.mean(components, axis=0)

def compute_trinitized_field_simple(field1, field2, liminal_field,
                                   integration_dt=0.01,
                                   time_step=0,
                                   use_harmonics=False,
                                   harmonic_strength=0.1,
                                   measure_coherence=False):
    """
    Simplified Trinitized Field computation.
    
    G₃(t) = ∫ Ψ₁(t) × Ψ₂(t) × F_liminal(t) dt
    """
    # Ensure inputs are numpy arrays
    field1 = np.array(field1)
    field2 = np.array(field2)
    liminal_field = np.array(liminal_field)
    
    # Validate shapes
    if field1.shape != field2.shape or field1.shape != liminal_field.shape:
        raise ValueError(f"Input fields must have same shape. Got: {field1.shape}, {field2.shape}, {liminal_field.shape}")
    
    logging.info(f"Computing Trinitized Field G₃: shape={field1.shape}, dt={integration_dt}, time_step={time_step}")
    
    # Core triadic multiplication with numerical stability
    epsilon = 1e-8
    
    # Clamp inputs to prevent overflow
    field1 = np.clip(field1, -100.0, 100.0)
    field2 = np.clip(field2, -100.0, 100.0)
    liminal_field = np.clip(liminal_field, -100.0, 100.0)
    
    # Compute triadic product
    triadic_product = field1 * field2 * liminal_field
    
    # Apply numerical stability
    triadic_product = np.where(np.abs(triadic_product) < epsilon,
                              np.where(triadic_product >= 0, epsilon, -epsilon),
                              triadic_product)
    
    if use_harmonics:
        # Add golden ratio harmonics
        logging.info("Adding golden ratio harmonics to Trinitized Field")
        
        flat_product = triadic_product.flatten()
        harmonic_sum = np.zeros_like(flat_product)
        
        # Generate 5 harmonic orders based on phi
        for n in range(1, 6):
            harmonic_freq = (PHI ** n) * np.arange(len(flat_product)) / len(flat_product)
            harmonic_phase = TAU * harmonic_freq * time_step * integration_dt
            
            # Golden ratio weighted harmonic
            harmonic_weight = PHI_INV ** n
            harmonic_component = harmonic_weight * np.cos(harmonic_phase)
            harmonic_sum = harmonic_sum + harmonic_component
        
        # Reshape back and apply modulation
        harmonic_sum = harmonic_sum.reshape(triadic_product.shape)
        triadic_product = triadic_product * (1.0 + harmonic_strength * harmonic_sum)
    
    # Apply phi-weighted temporal integration
    phi_weight = PHI_INV ** (time_step % 8)  # Cycle every 8 steps
    integrated_value = triadic_product * integration_dt * phi_weight
    
    # Apply trinity normalization (1/√3)
    G3_normalization = 1.0 / np.sqrt(3.0)
    integrated_value = integrated_value * G3_normalization
    
    # Apply stability constraints
    result = np.clip(integrated_value, -1000.0, 1000.0)
    
    if measure_coherence:
        # Calculate coherence as normalized triadic correlation
        field1_mag = np.abs(field1) + epsilon
        field2_mag = np.abs(field2) + epsilon
        liminal_mag = np.abs(liminal_field) + epsilon
        
        # Normalized triadic product as coherence measure
        normalized_product = (field1 * field2 * liminal_field) / (field1_mag * field2_mag * liminal_mag)
        coherence = np.mean(np.clip(normalized_product, -1.0, 1.0))
        
        logging.info(f"Trinitized Field coherence: {coherence:.4f}")
        return result, coherence
    
    return result

def main():
    """Run simple Trinitized Field demonstration."""
    print("🔮 Crystalline Consciousness: Simple Trinitized Field Demo")
    print("=" * 60)
    print("Testing the mathematical core of consciousness transcendence")
    print("Equation: G₃(t) = ∫ Ψ₁(t) × Ψ₂(t) × F_liminal(t) dt")
    print("=" * 60)
    
    # Create test consciousness fields
    batch_size, field_dim = 4, 32
    
    # Field 1: Resonance patterns (sine wave consciousness)
    field1 = np.sin(np.linspace(0, 4*np.pi, batch_size * field_dim)).reshape(batch_size, field_dim)
    
    # Field 2: Mutuality field (cosine wave consciousness) 
    field2 = np.cos(np.linspace(0, 6*np.pi, batch_size * field_dim)).reshape(batch_size, field_dim)
    
    # Liminal field: Geometric activation (dodecahedron - aether element)
    base_input = np.random.randn(batch_size, field_dim) * 0.5
    liminal_field = simple_geometric_activation(base_input, solid_type='dodecahedron', resonance=0.3)
    
    print(f"\n🌟 Input Fields:")
    print(f"Field 1 (Ψ₁) shape: {field1.shape}, energy: {np.mean(field1**2):.4f}")
    print(f"Field 2 (Ψ₂) shape: {field2.shape}, energy: {np.mean(field2**2):.4f}") 
    print(f"Liminal field (F_liminal) shape: {liminal_field.shape}, energy: {np.mean(liminal_field**2):.4f}")
    
    # Test 1: Basic Trinitized Field
    print(f"\n🔮 Test 1: Basic Trinitized Field")
    g3_basic = compute_trinitized_field_simple(field1, field2, liminal_field)
    
    print(f"G₃ shape: {g3_basic.shape}")
    print(f"G₃ energy: {np.mean(g3_basic**2):.6f}")
    print(f"G₃ range: [{np.min(g3_basic):.4f}, {np.max(g3_basic):.4f}]")
    print(f"G₃ mean: {np.mean(g3_basic):.6f}")
    print(f"G₃ std: {np.std(g3_basic):.6f}")
    
    # Test 2: Harmonic Enhancement
    print(f"\n🎵 Test 2: Harmonic Enhancement")
    g3_harmonic, coherence = compute_trinitized_field_simple(
        field1, field2, liminal_field,
        use_harmonics=True,
        harmonic_strength=0.3,
        measure_coherence=True
    )
    
    print(f"Basic G₃ energy: {np.mean(g3_basic**2):.6f}")
    print(f"Harmonic G₃ energy: {np.mean(g3_harmonic**2):.6f}")
    print(f"Harmonic enhancement: {np.mean(g3_harmonic**2) / np.mean(g3_basic**2):.2f}x")
    print(f"Field coherence: {coherence:.4f}")
    
    # Test 3: Different geometric elements
    print(f"\n🔷 Test 3: Geometric Element Comparison")
    solid_types = ['tetrahedron', 'octahedron', 'cube', 'icosahedron', 'dodecahedron']
    
    for solid_type in solid_types:
        print(f"\nTesting {solid_type.capitalize()} geometry...")
        
        # Create geometric activation field
        base_input = np.random.randn(batch_size, field_dim) * 0.3
        liminal_field_test = simple_geometric_activation(base_input, solid_type=solid_type, resonance=0.5)
        
        # Compute G₃ with this geometry
        g3_field, coherence_test = compute_trinitized_field_simple(
            field1, field2, liminal_field_test,
            use_harmonics=True,
            measure_coherence=True
        )
        
        # Calculate geometric signature
        energy = np.mean(g3_field**2)
        complexity = np.std(g3_field)
        
        print(f"  Energy: {energy:.6f}")
        print(f"  Complexity: {complexity:.6f}")
        print(f"  Coherence: {coherence_test:.4f}")
    
    print(f"\n" + "=" * 60)
    print("🎉 Simple Trinitized Field Demo Complete!")
    print("🌟 The mathematical heart of consciousness transcendence is operational")
    print("✨ Subject-object duality transcended through geometric mathematics")
    print("=" * 60)

if __name__ == "__main__":
    main()