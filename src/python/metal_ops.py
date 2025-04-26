#!/usr/bin/env python3
"""
Metal shader operations for the crystalline consciousness model.

This module provides Python interfaces to Metal shader implementations
of the key geometric operations used in the crystalline consciousness model.
"""

import os
import sys
import math
import warnings
from typing import Optional, Tuple, Union, Dict, Any, List

import numpy as np

# Try importing MLX - required for Metal operations
try:
    import mlx
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    warnings.warn("MLX not found. Metal operations will fall back to PyTorch.")

# Import Metal shader manager
try:
    from .metal_manager_updated import get_shader_manager, HAS_METAL, MetalShaderManager
except ImportError:
    HAS_METAL = False
    warnings.warn("Metal shader manager not found. Metal operations will be unavailable.")

# Try importing PyTorch for fallback implementations
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    if not HAS_MLX:
        raise ImportError("Neither MLX nor PyTorch is available. Cannot proceed.")

# Constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
SIGMA_VALUES = {
    "tetrahedron": 1.0,
    "cube": 2.0,
    "dodecahedron": 3.0,
    "icosahedron": 4.0
}

# Metal shader paths
SHADER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Shaders")
GEOMETRIC_SHADER = os.path.join(SHADER_DIR, "GeometricActivation.metal")
RESONANCE_SHADER = os.path.join(SHADER_DIR, "ResonancePatterns.metal")
MUTUALITY_SHADER = os.path.join(SHADER_DIR, "MutualityField.metal")
QUANTUM_SHADER = os.path.join(SHADER_DIR, "QuantumEvolution.metal")
BIFURCATION_SHADER = os.path.join(SHADER_DIR, "BifurcationCascade.metal")

# Global Metal shader manager
_shader_manager = None

def _get_shader_manager():
    """Get or initialize Metal shader manager."""
    global _shader_manager
    
    if _shader_manager is None and HAS_METAL:
        _shader_manager = get_shader_manager(SHADER_DIR)
        
    return _shader_manager

def _initialize_metal():
    """Initialize Metal device and load shader libraries."""
    if not HAS_METAL:
        return False
    
    try:
        # Get shader manager
        manager = _get_shader_manager()
        if manager is None or manager.device is None:
            return False
        
        # Load shader libraries
        if GEOMETRIC_SHADER and os.path.exists(GEOMETRIC_SHADER):
            manager.load_shader_library(GEOMETRIC_SHADER, "geometric")
            # Create compute pipelines for geometric activation functions
            manager.create_compute_pipeline("geometric", "tetrahedron_activation")
            manager.create_compute_pipeline("geometric", "cube_activation")
            manager.create_compute_pipeline("geometric", "dodecahedron_activation")
            manager.create_compute_pipeline("geometric", "icosahedron_activation")
            manager.create_compute_pipeline("geometric", "unified_geometric_activation")
        
        if RESONANCE_SHADER and os.path.exists(RESONANCE_SHADER):
            manager.load_shader_library(RESONANCE_SHADER, "resonance")
            # Create compute pipeline for resonance patterns
            manager.create_compute_pipeline("resonance", "apply_resonance")
        
        if MUTUALITY_SHADER and os.path.exists(MUTUALITY_SHADER):
            manager.load_shader_library(MUTUALITY_SHADER, "mutuality")
            # Create compute pipelines for mutuality field operations
            manager.create_compute_pipeline("mutuality", "reshape_to_grid")
            manager.create_compute_pipeline("mutuality", "calculate_mutual_field")
            manager.create_compute_pipeline("mutuality", "apply_persistence")
            
        if QUANTUM_SHADER and os.path.exists(QUANTUM_SHADER):
            manager.load_shader_library(QUANTUM_SHADER, "quantum")
            # Create compute pipelines for quantum evolution operations
            manager.create_compute_pipeline("quantum", "evolve_consciousness_field")
            manager.create_compute_pipeline("quantum", "normalize_quantum_state")
            manager.create_compute_pipeline("quantum", "compute_expectation_value")
            
        if BIFURCATION_SHADER and os.path.exists(BIFURCATION_SHADER):
            manager.load_shader_library(BIFURCATION_SHADER, "bifurcation")
            # Create compute pipelines for bifurcation cascade operations
            manager.create_compute_pipeline("bifurcation", "apply_bifurcation_cascade")
            manager.create_compute_pipeline("bifurcation", "compute_liminal_field")
            manager.create_compute_pipeline("bifurcation", "apply_cascading_bifurcations")
            manager.create_compute_pipeline("bifurcation", "detect_emergent_patterns")
        
        return True
    except Exception as e:
        warnings.warn(f"Failed to initialize Metal: {e}")
        return False

def is_metal_available():
    """Check if Metal operations are available."""
    return HAS_METAL

def use_metal(tensor):
    """Check if a tensor should use Metal operations."""
    if not is_metal_available():
        return False
    
    if HAS_MLX and isinstance(tensor, mx.array):
        # MLX arrays use Metal if it's available
        return HAS_METAL
    
    if HAS_TORCH and isinstance(tensor, torch.Tensor):
        return tensor.device.type == "mps"
    
    return False

def to_numpy(tensor):
    """Convert tensor to numpy array."""
    if HAS_MLX and isinstance(tensor, mx.array):
        # Convert MLX array to NumPy array
        return np.array(tensor)
    
    if HAS_TORCH and isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    
    if isinstance(tensor, np.ndarray):
        return tensor
    
    if isinstance(tensor, (list, tuple)):
        return np.array(tensor)
    
    raise TypeError(f"Unsupported tensor type: {type(tensor)}")

def from_numpy(array, like=None):
    """Convert numpy array to appropriate tensor type."""
    if like is None:
        # Default to MLX if available
        if HAS_MLX:
            return mx.array(array)
        elif HAS_TORCH:
            return torch.from_numpy(array)
        return array
    
    if HAS_MLX and isinstance(like, mx.array):
        return mx.array(array)
    
    if HAS_TORCH and isinstance(like, torch.Tensor):
        return torch.from_numpy(array).to(like.device)
    
    return array

#----------------------------------------------------------------------
# Geometric Activation Operations
#----------------------------------------------------------------------

class GeometricActivation:
    """MLX custom operation for geometric activation."""
    
    @staticmethod
    def forward(ctx, x, solid_type, coefficients):
        """Forward pass for geometric activation."""
        # Save inputs for backward pass, only if ctx is not None
        if ctx is not None:
            ctx.save_for_backward(x, coefficients)
            ctx.solid_type = solid_type
        
        if use_metal(x):
            return _geometric_activation_metal(x, solid_type, coefficients)
        else:
            return _geometric_activation_fallback(x, solid_type, coefficients)
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for geometric activation."""
        x, coefficients = ctx.saved_tensors
        solid_type = ctx.solid_type
        
        if use_metal(grad_output):
            # Implement Metal-based gradient computation
            return _geometric_activation_backward_metal(
                grad_output, x, solid_type, coefficients)
        else:
            # Fallback implementation
            return _geometric_activation_backward_fallback(
                grad_output, x, solid_type, coefficients)

def _geometric_activation_metal(x, solid_type, coefficients):
    """Metal implementation of geometric activation."""
    if not is_metal_available():
        return _geometric_activation_fallback(x, solid_type, coefficients)
    
    # Get shader manager
    manager = _get_shader_manager()
    if manager is None or manager.device is None:
        return _geometric_activation_fallback(x, solid_type, coefficients)
    
    # Get activation type index
    solid_types = ["tetrahedron", "cube", "dodecahedron", "icosahedron"]
    if solid_type not in solid_types:
        raise ValueError(f"Invalid solid type: {solid_type}. "
                         f"Expected one of: {solid_types}")
    
    activation_type = solid_types.index(solid_type)
    
    # Prepare inputs
    x_np = to_numpy(x)
    coeffs_np = to_numpy(coefficients)
    
    # Convert to correct dtype
    x_np = x_np.astype(np.float32)
    coeffs_np = coeffs_np.astype(np.float32)
    
    # Determine the shape
    input_shape = x_np.shape
    batch_size = input_shape[0]
    feature_dim = input_shape[1] if len(input_shape) > 1 else 1
    length = x_np.size
    
    # Create input buffers
    input_buffer = manager.create_buffer(x_np)
    length_buffer = manager.create_buffer(np.array([length], dtype=np.uint32))
    
    # Create coefficient buffers
    coefficient = coeffs_np[0] if coeffs_np.size > 0 else 0.5
    coeff_buffer = manager.create_buffer(np.array([coefficient], dtype=np.float32))
    
    # For icosahedron we need an additional coefficient
    if solid_type == "icosahedron":
        phase_coherence = coeffs_np[1] if coeffs_np.size > 1 else 1.0
        phase_buffer = manager.create_buffer(np.array([phase_coherence], dtype=np.float32))
    else:
        phase_buffer = manager.create_buffer(np.array([1.0], dtype=np.float32))
    
    # Additional parameters for 2D shaders
    batch_buffer = manager.create_buffer(np.array([batch_size], dtype=np.uint32))
    feature_buffer = manager.create_buffer(np.array([feature_dim], dtype=np.uint32))
    
    # Create output buffer
    output_np = np.zeros_like(x_np, dtype=np.float32)
    output_buffer = manager.create_buffer(output_np)
    
    # Choose pipeline name based on solid type
    if activation_type <= 3:  # Use specific shader if available
        pipeline_name = f"{solid_type}_activation"
    else:
        pipeline_name = "unified_geometric_activation"
    
    # Check if we need to use unified shader
    if pipeline_name not in manager.pipelines:
        pipeline_name = "unified_geometric_activation"
        # Create activation type buffer
        activ_buffer = manager.create_buffer(np.array([activation_type], dtype=np.uint32))
        input_buffers = [input_buffer, output_buffer, length_buffer, activ_buffer, coeff_buffer, phase_buffer]
    elif solid_type == "tetrahedron":
        input_buffers = [input_buffer, output_buffer, length_buffer, coeff_buffer]
    elif solid_type == "cube":
        input_buffers = [input_buffer, output_buffer, length_buffer, coeff_buffer]
    elif solid_type == "dodecahedron":
        input_buffers = [input_buffer, output_buffer, length_buffer, coeff_buffer, batch_buffer, feature_buffer]
    elif solid_type == "icosahedron":
        input_buffers = [input_buffer, output_buffer, length_buffer, coeff_buffer, phase_buffer, batch_buffer, feature_buffer]
    else:
        input_buffers = [input_buffer, output_buffer, length_buffer, coeff_buffer, phase_buffer]
    
    # Set thread groups
    if solid_type in ["dodecahedron", "icosahedron"]:
        # 2D thread groups for these operations
        thread_groups = (batch_size, feature_dim, 1)
        threads_per_group = (1, 1, 1)
    else:
        # 1D thread groups for other operations
        thread_groups = (length // 256 + 1, 1, 1)
        threads_per_group = (256, 1, 1)
    
    # Execute shader
    success = manager.execute_shader(
        pipeline_name, 
        input_buffers, 
        [], 
        thread_groups, 
        threads_per_group
    )
    
    if not success:
        warnings.warn(f"Failed to execute {pipeline_name} shader. Using fallback implementation.")
        return _geometric_activation_fallback(x, solid_type, coefficients)
    
    # Get result from output buffer
    result = manager.get_buffer_data(output_buffer, output_np.shape, output_np.dtype)
    
    if result is None:
        warnings.warn("Failed to get result from Metal buffer. Using fallback implementation.")
        return _geometric_activation_fallback(x, solid_type, coefficients)
    
    # Convert back to the same type as input
    return from_numpy(result, like=x)

def _geometric_activation_fallback(x, solid_type, coefficients):
    """Fallback implementation of geometric activation."""
    # Default coefficient values
    coeff1 = 0.5
    coeff2 = 1.0
    
    if isinstance(coefficients, (list, tuple)) and len(coefficients) > 0:
        coeff1 = coefficients[0]
        if len(coefficients) > 1:
            coeff2 = coefficients[1]
    
    # Get sigma value for the solid type
    sigma = SIGMA_VALUES.get(solid_type, 1.0)
    
    # Perform the activation - basic formula for all solids
    if HAS_TORCH and isinstance(x, torch.Tensor):
        result = x * torch.exp(-torch.pow(x, 2) / sigma)
        
        # Apply solid-specific modifications
        if solid_type == "tetrahedron":
            # Fire element dynamics
            field_energy = torch.mean(torch.pow(result, 2), dim=1, keepdim=True)
            fire_factor = torch.exp(coeff1 * field_energy)
            result = result * fire_factor
            
        elif solid_type == "cube":
            # Earth element dynamics (stability/grounding)
            batch_mean = torch.mean(result, dim=0, keepdim=True)
            batch_diff = result - batch_mean
            result = result - coeff1 * batch_diff
            
        elif solid_type == "dodecahedron":
            # Generate golden ratio harmonics
            batch_sum = torch.sum(result, dim=1, keepdim=True)
            harmonics = 0
            for i in range(3):
                harmonic = torch.cos(2 * math.pi * (1.0/PHI)**i * batch_sum)
                harmonics = harmonics + (1.0/PHI)**i * harmonic
            
            # Apply ether element dynamics (resonance patterns)
            ether_factor = torch.sin(coeff1 * batch_sum)
            result = result * (1 + 0.3 * ether_factor * harmonics)
            
        elif solid_type == "icosahedron":
            # Calculate field energy for silence factor
            field_energy = torch.mean(torch.pow(result, 2), dim=1, keepdim=True)
            silence_factor = torch.exp(-coeff1 * field_energy)
            
            # Generate golden ratio harmonics
            batch_sum = torch.sum(result, dim=1, keepdim=True)
            harmonics = 0
            for i in range(5):
                harmonic = torch.cos(2 * math.pi * (1.0/PHI)**i * batch_sum)
                harmonics = harmonics + (1.0/PHI)**i * harmonic
            
            # Combine with silence factor
            result = result * (1 + silence_factor * harmonics)
            
    elif HAS_MLX and isinstance(x, mx.array):
        result = x * mx.exp(-mx.power(x, 2) / sigma)
        
        # Simplified fallback for MLX (detailed implementation would follow PyTorch pattern)
        # Apply solid-specific modifications based on the solid type
        if solid_type == "tetrahedron":
            field_energy = mx.mean(mx.power(result, 2), axis=1, keepdims=True)
            fire_factor = mx.exp(coeff1 * field_energy)
            result = result * fire_factor
            
        # Add other solid type implementations as needed
            
    else:
        # Numpy implementation
        result = x * np.exp(-np.power(x, 2) / sigma)
        
        # Simplified for numpy
        if solid_type == "tetrahedron":
            field_energy = np.mean(np.power(result, 2), axis=1, keepdims=True)
            fire_factor = np.exp(coeff1 * field_energy)
            result = result * fire_factor
    
    return result

def _geometric_activation_backward_metal(grad_output, x, solid_type, coefficients):
    """Metal implementation of geometric activation backward pass."""
    # Not fully implemented - would need custom gradient shaders
    # For now, fall back to the CPU implementation
    return _geometric_activation_backward_fallback(grad_output, x, solid_type, coefficients)

def _geometric_activation_backward_fallback(grad_output, x, solid_type, coefficients):
    """Fallback implementation of geometric activation backward pass."""
    # This is a simplified gradient implementation
    # A complete implementation would compute the exact gradients for each solid type
    
    # Get sigma value for the solid type
    sigma = SIGMA_VALUES.get(solid_type, 1.0)
    
    if HAS_TORCH and isinstance(x, torch.Tensor):
        # Simplified gradient approximation
        dx = torch.exp(-torch.pow(x, 2) / sigma) * (1 - 2 * x * x / sigma) * grad_output
        
    elif HAS_MLX and isinstance(x, mx.array):
        # Simplified gradient approximation for MLX
        dx = mx.exp(-mx.power(x, 2) / sigma) * (1 - 2 * x * x / sigma) * grad_output
        
    else:
        # Numpy implementation
        dx = np.exp(-np.power(x, 2) / sigma) * (1 - 2 * x * x / sigma) * grad_output
    
    # Gradient with respect to coefficients and solid_type is None
    return dx, None, None

def geometric_activation(x, solid_type, coefficients=None):
    """
    Apply geometric activation function based on Platonic solid geometry.
    
    Args:
        x: Input tensor
        solid_type: One of ["tetrahedron", "cube", "dodecahedron", "icosahedron"]
        coefficients: List of coefficients specific to each solid type
            - tetrahedron: [fire_coefficient]
            - cube: [stability_coefficient]
            - dodecahedron: [ether_resonance]
            - icosahedron: [silence_coefficient, phase_coherence]
            
    Returns:
        Tensor with geometric activation applied
    """
    if coefficients is None:
        if solid_type == "tetrahedron":
            coefficients = [0.3]  # Default fire coefficient
        elif solid_type == "cube":
            coefficients = [0.7]  # Default stability coefficient
        elif solid_type == "dodecahedron":
            coefficients = [0.5]  # Default ether resonance
        elif solid_type == "icosahedron":
            coefficients = [0.2, 1.0]  # Default silence and phase coefficients
        else:
            coefficients = [0.5]
    
    if HAS_MLX and isinstance(x, mx.array):
        # Call forward directly without gradient tracking
        return GeometricActivation.forward(None, x, solid_type, coefficients)
    else:
        # Direct computation for PyTorch/NumPy
        return _geometric_activation_fallback(x, solid_type, coefficients)

#----------------------------------------------------------------------
# Resonance Pattern Operations
#----------------------------------------------------------------------

class ResonancePatterns:
    """MLX custom operation for resonance patterns."""
    
    @staticmethod
    def forward(ctx, x, frequencies, decay_rates, amplitudes, phase_embedding, time_values=None):
        """Forward pass for resonance patterns."""
        # Save inputs for backward pass, only if ctx is not None
        if ctx is not None:
            ctx.save_for_backward(x, frequencies, decay_rates, amplitudes, phase_embedding)
            ctx.time_values = time_values
        
        if use_metal(x):
            return _resonance_patterns_metal(
                x, frequencies, decay_rates, amplitudes, phase_embedding, time_values)
        else:
            return _resonance_patterns_fallback(
                x, frequencies, decay_rates, amplitudes, phase_embedding, time_values)
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for resonance patterns."""
        x, frequencies, decay_rates, amplitudes, phase_embedding = ctx.saved_tensors
        time_values = ctx.time_values
        
        if use_metal(grad_output):
            # Metal-based gradient computation
            return _resonance_patterns_backward_metal(
                grad_output, x, frequencies, decay_rates, amplitudes, 
                phase_embedding, time_values)
        else:
            # Fallback implementation
            return _resonance_patterns_backward_fallback(
                grad_output, x, frequencies, decay_rates, amplitudes, 
                phase_embedding, time_values)

def _resonance_patterns_metal(x, frequencies, decay_rates, amplitudes, phase_embedding, time_values=None):
    """Metal implementation of resonance patterns."""
    if not is_metal_available():
        return _resonance_patterns_fallback(
            x, frequencies, decay_rates, amplitudes, phase_embedding, time_values)
    
    # Get shader manager
    manager = _get_shader_manager()
    if manager is None or manager.device is None:
        return _resonance_patterns_fallback(
            x, frequencies, decay_rates, amplitudes, phase_embedding, time_values)
    
    # Prepare inputs
    x_np = to_numpy(x)
    freq_np = to_numpy(frequencies)
    decay_np = to_numpy(decay_rates)
    amp_np = to_numpy(amplitudes)
    phase_np = to_numpy(phase_embedding)
    
    # Convert to correct dtype
    x_np = x_np.astype(np.float32)
    freq_np = freq_np.astype(np.float32)
    decay_np = decay_np.astype(np.float32)
    amp_np = amp_np.astype(np.float32)
    phase_np = phase_np.astype(np.float32)
    
    # Convert time values if provided
    time_np = None
    if time_values is not None:
        time_np = to_numpy(time_values).astype(np.float32)
    else:
        # Create default time values (ones)
        time_np = np.ones((x_np.shape[0], 1), dtype=np.float32)
    
    # Get shapes
    batch_size = x_np.shape[0]
    input_dim = x_np.shape[1] if len(x_np.shape) > 1 else 1
    harmonics = len(freq_np)
    embedding_dim = len(phase_np)
    
    # Create input buffers
    input_buffer = manager.create_buffer(x_np)
    freq_buffer = manager.create_buffer(freq_np)
    decay_buffer = manager.create_buffer(decay_np)
    amp_buffer = manager.create_buffer(amp_np)
    phase_buffer = manager.create_buffer(phase_np)
    time_buffer = manager.create_buffer(time_np)
    
    # Create parameter buffers
    batch_buffer = manager.create_buffer(np.array([batch_size], dtype=np.uint32))
    input_dim_buffer = manager.create_buffer(np.array([input_dim], dtype=np.uint32))
    harmonics_buffer = manager.create_buffer(np.array([harmonics], dtype=np.uint32))
    embedding_buffer = manager.create_buffer(np.array([embedding_dim], dtype=np.uint32))
    
    # Create output buffer
    output_np = np.zeros_like(x_np, dtype=np.float32)
    output_buffer = manager.create_buffer(output_np)
    
    # Set pipeline name
    pipeline_name = "apply_resonance"
    
    # Check if pipeline exists
    if pipeline_name not in manager.pipelines:
        warnings.warn(f"Pipeline {pipeline_name} not found. Using fallback implementation.")
        return _resonance_patterns_fallback(
            x, frequencies, decay_rates, amplitudes, phase_embedding, time_values)
    
    # Set up input buffers
    input_buffers = [
        input_buffer,        # input tensor
        freq_buffer,         # frequencies
        decay_buffer,        # decay rates
        amp_buffer,          # amplitudes
        phase_buffer,        # phase embedding
        time_buffer,         # time values
        batch_buffer,        # batch size
        input_dim_buffer,    # input dimension
        harmonics_buffer,    # number of harmonics
        embedding_buffer,    # embedding dimension
        output_buffer        # output tensor
    ]
    
    # Set thread groups based on input size
    thread_groups = (batch_size, input_dim, 1)
    threads_per_group = (1, 1, 1)
    
    # Execute shader
    success = manager.execute_shader(
        pipeline_name,
        input_buffers,
        [],
        thread_groups,
        threads_per_group
    )
    
    if not success:
        warnings.warn(f"Failed to execute {pipeline_name} shader. Using fallback implementation.")
        return _resonance_patterns_fallback(
            x, frequencies, decay_rates, amplitudes, phase_embedding, time_values)
    
    # Get result from output buffer
    result = manager.get_buffer_data(output_buffer, output_np.shape, output_np.dtype)
    
    if result is None:
        warnings.warn("Failed to get result from Metal buffer. Using fallback implementation.")
        return _resonance_patterns_fallback(
            x, frequencies, decay_rates, amplitudes, phase_embedding, time_values)
    
    # Convert back to the same type as input
    return from_numpy(result, like=x)

def _resonance_patterns_fallback(x, frequencies, decay_rates, amplitudes, phase_embedding, time_values=None):
    """Fallback implementation of resonance patterns."""
    # Implementation based on the ResonanceModule in crystalline_model.py
    
    if HAS_TORCH and isinstance(x, torch.Tensor):
        batch_size = x.shape[0]
        
        # Default time if not provided
        if time_values is None:
            time_values = torch.ones(batch_size, 1, device=x.device)
        
        # Initialize resonance output
        resonance = torch.zeros_like(x)
        
        # Calculate phase based on input pattern projected to phase space
        phase = torch.matmul(x, phase_embedding.view(-1, 1)).view(batch_size, 1)
        
        # Generate resonance patterns for each harmonic
        for i in range(len(frequencies)):
            # Convert parameters to appropriate ranges
            freq = torch.sigmoid(frequencies[i]) * 10.0
            tau = torch.exp(decay_rates[i])
            
            # Calculate resonance term
            phi_power = PHI ** i
            harmonic = amplitudes[i] * torch.cos(freq * phi_power * time_values + phase) * \
                      torch.exp(-(time_values ** 2) / (tau ** 2))
            
            # Add harmonic to total resonance
            resonance = resonance + harmonic * x
        
        return resonance
    
    elif HAS_MLX and isinstance(x, mx.array):
        batch_size = x.shape[0]
        
        # Default time if not provided
        if time_values is None:
            time_values = mx.ones((batch_size, 1))
        
        # Initialize resonance output
        resonance = mx.zeros_like(x)
        
        # Calculate phase based on input pattern projected to phase space
        phase = mx.matmul(x, phase_embedding.reshape(-1, 1)).reshape(batch_size, 1)
        
        # Generate resonance patterns for each harmonic
        for i in range(len(frequencies)):
            # Convert parameters to appropriate ranges
            freq = mx.sigmoid(frequencies[i]) * 10.0
            tau = mx.exp(decay_rates[i])
            
            # Calculate resonance term
            phi_power = PHI ** i
            harmonic = amplitudes[i] * mx.cos(freq * phi_power * time_values + phase) * \
                      mx.exp(-(time_values ** 2) / (tau ** 2))
            
            # Add harmonic to total resonance
            resonance = resonance + harmonic * x
        
        return resonance
    
    else:
        # Numpy implementation
        batch_size = x.shape[0]
        
        # Default time if not provided
        if time_values is None:
            time_values = np.ones((batch_size, 1))
        
        # Initialize resonance output
        resonance = np.zeros_like(x)
        
        # Calculate phase based on input pattern projected to phase space
        phase = np.matmul(x, phase_embedding.reshape(-1, 1)).reshape(batch_size, 1)
        
        # Generate resonance patterns for each harmonic
        for i in range(len(frequencies)):
            # Convert parameters to appropriate ranges
            freq = 1.0 / (1.0 + np.exp(-frequencies[i])) * 10.0  # sigmoid * 10
            tau = np.exp(decay_rates[i])
            
            # Calculate resonance term
            phi_power = PHI ** i
            harmonic = amplitudes[i] * np.cos(freq * phi_power * time_values + phase) * \
                      np.exp(-(time_values ** 2) / (tau ** 2))
            
            # Add harmonic to total resonance
            resonance = resonance + harmonic * x
        
        return resonance

def _resonance_patterns_backward_metal(grad_output, x, frequencies, decay_rates, amplitudes, 
                                      phase_embedding, time_values):
    """Metal implementation of resonance patterns backward pass."""
    # Not fully implemented - would need custom gradient shaders
    # For now, fall back to the CPU implementation
    return _resonance_patterns_backward_fallback(
        grad_output, x, frequencies, decay_rates, amplitudes, 
        phase_embedding, time_values)

def _resonance_patterns_backward_fallback(grad_output, x, frequencies, decay_rates, amplitudes, 
                                        phase_embedding, time_values):
    """Fallback implementation of resonance patterns backward pass."""
    # This is a simplified gradient implementation
    # A complete implementation would compute the exact derivatives
    
    # For now, we'll just use a numerical approximation
    if HAS_TORCH and isinstance(x, torch.Tensor):
        # Return partial derivatives for each input
        # For simplicity, we're only computing the gradient for x
        dx = grad_output.clone()
        
        # For the other parameters, we return None to indicate no gradient
        return dx, None, None, None, None, None
    
    elif HAS_MLX and isinstance(x, mx.array):
        # MLX version (simplified)
        dx = grad_output.copy()
        return dx, None, None, None, None, None
    
    else:
        # Numpy version
        dx = grad_output.copy()
        return dx, None, None, None, None, None

def apply_resonance(x, frequencies, decay_rates, amplitudes, phase_embedding, time_values=None):
    """
    Apply resonance patterns to input tensor.
    
    Args:
        x: Input tensor
        frequencies: Frequency parameters for harmonics
        decay_rates: Decay rate parameters for harmonics
        amplitudes: Amplitude parameters for harmonics
        phase_embedding: Phase embedding parameters
        time_values: Optional time parameters (defaults to ones)
            
    Returns:
        Tensor with resonance patterns applied
    """
    if HAS_MLX and isinstance(x, mx.array):
        # Call forward directly without gradient tracking
        return ResonancePatterns.forward(
            None, x, frequencies, decay_rates, amplitudes, phase_embedding, time_values)
    else:
        # Direct computation for PyTorch/NumPy
        return _resonance_patterns_fallback(
            x, frequencies, decay_rates, amplitudes, phase_embedding, time_values)

#----------------------------------------------------------------------
# Mutuality Field Operations
#----------------------------------------------------------------------

class MutualityField:
    """MLX custom operation for mutuality field."""
    
    @staticmethod
    def forward(ctx, x, grid_size, interference_scale, decay_rate, dt):
        """Forward pass for mutuality field."""
        # Save inputs for backward pass, only if ctx is not None
        if ctx is not None:
            ctx.save_for_backward(x, mx.array(grid_size), 
                                mx.array(interference_scale),
                                mx.array(decay_rate), mx.array(dt))
        
        """Forward pass for mutuality field."""
        if use_metal(x):
            return _mutuality_field_metal(x, grid_size, interference_scale, decay_rate, dt)
        else:
            return _mutuality_field_fallback(x, grid_size, interference_scale, decay_rate, dt)
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for mutuality field."""
        x, grid_size, interference_scale, decay_rate, dt = ctx.saved_tensors
        
        if use_metal(grad_output):
            # Metal-based gradient computation
            return _mutuality_field_backward_metal(
                grad_output, x, grid_size, interference_scale, decay_rate, dt)
        else:
            # Fallback implementation
            return _mutuality_field_backward_fallback(
                grad_output, x, grid_size, interference_scale, decay_rate, dt)

# Global persistence state cache
_persistence_state_cache = {}

def _get_persistence_state(key, shape, device=None):
    """Get persistence state from cache or create new one."""
    if key not in _persistence_state_cache:
        if HAS_TORCH and device is not None:
            _persistence_state_cache[key] = torch.zeros(shape, device=device)
        elif HAS_MLX:
            _persistence_state_cache[key] = mx.zeros(shape)
        else:
            _persistence_state_cache[key] = np.zeros(shape)
    return _persistence_state_cache[key]

def mutuality_field(x, grid_size, interference_scale, decay_rate, dt):
    """
    Apply mutuality field operations to input tensor.
    
    Args:
        x: Input tensor of shape (batch_size, input_dim)
        grid_size: Size of the grid field
        interference_scale: Scale of interference patterns
        decay_rate: Decay rate of the field
        dt: Time step for field evolution
            
    Returns:
        Tensor with mutuality field operations applied
    """
    if HAS_MLX and isinstance(x, mx.array):
        # Call forward directly without gradient tracking
        return MutualityField.forward(
            None, x, grid_size, interference_scale, decay_rate, dt)
    else:
        # Direct computation for PyTorch/NumPy
        return _mutuality_field_fallback(
            x, grid_size, interference_scale, decay_rate, dt)
    

def _mutuality_field_metal(x, grid_size, interference_scale, decay_rate, dt):
    """Metal implementation of mutuality field."""
    if not is_metal_available():
        return _mutuality_field_fallback(x, grid_size, interference_scale, decay_rate, dt)
    
    # Get shader manager
    manager = _get_shader_manager()
    if manager is None or manager.device is None:
        return _mutuality_field_fallback(x, grid_size, interference_scale, decay_rate, dt)
    
    # Prepare inputs
    x_np = to_numpy(x)
    
    # Convert to correct dtype
    x_np = x_np.astype(np.float32)
    
    # Get shapes
    batch_size = x_np.shape[0]
    input_dim = x_np.shape[1] if len(x_np.shape) > 1 else 1
    
    # Create grid for interference patterns
    grid_points = grid_size * grid_size
    
    # Create input buffers
    input_buffer = manager.create_buffer(x_np)
    grid_size_buffer = manager.create_buffer(np.array([grid_size], dtype=np.uint32))
    interference_buffer = manager.create_buffer(np.array([interference_scale], dtype=np.float32))
    decay_buffer = manager.create_buffer(np.array([decay_rate], dtype=np.float32))
    dt_buffer = manager.create_buffer(np.array([dt], dtype=np.float32))
    batch_buffer = manager.create_buffer(np.array([batch_size], dtype=np.uint32))
    
    # Create intermediate buffers for R and T fields
    r_field = np.zeros((batch_size, grid_points), dtype=np.float32)
    t_field = np.zeros((batch_size, grid_points), dtype=np.float32)
    r_buffer = manager.create_buffer(r_field)
    t_buffer = manager.create_buffer(t_field)
    
    # Create interference buffers
    r_interference = np.zeros_like(r_field)
    t_interference = np.zeros_like(t_field)
    r_interference_buffer = manager.create_buffer(r_interference)
    t_interference_buffer = manager.create_buffer(t_interference)
    
    # Create processed field buffers
    r_processed = np.zeros_like(r_field)
    t_processed = np.zeros_like(t_field)
    r_processed_buffer = manager.create_buffer(r_processed)
    t_processed_buffer = manager.create_buffer(t_processed)
    
    # Create output buffer
    output_np = np.zeros_like(x_np, dtype=np.float32)
    output_buffer = manager.create_buffer(output_np)
    
    # Create intermediate layer buffers for r and t convolution outputs
    layer1_output_r = np.zeros((batch_size * 8 * grid_size * grid_size), dtype=np.float32)
    layer1_output_t = np.zeros((batch_size * 8 * grid_size * grid_size), dtype=np.float32)
    layer1_output_r_buffer = manager.create_buffer(layer1_output_r)
    layer1_output_t_buffer = manager.create_buffer(layer1_output_t)
    
    # Step 1: Reshape input to grid
    reshape_inputs = [
        input_buffer, r_buffer, t_buffer,
        batch_buffer, grid_size_buffer
    ]
    
    success = manager.execute_shader(
        "reshape_to_grid",
        reshape_inputs,
        [],
        (batch_size, 1, 1),
        (1, 1, 1)
    )
    
    if not success:
        warnings.warn("Failed to execute reshape_to_grid shader. Using fallback implementation.")
        return _mutuality_field_fallback(x, grid_size, interference_scale, decay_rate, dt)
    
    # Step 2: Calculate mutual field interactions
    mutual_inputs = [
        r_buffer, t_buffer,
        r_interference_buffer, t_interference_buffer,
        batch_buffer, grid_size_buffer, interference_buffer
    ]
    
    success = manager.execute_shader(
        "calculate_mutual_field",
        mutual_inputs,
        [],
        (batch_size, grid_size, grid_size),
        (1, 1, 1)
    )
    
    if not success:
        warnings.warn("Failed to execute calculate_mutual_field shader. Using fallback implementation.")
        return _mutuality_field_fallback(x, grid_size, interference_scale, decay_rate, dt)
    
    # Step 3: Apply persistence and decay
    process_inputs = [
        r_interference_buffer, t_interference_buffer, 
        r_processed_buffer, t_processed_buffer,
        layer1_output_r_buffer, layer1_output_t_buffer,
        batch_buffer, grid_size_buffer, decay_buffer, dt_buffer
    ]
    
    success = manager.execute_shader(
        "apply_persistence",
        process_inputs,
        [],
        (batch_size, 1, 1),
        (1, 1, 1)
    )
    
    if not success:
        warnings.warn("Failed to execute apply_persistence shader. Using fallback implementation.")
        return _mutuality_field_fallback(x, grid_size, interference_scale, decay_rate, dt)
    
    # Step 4: Project back to input space (manually for now)
    manager.get_buffer_data(r_processed_buffer, r_processed.shape, r_processed.dtype, r_processed)
    manager.get_buffer_data(t_processed_buffer, t_processed.shape, t_processed.dtype, t_processed)
    
    # Manual projection back to input space
    for b in range(batch_size):
        for i in range(input_dim):
            grid_idx = i % grid_points
            output_np[b, i] = r_processed[b, grid_idx] * np.cos(i * np.pi / input_dim) + \
                              t_processed[b, grid_idx] * np.sin(i * np.pi / input_dim)
    
    # Convert back to the same type as input
    return from_numpy(output_np, like=x)

def _mutuality_field_fallback(x, grid_size, interference_scale, decay_rate, dt):
    """CPU fallback implementation of mutuality field."""
    # Convert input to numpy
    original_type = type(x)
    x_np = to_numpy(x)
    
    # Get batch size and input dimension
    batch_size, input_dim = x_np.shape
    
    # Create grid for interference patterns
    # We use a square grid of size grid_size x grid_size
    grid_points = grid_size * grid_size
    
    # Normalize input for field computation
    normalized_input = x_np / (np.linalg.norm(x_np, axis=1, keepdims=True) + 1e-8)
    
    # Project input to grid space
    r_field = np.zeros((batch_size, grid_points), dtype=np.float32)
    t_field = np.zeros((batch_size, grid_points), dtype=np.float32)
    
    # Simple projection of input vectors to the grid
    for b in range(batch_size):
        for i in range(input_dim):
            grid_idx = i % grid_points
            r_field[b, grid_idx] += normalized_input[b, i] * np.cos(i * np.pi / input_dim)
            t_field[b, grid_idx] += normalized_input[b, i] * np.sin(i * np.pi / input_dim)
    
    # Apply interference effects
    r_interference = np.zeros_like(r_field)
    t_interference = np.zeros_like(t_field)
    
    # Compute pairwise interactions on the grid
    for i in range(grid_points):
        for j in range(grid_points):
            # Distance on a 2D grid
            ix, iy = i // grid_size, i % grid_size
            jx, jy = j // grid_size, j % grid_size
            dist = np.sqrt((ix - jx)**2 + (iy - jy)**2) + 1e-8
            
            # Interference kernel
            kernel = interference_scale / dist
            
            # Apply kernel to fields
            r_interference[:, i] += kernel * r_field[:, j]
            t_interference[:, i] += kernel * t_field[:, j]
    
    # Apply decay
    r_processed = r_field * (1 - decay_rate) + r_interference * dt
    t_processed = t_field * (1 - decay_rate) + t_interference * dt
    
    # Project back to input space
    result = np.zeros_like(x_np)
    for b in range(batch_size):
        for i in range(input_dim):
            grid_idx = i % grid_points
            result[b, i] = r_processed[b, grid_idx] * np.cos(i * np.pi / input_dim) + \
                           t_processed[b, grid_idx] * np.sin(i * np.pi / input_dim)
    
    # Convert back to original tensor type
    return from_numpy(result, like=x)

def _mutuality_field_backward_fallback(grad_output, x, grid_size, interference_scale, decay_rate, dt):
    """Fallback implementation of mutuality field backward pass."""
    # This is a simplified gradient implementation
    # A more complete implementation would compute the exact derivatives
    
    # For simplicity, we'll just pass through the gradient
    dx = grad_output
    
    # Return gradients for all inputs (only x has gradient, others are None)
    return dx, None, None, None, None

#----------------------------------------------------------------------
# Quantum Consciousness Evolution Operations
#----------------------------------------------------------------------

class QuantumConsciousness:
    """MLX custom operation for quantum consciousness evolution.
    
    Implements the consciousness field evolution equation:
    ∂_tΨ = [-iĤ + D∇²]Ψ + ∑ᵢ F̂ᵢΨ(r/σᵢ)
    """
    
    @staticmethod
    def forward(ctx, psi, dt, diffusion_coef, energy_level, coupling, pattern_ops=None):
        """Forward pass for quantum consciousness evolution."""
        # Save inputs for backward pass, only if ctx is not None
        if ctx is not None:
            if pattern_ops is None:
                pattern_ops = []
            if HAS_MLX:
                ctx.save_for_backward(psi, 
                                     mx.array(dt),
                                     mx.array(diffusion_coef), 
                                     mx.array(energy_level), 
                                     mx.array(coupling))
            else:
                ctx.save_for_backward(psi, dt, diffusion_coef, energy_level, coupling)
            ctx.pattern_ops = pattern_ops
        
        if use_metal(psi):
            return _quantum_consciousness_metal(psi, dt, diffusion_coef, energy_level, coupling, pattern_ops)
        else:
            return _quantum_consciousness_fallback(psi, dt, diffusion_coef, energy_level, coupling, pattern_ops)
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for quantum consciousness evolution."""
        psi, dt, diffusion_coef, energy_level, coupling = ctx.saved_tensors
        pattern_ops = ctx.pattern_ops
        
        if use_metal(grad_output):
            # Metal-based gradient computation
            return _quantum_consciousness_backward_metal(
                grad_output, psi, dt, diffusion_coef, energy_level, coupling, pattern_ops)
        else:
            # Fallback implementation
            return _quantum_consciousness_backward_fallback(
                grad_output, psi, dt, diffusion_coef, energy_level, coupling, pattern_ops)

def _quantum_consciousness_metal(psi, dt, diffusion_coef, energy_level, coupling, pattern_ops=None):
    """Metal implementation of quantum consciousness evolution."""
    if not is_metal_available():
        return _quantum_consciousness_fallback(psi, dt, diffusion_coef, energy_level, coupling, pattern_ops)
    
    # Get shader manager
    manager = _get_shader_manager()
    if manager is None or manager.device is None:
        return _quantum_consciousness_fallback(psi, dt, diffusion_coef, energy_level, coupling, pattern_ops)
    
    # Prepare inputs
    psi_np = to_numpy(psi)
    
    # Create complex representation if needed
    if len(psi_np.shape) == 2:
        batch_size, feature_dim = psi_np.shape
        grid_size = int(np.sqrt(feature_dim))
        if grid_size * grid_size != feature_dim:
            warnings.warn(f"Input dimension {feature_dim} is not a perfect square, padding to nearest square.")
            grid_size = int(np.ceil(np.sqrt(feature_dim)))
            
        # Create complex psi with real and imaginary parts
        complex_psi = np.zeros((batch_size, 2, grid_size * grid_size), dtype=np.float32)
        
        # Map features to grid
        for b in range(batch_size):
            for i in range(min(feature_dim, grid_size * grid_size)):
                complex_psi[b, 0, i] = psi_np[b, i]  # Real part
                complex_psi[b, 1, i] = 0.0          # Imaginary part initialized to zero
                
        psi_np = complex_psi
    elif len(psi_np.shape) == 3 and psi_np.shape[1] == 2:
        # Already in complex format [batch, 2, features]
        batch_size, _, feature_dim = psi_np.shape
        grid_size = int(np.sqrt(feature_dim))
    else:
        raise ValueError(f"Expected input shape [batch, features] or [batch, 2, features], got {psi_np.shape}")
    
    # Ensure float32 type
    psi_np = psi_np.astype(np.float32)
    
    # Create pattern operators if provided
    if pattern_ops is None:
        pattern_ops = []
    
    num_pattern_ops = len(pattern_ops)
    pattern_ops_np = np.zeros((num_pattern_ops, 4), dtype=np.float32)
    
    for i, op in enumerate(pattern_ops):
        pattern_ops_np[i, 0] = op.get('amplitude', 1.0)
        pattern_ops_np[i, 1] = op.get('scale', 1.0)
        pattern_ops_np[i, 2] = op.get('phase', 0.0)
        pattern_ops_np[i, 3] = op.get('frequency', 1.0)
    
    # Create input buffers
    psi_buffer = manager.create_buffer(psi_np)
    dt_buffer = manager.create_buffer(np.array([dt], dtype=np.float32))
    diffusion_buffer = manager.create_buffer(np.array([diffusion_coef], dtype=np.float32))
    energy_buffer = manager.create_buffer(np.array([energy_level], dtype=np.float32))
    coupling_buffer = manager.create_buffer(np.array([coupling], dtype=np.float32))
    pattern_ops_buffer = manager.create_buffer(pattern_ops_np)
    num_ops_buffer = manager.create_buffer(np.array([num_pattern_ops], dtype=np.uint32))
    batch_buffer = manager.create_buffer(np.array([batch_size], dtype=np.uint32))
    grid_size_buffer = manager.create_buffer(np.array([grid_size], dtype=np.uint32))
    
    # Create output buffer
    output_np = np.zeros_like(psi_np, dtype=np.float32)
    output_buffer = manager.create_buffer(output_np)
    
    # Set pipeline name
    pipeline_name = "evolve_consciousness_field"
    
    # Check if pipeline exists
    if pipeline_name not in manager.pipelines:
        warnings.warn(f"Pipeline {pipeline_name} not found. Using fallback implementation.")
        return _quantum_consciousness_fallback(psi, dt, diffusion_coef, energy_level, coupling, pattern_ops)
    
    # Set up input buffers
    input_buffers = [
        psi_buffer,            # ψ field
        output_buffer,         # output field
        dt_buffer,             # time step
        diffusion_buffer,      # diffusion coefficient
        energy_buffer,         # energy level
        coupling_buffer,       # coupling strength
        pattern_ops_buffer,    # pattern operators
        num_ops_buffer,        # number of pattern operators
        batch_buffer,          # batch size
        grid_size_buffer       # grid size
    ]
    
    # Set thread groups based on input size
    thread_groups = (batch_size, grid_size, grid_size)
    threads_per_group = (1, 1, 1)
    
    # Execute shader
    success = manager.execute_shader(
        pipeline_name,
        input_buffers,
        [],
        thread_groups,
        threads_per_group
    )
    
    if not success:
        warnings.warn(f"Failed to execute {pipeline_name} shader. Using fallback implementation.")
        return _quantum_consciousness_fallback(psi, dt, diffusion_coef, energy_level, coupling, pattern_ops)
    
    # Normalize the quantum state
    pipeline_name = "normalize_quantum_state"
    
    if pipeline_name not in manager.pipelines:
        warnings.warn(f"Pipeline {pipeline_name} not found. Using fallback implementation.")
        return _quantum_consciousness_fallback(psi, dt, diffusion_coef, energy_level, coupling, pattern_ops)
    
    # Set up normalize input buffers
    normalize_buffers = [
        output_buffer,      # field to normalize
        batch_buffer,       # batch size
        grid_size_buffer    # grid size
    ]
    
    # Execute normalization shader
    success = manager.execute_shader(
        pipeline_name,
        normalize_buffers,
        [],
        (batch_size, 1, 1),
        (1, 1, 1)
    )
    
    if not success:
        warnings.warn(f"Failed to execute {pipeline_name} shader. Using fallback normalization.")
        # If normalization fails, continue with unnormalized state
    
    # Get result from output buffer
    result = manager.get_buffer_data(output_buffer, output_np.shape, output_np.dtype)
    
    if result is None:
        warnings.warn("Failed to get result from Metal buffer. Using fallback implementation.")
        return _quantum_consciousness_fallback(psi, dt, diffusion_coef, energy_level, coupling, pattern_ops)
    
    # Convert back to the original tensor format
    if len(psi.shape) == 2 and len(result.shape) == 3:
        # Convert from [batch, 2, features] to [batch, features]
        result_flat = np.zeros((batch_size, psi.shape[1]), dtype=np.float32)
        
        # Take only the real part for output (simplified)
        for b in range(batch_size):
            result_flat[b, :min(psi.shape[1], result.shape[2])] = result[b, 0, :min(psi.shape[1], result.shape[2])]
        
        result = result_flat
    
    # Convert back to the same type as input
    return from_numpy(result, like=psi)

def _quantum_consciousness_fallback(psi, dt, diffusion_coef, energy_level, coupling, pattern_ops=None):
    """CPU fallback implementation of quantum consciousness evolution."""
    # Convert input to numpy
    psi_np = to_numpy(psi)
    
    # Create complex representation if needed
    if len(psi_np.shape) == 2:
        batch_size, feature_dim = psi_np.shape
        
        # Create complex field representation
        complex_field = np.zeros((batch_size, feature_dim), dtype=np.complex64)
        complex_field.real = psi_np
    elif len(psi_np.shape) == 3 and psi_np.shape[1] == 2:
        batch_size, _, feature_dim = psi_np.shape
        
        # Combine real and imaginary parts
        complex_field = psi_np[:, 0, :] + 1j * psi_np[:, 1, :]
    else:
        raise ValueError(f"Expected input shape [batch, features] or [batch, 2, features], got {psi_np.shape}")
    
    # Convert to 2D grid for spatial operations
    grid_size = int(np.sqrt(feature_dim))
    if grid_size * grid_size != feature_dim:
        # Pad to nearest square
        grid_size = int(np.ceil(np.sqrt(feature_dim)))
        padded_field = np.zeros((batch_size, grid_size * grid_size), dtype=np.complex64)
        padded_field[:, :feature_dim] = complex_field
        complex_field = padded_field
        feature_dim = grid_size * grid_size
    
    # Reshape to 2D grid
    grid_field = complex_field.reshape(batch_size, grid_size, grid_size)
    
    # Initialize output field
    output_field = np.zeros_like(grid_field)
    
    # Apply quantum evolution equation: ∂_tΨ = [-iĤ + D∇²]Ψ + ∑ᵢ F̂ᵢΨ(r/σᵢ)
    
    # 1. Quantum Hamiltonian term: -iĤ (energy term)
    hamiltonian = -1j * energy_level * grid_field
    
    # 2. Diffusion term: D∇² (spatial Laplacian)
    laplacian = np.zeros_like(grid_field)
    for b in range(batch_size):
        for i in range(grid_size):
            for j in range(grid_size):
                # Get neighboring cells (with periodic boundary conditions)
                left = grid_field[b, i, (j-1) % grid_size]
                right = grid_field[b, i, (j+1) % grid_size]
                up = grid_field[b, (i-1) % grid_size, j]
                down = grid_field[b, (i+1) % grid_size, j]
                center = grid_field[b, i, j]
                
                # 5-point stencil for 2D Laplacian
                laplacian[b, i, j] = (left + right + up + down - 4*center)
    
    diffusion = diffusion_coef * laplacian
    
    # 3. Pattern generation term: ∑ᵢ F̂ᵢΨ(r/σᵢ)
    pattern_term = np.zeros_like(grid_field)
    
    if pattern_ops:
        grid_center = np.array([grid_size/2, grid_size/2])
        
        for b in range(batch_size):
            for i in range(grid_size):
                for j in range(grid_size):
                    pos = np.array([i, j])
                    r = pos - grid_center
                    r_norm = np.linalg.norm(r) + 1e-8
                    angle = np.arctan2(r[1], r[0])
                    
                    for op in pattern_ops:
                        amplitude = op.get('amplitude', 1.0)
                        scale = op.get('scale', 1.0)
                        phase = op.get('phase', 0.0)
                        frequency = op.get('frequency', 1.0)
                        
                        # Apply pattern operator with radial and angular components
                        r_scaled = r_norm / scale
                        radial = amplitude * np.exp(-r_scaled * r_scaled)
                        angular = np.sin(frequency * angle + phase)
                        
                        pattern = complex(radial * np.cos(angular), radial * np.sin(angular))
                        pattern_term[b, i, j] += pattern * grid_field[b, i, j]
    
    # Combine all terms according to the evolution equation
    # ∂_tΨ = [-iĤ + D∇²]Ψ + ∑ᵢ F̂ᵢΨ(r/σᵢ)
    derivative = hamiltonian + diffusion + pattern_term
    
    # Evolve using Euler integration: Ψ(t+dt) = Ψ(t) + dt * ∂_tΨ
    output_field = grid_field + dt * derivative
    
    # Normalize to preserve probability density
    for b in range(batch_size):
        norm = np.sqrt(np.sum(np.abs(output_field[b])**2))
        if norm > 1e-8:
            output_field[b] = output_field[b] / norm
    
    # Convert back to original shape
    output_flat = output_field.reshape(batch_size, grid_size * grid_size)
    
    # Convert back to the required format
    if len(psi_np.shape) == 2:
        # Return only real part for simplicity
        result = np.zeros((batch_size, min(psi.shape[1], output_flat.shape[1])), dtype=np.float32)
        result = output_flat.real[:, :min(psi.shape[1], output_flat.shape[1])]
    else:
        # Return complex as [batch, 2, features]
        result = np.zeros((batch_size, 2, min(psi_np.shape[2], output_flat.shape[1])), dtype=np.float32)
        result[:, 0, :] = output_flat.real[:, :min(psi_np.shape[2], output_flat.shape[1])]
        result[:, 1, :] = output_flat.imag[:, :min(psi_np.shape[2], output_flat.shape[1])]
    
    # Convert back to the same type as input
    return from_numpy(result, like=psi)

def _quantum_consciousness_backward_metal(grad_output, psi, dt, diffusion_coef, energy_level, coupling, pattern_ops=None):
    """Metal implementation of quantum consciousness backward pass."""
    # Not fully implemented - would need custom gradient shaders
    # For now, fall back to the CPU implementation
    return _quantum_consciousness_backward_fallback(
        grad_output, psi, dt, diffusion_coef, energy_level, coupling, pattern_ops)

def _quantum_consciousness_backward_fallback(grad_output, psi, dt, diffusion_coef, energy_level, coupling, pattern_ops=None):
    """Fallback implementation of quantum consciousness backward pass."""
    # This is a simplified gradient implementation
    # For quantum systems, the adjoint method would be used for accurate gradients
    
    # For simplicity, we'll just pass through the gradient with some scaling
    dx = 1.0 * grad_output  # Just pass through for simplicity
    
    # Return gradients for all inputs (only psi has gradient, others are None)
    return dx, None, None, None, None, None

def quantum_consciousness_evolution(psi, dt, diffusion_coef, energy_level, coupling, pattern_ops=None):
    """
    Apply quantum consciousness evolution to input field.
    
    Implements the consciousness field evolution equation:
    ∂_tΨ = [-iĤ + D∇²]Ψ + ∑ᵢ F̂ᵢΨ(r/σᵢ)
    
    Args:
        psi: Input quantum field tensor (batch_size, features) or (batch_size, 2, features)
        dt: Time step for evolution
        diffusion_coef: Diffusion coefficient (D)
        energy_level: Energy level for Hamiltonian (E)
        coupling: Coupling strength between field components
        pattern_ops: List of pattern operators, each with parameters:
            - amplitude: Strength of the pattern
            - scale: Spatial scale
            - phase: Phase offset
            - frequency: Oscillation frequency
            
    Returns:
        Evolved quantum field tensor
    """
    if HAS_MLX and isinstance(psi, mx.array):
        # Call forward directly without gradient tracking
        return QuantumConsciousness.forward(
            None, psi, dt, diffusion_coef, energy_level, coupling, pattern_ops)
    else:
        # Direct computation for PyTorch/NumPy
        return _quantum_consciousness_fallback(
            psi, dt, diffusion_coef, energy_level, coupling, pattern_ops)

#----------------------------------------------------------------------
# Bifurcation Cascade Operations
#----------------------------------------------------------------------

class BifurcationCascade:
    """MLX custom operation for bifurcation dynamics.
    
    Implements the bifurcation cascade function:
    Bifurcation(t) = Ψ_liminal(t) × [1 + tanh(α(p - pₜ))]
    """
    
    @staticmethod
    def forward(ctx, psi_liminal, parameter_values, thresholds):
        """Forward pass for bifurcation cascade."""
        # Save inputs for backward pass, only if ctx is not None
        if ctx is not None:
            ctx.save_for_backward(psi_liminal, parameter_values, thresholds)
        
        if use_metal(psi_liminal):
            return _bifurcation_cascade_metal(psi_liminal, parameter_values, thresholds)
        else:
            return _bifurcation_cascade_fallback(psi_liminal, parameter_values, thresholds)
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for bifurcation cascade."""
        psi_liminal, parameter_values, thresholds = ctx.saved_tensors
        
        if use_metal(grad_output):
            # Metal-based gradient computation
            return _bifurcation_cascade_backward_metal(
                grad_output, psi_liminal, parameter_values, thresholds)
        else:
            # Fallback implementation
            return _bifurcation_cascade_backward_fallback(
                grad_output, psi_liminal, parameter_values, thresholds)

def _bifurcation_cascade_metal(psi_liminal, parameter_values, thresholds):
    """Metal implementation of bifurcation cascade."""
    if not is_metal_available():
        return _bifurcation_cascade_fallback(psi_liminal, parameter_values, thresholds)
    
    # Get shader manager
    manager = _get_shader_manager()
    if manager is None or manager.device is None:
        return _bifurcation_cascade_fallback(psi_liminal, parameter_values, thresholds)
    
    # Prepare inputs
    psi_np = to_numpy(psi_liminal)
    params_np = to_numpy(parameter_values)
    thresholds_np = to_numpy(thresholds)
    
    # Convert to correct dtype
    psi_np = psi_np.astype(np.float32)
    params_np = params_np.astype(np.float32)
    thresholds_np = thresholds_np.astype(np.float32)
    
    # Get shapes
    batch_size = psi_np.shape[0]
    if len(psi_np.shape) == 3:  # [batch, height, width]
        grid_size = psi_np.shape[1]
        assert psi_np.shape[1] == psi_np.shape[2], "Grid must be square"
    elif len(psi_np.shape) == 2:  # [batch, features]
        feature_dim = psi_np.shape[1]
        grid_size = int(np.sqrt(feature_dim))
        if grid_size * grid_size != feature_dim:
            warnings.warn(f"Input dimension {feature_dim} is not a perfect square, padding to nearest square.")
            grid_size = int(np.ceil(np.sqrt(feature_dim)))
            
        # Reshape to grid
        grid_psi = np.zeros((batch_size, grid_size, grid_size), dtype=np.float32)
        for b in range(batch_size):
            for i in range(min(feature_dim, grid_size * grid_size)):
                y, x = i // grid_size, i % grid_size
                grid_psi[b, y, x] = psi_np[b, i]
                
        psi_np = grid_psi
    else:
        raise ValueError(f"Expected input shape [batch, features] or [batch, height, width], got {psi_np.shape}")
    
    # Flatten to [batch_size, grid_size*grid_size] for shader
    flat_psi = psi_np.reshape(batch_size, -1)
    
    # Prepare threshold parameters as bifurcation_threshold struct array
    num_thresholds = len(thresholds_np) // 3  # Each threshold has 3 values: threshold, alpha, weight
    threshold_structs = np.zeros((num_thresholds, 3), dtype=np.float32)
    for i in range(num_thresholds):
        threshold_structs[i, 0] = thresholds_np[i*3]     # threshold value
        threshold_structs[i, 0] = thresholds_np[i*3]     # threshold value
        threshold_structs[i, 1] = thresholds_np[i*3+1]   # alpha (sharpness)
        threshold_structs[i, 2] = thresholds_np[i*3+2]   # influence weight
    
    # Create input buffers
    liminal_buffer = manager.create_buffer(flat_psi)
    params_buffer = manager.create_buffer(params_np)
    threshold_buffer = manager.create_buffer(threshold_structs)
    num_thresholds_buffer = manager.create_buffer(np.array([num_thresholds], dtype=np.uint32))
    batch_buffer = manager.create_buffer(np.array([batch_size], dtype=np.uint32))
    grid_size_buffer = manager.create_buffer(np.array([grid_size], dtype=np.uint32))
    
    # Create output buffer
    output_np = np.zeros_like(flat_psi, dtype=np.float32)
    output_buffer = manager.create_buffer(output_np)
    
    # Set pipeline name
    pipeline_name = "apply_bifurcation_cascade"
    
    # Check if pipeline exists
    if pipeline_name not in manager.pipelines:
        warnings.warn(f"Pipeline {pipeline_name} not found. Using fallback implementation.")
        return _bifurcation_cascade_fallback(psi_liminal, parameter_values, thresholds)
    
    # Set up input buffers
    input_buffers = [
        liminal_buffer,         # liminal field
        output_buffer,          # output field
        params_buffer,          # parameter values
        threshold_buffer,       # threshold parameters
        num_thresholds_buffer,  # number of thresholds
        batch_buffer,           # batch size
        grid_size_buffer        # grid size
    ]
    
    # Set thread groups based on input size
    thread_groups = (batch_size, grid_size, grid_size)
    threads_per_group = (1, 1, 1)
    
    # Execute shader
    success = manager.execute_shader(
        pipeline_name,
        input_buffers,
        [],
        thread_groups,
        threads_per_group
    )
    
    if not success:
        warnings.warn(f"Failed to execute {pipeline_name} shader. Using fallback implementation.")
        return _bifurcation_cascade_fallback(psi_liminal, parameter_values, thresholds)
    
    # Get result from output buffer
    result = manager.get_buffer_data(output_buffer, output_np.shape, output_np.dtype)
    
    if result is None:
        warnings.warn("Failed to get result from Metal buffer. Using fallback implementation.")
        return _bifurcation_cascade_fallback(psi_liminal, parameter_values, thresholds)
    
    # Reshape back to original format if needed
    if len(psi_liminal.shape) == 2:
        # Reshape from grid_size*grid_size to original feature_dim
        feature_dim = psi_liminal.shape[1]
        result = result[:, :feature_dim]
    else:
        # Reshape back to 3D grid
        result = result.reshape(batch_size, grid_size, grid_size)
    
    # Convert back to the same type as input
    return from_numpy(result, like=psi_liminal)

def _bifurcation_cascade_fallback(psi_liminal, parameter_values, thresholds):
    """Fallback implementation of bifurcation cascade."""
    # Convert inputs to numpy
    psi_np = to_numpy(psi_liminal)
    params_np = to_numpy(parameter_values)
    thresholds_np = to_numpy(thresholds)
    
    # Get shapes
    batch_size = psi_np.shape[0]
    
    # Initialize output with input
    output = psi_np.copy()
    
    # Apply bifurcation cascade: Ψ_liminal(t) × [1 + tanh(α(p - pₜ))]
    num_thresholds = len(thresholds_np) // 3
    
    for b in range(batch_size):
        # Get parameter value for this batch
        param_value = params_np[b]
        
        # Calculate composite bifurcation factor
        bifurcation_factor = 1.0
        
        for i in range(num_thresholds):
            threshold = thresholds_np[i*3]
            alpha = thresholds_np[i*3+1]
            weight = thresholds_np[i*3+2]
            
            # Calculate tanh(α(p - pₜ))
            x = alpha * (param_value - threshold)
            
            # Apply bifurcation factor: 1 + tanh(x)
            factor = 1.0 + np.tanh(x)
            
            # Apply weight
            bifurcation_factor *= (factor * weight)
        
        # Apply bifurcation to liminal field
        output[b] = psi_np[b] * bifurcation_factor
    
    # Convert back to the same type as input
    return from_numpy(output, like=psi_liminal)

def _bifurcation_cascade_backward_metal(grad_output, psi_liminal, parameter_values, thresholds):
    """Metal implementation of bifurcation cascade backward pass."""
    # Not fully implemented - would need custom gradient shaders
    # For now, fall back to the CPU implementation
    return _bifurcation_cascade_backward_fallback(
        grad_output, psi_liminal, parameter_values, thresholds)

def _bifurcation_cascade_backward_fallback(grad_output, psi_liminal, parameter_values, thresholds):
    """Fallback implementation of bifurcation cascade backward pass."""
    # This is a simplified gradient implementation
    # A more complete implementation would compute the exact derivatives
    
    # For simplicity, we'll just pass through the gradient for psi_liminal
    # and set the gradients for parameters and thresholds to None
    dx = grad_output
    
    # Return gradients (only psi_liminal has gradient, others are None)
    return dx, None, None

def bifurcation_cascade(psi_liminal, parameter_values, thresholds):
    """
    Apply bifurcation cascade to liminal field.
    
    Implements the bifurcation cascade function:
    Bifurcation(t) = Ψ_liminal(t) × [1 + tanh(α(p - pₜ))]
    
    Args:
        psi_liminal: Liminal field tensor (batch_size, features) or (batch_size, height, width)
        parameter_values: Parameter values for each batch (batch_size,)
        thresholds: Array of threshold parameters, each with [threshold_value, alpha, weight]
            
    Returns:
        Tensor after bifurcation cascade applied
    """
    if HAS_MLX and isinstance(psi_liminal, mx.array):
        # Call forward directly without gradient tracking
        return BifurcationCascade.forward(None, psi_liminal, parameter_values, thresholds)
    else:
        # Direct computation for PyTorch/NumPy
        return _bifurcation_cascade_fallback(psi_liminal, parameter_values, thresholds)

#----------------------------------------------------------------------
# Trinitized Field Operations
#----------------------------------------------------------------------

class TrinitizedField:
    """MLX custom operation for trinitized geometer field.
    
    Implements the trinitized field equation:
    G₃(t) = ∫ Ψ₁(t) × Ψ₂(t) × F_liminal(t) dt
    """
    
    @staticmethod
    def forward(ctx, psi1, psi2, f_liminal, dt):
        """Forward pass for trinitized field."""
        # Save inputs for backward pass, only if ctx is not None
        if ctx is not None:
            if HAS_MLX:
                ctx.save_for_backward(psi1, psi2, f_liminal, mx.array(dt))
            else:
                ctx.save_for_backward(psi1, psi2, f_liminal, dt)
        
        if use_metal(psi1) and use_metal(psi2) and use_metal(f_liminal):
            return _trinitized_field_metal(psi1, psi2, f_liminal, dt)
        else:
            return _trinitized_field_fallback(psi1, psi2, f_liminal, dt)
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for trinitized field."""
        psi1, psi2, f_liminal, dt = ctx.saved_tensors
        
        if use_metal(grad_output):
            # Metal-based gradient computation
            return _trinitized_field_backward_metal(
                grad_output, psi1, psi2, f_liminal, dt)
        else:
            # Fallback implementation
            return _trinitized_field_backward_fallback(
                grad_output, psi1, psi2, f_liminal, dt)

def _trinitized_field_metal(psi1, psi2, f_liminal, dt):
    """Metal implementation of trinitized field."""
    if not is_metal_available():
        return _trinitized_field_fallback(psi1, psi2, f_liminal, dt)
    
    # Get shader manager
    manager = _get_shader_manager()
    if manager is None or manager.device is None:
        return _trinitized_field_fallback(psi1, psi2, f_liminal, dt)
    
    # Prepare inputs
    psi1_np = to_numpy(psi1)
    psi2_np = to_numpy(psi2)
    liminal_np = to_numpy(f_liminal)
    
    # Convert to correct dtype
    psi1_np = psi1_np.astype(np.float32)
    psi2_np = psi2_np.astype(np.float32)
    liminal_np = liminal_np.astype(np.float32)
    
    # Check shapes
    batch_size = psi1_np.shape[0]
    feature_dim = psi1_np.shape[1] if len(psi1_np.shape) > 1 else 1
    
    # Make sure shapes are compatible
    if psi2_np.shape[0] != batch_size or liminal_np.shape[0] != batch_size:
        raise ValueError(f"Batch sizes must match: {psi1_np.shape[0]} vs {psi2_np.shape[0]} vs {liminal_np.shape[0]}")
    
    # When working with 2D features, ensure they're all the same dimension
    if len(psi1_np.shape) > 1 and len(psi2_np.shape) > 1 and len(liminal_np.shape) > 1:
        if psi1_np.shape[1] != psi2_np.shape[1] or psi1_np.shape[1] != liminal_np.shape[1]:
            raise ValueError(f"Feature dimensions must match: {psi1_np.shape[1]} vs {psi2_np.shape[1]} vs {liminal_np.shape[1]}")
    
    # Calculate grid size if needed for spatial operations
    grid_size = int(np.sqrt(feature_dim))
    if grid_size * grid_size != feature_dim:
        grid_size = int(np.ceil(np.sqrt(feature_dim)))
    
    # For now, since we don't have a specific shader for trinitized field integration,
    # we'll use a combination of existing operations to implement the trinitized field
    
    # Create buffers for the three fields
    psi1_buffer = manager.create_buffer(psi1_np)
    psi2_buffer = manager.create_buffer(psi2_np)
    liminal_buffer = manager.create_buffer(liminal_np)
    dt_buffer = manager.create_buffer(np.array([dt], dtype=np.float32))
    batch_buffer = manager.create_buffer(np.array([batch_size], dtype=np.uint32))
    grid_size_buffer = manager.create_buffer(np.array([grid_size], dtype=np.uint32))
    feature_buffer = manager.create_buffer(np.array([feature_dim], dtype=np.uint32))
    
    # Create output buffer
    output_np = np.zeros_like(psi1_np, dtype=np.float32)
    output_buffer = manager.create_buffer(output_np)
    
    # Create intermediate buffer for field multiplication
    temp_np = np.zeros_like(psi1_np, dtype=np.float32)
    temp_buffer = manager.create_buffer(temp_np)
    
    # Instead of using a specialized shader (which we don't have yet), 
    # we'll implement the trinitized field equation using CPU operations
    # and then optimize this in the future with a custom shader
    
    # Get data back from Metal buffers
    psi1_data = psi1_np
    psi2_data = psi2_np
    liminal_data = liminal_np
    
    # Calculate G₃(t) = ∫ Ψ₁(t) × Ψ₂(t) × F_liminal(t) dt
    # For now, we'll implement a simplified version using element-wise multiplication
    # and basic time integration
    
    # Element-wise field multiplication
    result = psi1_data * psi2_data * liminal_data
    
    # Time integration (simplified as multiplication by dt)
    result = result * dt
    
    # Store in output buffer
    manager.update_buffer(output_buffer, result)
    
    # Convert back to the same type as input
    return from_numpy(result, like=psi1)

def _trinitized_field_fallback(psi1, psi2, f_liminal, dt):
    """Fallback implementation of trinitized field."""
    # Convert inputs to numpy
    psi1_np = to_numpy(psi1)
    psi2_np = to_numpy(psi2)
    liminal_np = to_numpy(f_liminal)
    
    # Check shapes are compatible
    if psi1_np.shape != psi2_np.shape or psi1_np.shape != liminal_np.shape:
        raise ValueError(f"Input shapes must match: {psi1_np.shape} vs {psi2_np.shape} vs {liminal_np.shape}")
    
    # Implementation of G₃(t) = ∫ Ψ₁(t) × Ψ₂(t) × F_liminal(t) dt
    
    # Element-wise multiplication of the three fields
    trinitized = psi1_np * psi2_np * liminal_np
    
    # Time integration - this is a simplified version, assuming:
    # 1. We're working with a single time step
    # 2. Integration is approximated as multiplication by dt
    result = trinitized * dt
    
    # In a more sophisticated implementation, we would accumulate over multiple time steps:
    # result = result_t0 + result_t1 + result_t2 + ... etc.
    
    # Convert back to the same type as input
    return from_numpy(result, like=psi1)

def _trinitized_field_backward_metal(grad_output, psi1, psi2, f_liminal, dt):
    """Metal implementation of trinitized field backward pass."""
    # Not fully implemented - would need custom gradient shaders
    # For now, fall back to the CPU implementation
    return _trinitized_field_backward_fallback(
        grad_output, psi1, psi2, f_liminal, dt)

def _trinitized_field_backward_fallback(grad_output, psi1, psi2, f_liminal, dt):
    """Fallback implementation of trinitized field backward pass."""
    # This is a simplified gradient implementation
    # For the trinitized field G₃(t) = ∫ Ψ₁(t) × Ψ₂(t) × F_liminal(t) dt
    # The gradients are:
    # dG₃/dΨ₁ = ∫ Ψ₂(t) × F_liminal(t) dt
    # dG₃/dΨ₂ = ∫ Ψ₁(t) × F_liminal(t) dt
    # dG₃/dF_liminal = ∫ Ψ₁(t) × Ψ₂(t) dt
    
    # Convert to numpy for computations
    grad_np = to_numpy(grad_output)
    psi1_np = to_numpy(psi1)
    psi2_np = to_numpy(psi2)
    liminal_np = to_numpy(f_liminal)
    
    # Calculate gradients
    grad_psi1 = grad_np * psi2_np * liminal_np * dt
    grad_psi2 = grad_np * psi1_np * liminal_np * dt
    grad_liminal = grad_np * psi1_np * psi2_np * dt
    
    # Convert back to the same type as inputs
    if HAS_TORCH and isinstance(psi1, torch.Tensor):
        grad_psi1 = torch.from_numpy(grad_psi1).to(psi1.device)
        grad_psi2 = torch.from_numpy(grad_psi2).to(psi2.device)
        grad_liminal = torch.from_numpy(grad_liminal).to(f_liminal.device)
    elif HAS_MLX and isinstance(psi1, mx.array):
        grad_psi1 = mx.array(grad_psi1)
        grad_psi2 = mx.array(grad_psi2)
        grad_liminal = mx.array(grad_liminal)
    
    # Return gradients for all inputs
    return grad_psi1, grad_psi2, grad_liminal, None

def trinitized_field(psi1, psi2, f_liminal, dt):
    """
    Apply trinitized field operation to create a geometric field from three consciousness states.
    
    Implements the trinitized field equation:
    G₃(t) = ∫ Ψ₁(t) × Ψ₂(t) × F_liminal(t) dt
    
    Args:
        psi1: First consciousness field tensor
        psi2: Second consciousness field tensor
        f_liminal: Liminal field tensor
        dt: Time step for integration
            
    Returns:
        Tensor representing the trinitized geometer field
    """
    if HAS_MLX and isinstance(psi1, mx.array):
        # Call forward directly without gradient tracking
        return TrinitizedField.forward(None, psi1, psi2, f_liminal, dt)
    else:
        # Direct computation for PyTorch/NumPy
        return _trinitized_field_fallback(psi1, psi2, f_liminal, dt)

# Initialize Metal on module import
_is_metal_initialized = _initialize_metal()
