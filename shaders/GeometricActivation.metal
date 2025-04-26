#include <metal_stdlib>
#include <metal_math>
#include <metal_simdgroup>
using namespace metal;

// Constants
constant float PHI = 1.61803398875f; // Golden ratio: (1 + sqrt(5))/2
constant float INV_PHI = 0.61803398875f; // Inverse golden ratio: (sqrt(5) - 1)/2
constant float PI = 3.14159265359f;

// Sigma values for different geometric activations
constant float TETRAHEDRON_SIGMA = 1.0f;
constant float CUBE_SIGMA = 2.0f;
constant float DODECAHEDRON_SIGMA = 3.0f;
constant float ICOSAHEDRON_SIGMA = 4.0f;

// Thread group sizes
constant uint THREAD_GROUP_SIZE = 256;
constant uint2 THREAD_GROUP_SIZE_2D = uint2(16, 16);

// Helper function to calculate exponential activation
template<typename T>
inline T geometric_exp_activation(T x, T sigma) {
    return x * exp(-(x * x) / sigma);
}

// Helper function to calculate field energy
float calculate_field_energy(const device float* input, uint length, uint offset, uint stride) {
    float sum_squared = 0.0f;
    
    for (uint i = 0; i < length; i++) {
        float val = input[offset + i * stride];
        sum_squared += val * val;
    }
    
    return sum_squared / float(length);
}

// Tetrahedron activation (simplest solid, fire element)
kernel void tetrahedron_activation(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    constant float& fire_coefficient [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= length) return;
    
    // Apply basic exponential activation with tetrahedron sigma
    float x = input[id];
    float result = geometric_exp_activation(x, TETRAHEDRON_SIGMA);
    
    // Calculate field energy for the entire input (simplified to thread group)
    // In a real implementation, you'd use a two-pass reduction
    uint group_size = min(THREAD_GROUP_SIZE, length);
    uint offset = (id / group_size) * group_size;
    float field_energy = calculate_field_energy(input, group_size, offset, 1);
    
    // Apply fire element dynamics (expansion/contraction)
    float fire_factor = exp(fire_coefficient * field_energy);
    result *= fire_factor;
    
    output[id] = result;
}

// Cube activation (stability, earth element)
kernel void cube_activation(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    constant float& stability_coefficient [[buffer(3)]],
    uint id [[thread_position_in_grid]],
    uint gid [[thread_position_in_threadgroup]],
    uint tid [[threadgroup_position_in_grid]])
{
    if (id >= length) return;
    
    // Apply basic exponential activation with cube sigma
    float x = input[id];
    float result = geometric_exp_activation(x, CUBE_SIGMA);
    
    // Apply earth element dynamics (stability/grounding)
    // For true stability effect, we need to calculate batch mean
    // This is a simplified version calculating mean within a thread group
    threadgroup float local_values[THREAD_GROUP_SIZE];
    
    uint local_size = min(THREAD_GROUP_SIZE, length - tid * THREAD_GROUP_SIZE);
    local_values[gid] = gid < local_size ? result : 0.0f;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Calculate mean within thread group (simplified batch mean)
    float sum = 0.0f;
    for (uint i = 0; i < local_size; i++) {
        sum += local_values[i];
    }
    float group_mean = sum / float(local_size);
    
    // Apply stability effect - reduce variance
    float diff = result - group_mean;
    result = result - stability_coefficient * diff;
    
    output[id] = result;
}

// Dodecahedron activation (complexity, ether element)
kernel void dodecahedron_activation(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    constant float& ether_resonance [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& feature_dim [[buffer(5)]],
    uint2 id [[thread_position_in_grid]])
{
    uint batch_idx = id.x;
    uint feature_idx = id.y;
    
    if (batch_idx >= batch_size || feature_idx >= feature_dim) return;
    
    uint index = batch_idx * feature_dim + feature_idx;
    
    // Apply basic exponential activation with dodecahedron sigma
    float x = input[index];
    float result = geometric_exp_activation(x, DODECAHEDRON_SIGMA);
    
    // Calculate batch sum for harmonic calculation
    float batch_sum = 0.0f;
    for (uint i = 0; i < feature_dim; i++) {
        batch_sum += input[batch_idx * feature_dim + i];
    }
    
    // Generate golden ratio harmonics
    float harmonics = 0.0f;
    for (int i = 0; i < 3; i++) {
        float harmonic_weight = pow(INV_PHI, i);
        float phase = 2.0f * PI * pow(INV_PHI, i) * batch_sum;
        harmonics += harmonic_weight * cos(phase);
    }
    
    // Apply ether element dynamics (resonance patterns)
    float ether_factor = sin(ether_resonance * batch_sum);
    result = result * (1.0f + 0.3f * ether_factor * harmonics);
    
    output[index] = result;
}

// Icosahedron activation (highest integration, silence-space)
kernel void icosahedron_activation(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    constant float& silence_coefficient [[buffer(3)]],
    constant float& phase_coherence [[buffer(4)]],
    constant uint& batch_size [[buffer(5)]],
    constant uint& feature_dim [[buffer(6)]],
    uint2 id [[thread_position_in_grid]])
{
    uint batch_idx = id.x;
    uint feature_idx = id.y;
    
    if (batch_idx >= batch_size || feature_idx >= feature_dim) return;
    
    uint index = batch_idx * feature_dim + feature_idx;
    
    // Apply basic exponential activation with icosahedron sigma
    float x = input[index];
    float result = geometric_exp_activation(x, ICOSAHEDRON_SIGMA);
    
    // Calculate field energy for silence factor
    float field_energy = 0.0f;
    for (uint i = 0; i < feature_dim; i++) {
        float val = input[batch_idx * feature_dim + i];
        field_energy += val * val;
    }
    field_energy /= float(feature_dim);
    
    // Apply silence-space dynamics
    float silence_factor = exp(-silence_coefficient * field_energy);
    
    // Calculate batch sum for golden ratio harmonics
    float batch_sum = 0.0f;
    for (uint i = 0; i < feature_dim; i++) {
        batch_sum += input[batch_idx * feature_dim + i];
    }
    
    // Generate golden ratio harmonics
    float harmonics = 0.0f;
    for (int i = 0; i < 5; i++) {
        float harmonic_weight = pow(INV_PHI, i);
        float phase = 2.0f * PI * pow(INV_PHI, i) * batch_sum;
        harmonics += harmonic_weight * cos(phase);
    }
    
    // Combine with silence factor
    result = result * (1.0f + silence_factor * harmonics);
    
    output[index] = result;
}

// Half-precision versions of the activation functions
kernel void tetrahedron_activation_half(
    const device half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    constant float& fire_coefficient [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= length) return;
    
    // Apply basic exponential activation with tetrahedron sigma
    half x = input[id];
    half sigma = half(TETRAHEDRON_SIGMA);
    half result = geometric_exp_activation(x, sigma);
    
    // Calculate field energy (convert to float for accuracy)
    uint group_size = min(THREAD_GROUP_SIZE, length);
    uint offset = (id / group_size) * group_size;
    
    float field_energy = 0.0f;
    for (uint i = 0; i < group_size; i++) {
        float val = float(input[offset + i]);
        field_energy += val * val;
    }
    field_energy /= float(group_size);
    
    // Apply fire element dynamics
    float fire_factor = exp(fire_coefficient * field_energy);
    result *= half(fire_factor);
    
    output[id] = result;
}

// This kernel implements all geometric activations in a single function
// based on the provided activation_type parameter
kernel void unified_geometric_activation(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    constant uint& activation_type [[buffer(3)]], // 0=tetra, 1=cube, 2=dodeca, 3=icosa
    constant float& coefficient [[buffer(4)]],    // Generic coefficient (fire, stability, etc.)
    constant float& phase_coherence [[buffer(5)]], // Only used for icosahedron
    uint id [[thread_position_in_grid]])
{
    if (id >= length) return;
    
    float x = input[id];
    float result = 0.0f;
    float sigma = 1.0f;
    
    // Select sigma based on activation type
    switch (activation_type) {
        case 0: sigma = TETRAHEDRON_SIGMA; break;
        case 1: sigma = CUBE_SIGMA; break;
        case 2: sigma = DODECAHEDRON_SIGMA; break;
        case 3: sigma = ICOSAHEDRON_SIGMA; break;
        default: sigma = 1.0f;
    }
    
    // Apply basic exponential activation
    result = geometric_exp_activation(x, sigma);
    
    // Apply type-specific modifications based on activation_type
    // This is simplified - a full implementation would replicate the
    // functionality of the individual kernels above
    
    output[id] = result;
}

