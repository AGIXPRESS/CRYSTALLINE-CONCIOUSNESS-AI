#include <metal_stdlib>
#include <metal_math>
#include <metal_simdgroup>
using namespace metal;

// Constants
constant float PHI = 1.61803398875f; // Golden ratio: (1 + sqrt(5))/2
constant float INV_PHI = 0.61803398875f; // Inverse golden ratio: (sqrt(5) - 1)/2
constant float PI = 3.14159265359f;
constant float TWO_PI = 6.28318530718f;

// Thread group sizes
constant uint THREAD_GROUP_SIZE = 256;
constant uint2 THREAD_GROUP_SIZE_2D = uint2(16, 16);

// Helper function: sigmoid activation
// Moving this BEFORE any kernel that uses it
inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// Helper function to compute phase value through projection
float compute_phase(
    const device float* input,
    const device float* phase_embedding,
    uint batch_idx,
    uint input_dim,
    uint embedding_dim)
{
    // In the original PyTorch implementation:
    // phase = torch.matmul(x, self.phase_embedding.view(-1, 1))
    
    float phase_sum = 0.0f;
    uint limit = min(input_dim, embedding_dim);
    
    for (uint i = 0; i < limit; i++) {
        phase_sum += input[batch_idx * input_dim + i] * phase_embedding[i];
    }
    
    return phase_sum;
}

// Resonance pattern kernel for single-precision
kernel void apply_resonance(
    const device float* input [[buffer(0)]],
    const device float* frequencies [[buffer(1)]],
    const device float* decay_rates [[buffer(2)]],
    const device float* amplitudes [[buffer(3)]],
    const device float* phase_embedding [[buffer(4)]],
    const device float* time_values [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    constant uint& input_dim [[buffer(7)]],
    constant uint& harmonics [[buffer(8)]],
    constant uint& embedding_dim [[buffer(9)]],
    device float* output [[buffer(10)]],
    uint2 id [[thread_position_in_grid]])
{
    uint batch_idx = id.x;
    uint feature_idx = id.y;
    
    if (batch_idx >= batch_size || feature_idx >= input_dim) return;
    
    uint index = batch_idx * input_dim + feature_idx;
    
    // Get input value
    float x_val = input[index];
    
    // Initialize resonance output
    float resonance = 0.0f;
    
    // Calculate phase
    float phase = compute_phase(input, phase_embedding, batch_idx, input_dim, embedding_dim);
    
    // Get time value for this batch
    float time = time_values[batch_idx * 1];  // Assuming time is a single value per batch
    
    // Apply resonance patterns for each harmonic
    for (uint i = 0; i < harmonics; i++) {
        // Convert parameters to appropriate ranges
        float freq = sigmoid(frequencies[i]) * 10.0f;
        float tau = exp(decay_rates[i]);
        
        // Calculate resonance term
        float phi_power = pow(PHI, float(i));
        float harmonic = amplitudes[i] * cos(freq * phi_power * time + phase) * 
                         exp(-(time * time) / (tau * tau));
        
        // Add harmonic to total resonance
        resonance += harmonic * x_val;
    }
    
    // Write output
    output[index] = resonance;
}

// Rest of the original file can follow...

// Half-precision version (keeping this for reference)
kernel void apply_resonance_patterns_half(
    const device half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    const device half* frequencies [[buffer(2)]],
    const device half* decay_rates [[buffer(3)]],
    const device half* amplitudes [[buffer(4)]],
    const device half* phase_embedding [[buffer(5)]],
    const device half* time_values [[buffer(6)]],
    constant uint& batch_size [[buffer(7)]],
    constant uint& input_dim [[buffer(8)]],
    constant uint& harmonics [[buffer(9)]],
    constant uint& embedding_dim [[buffer(10)]],
    uint2 id [[thread_position_in_grid]])
{
    // Same implementation as above, but with half precision
    // ...
}

// Optimized version for better performance
kernel void apply_resonance_patterns_optimized(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    const device float* frequencies [[buffer(2)]],
    const device float* decay_rates [[buffer(3)]],
    const device float* amplitudes [[buffer(4)]],
    const device float* phase_embedding [[buffer(5)]],
    const device float* time_values [[buffer(6)]],
    constant uint& batch_size [[buffer(7)]],
    constant uint& input_dim [[buffer(8)]],
    constant uint& harmonics [[buffer(9)]],
    constant uint& embedding_dim [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    // Optimized implementation...
    // ...
}

// Batch processing version
kernel void batch_apply_resonance(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    const device float* frequencies [[buffer(2)]],
    const device float* decay_rates [[buffer(3)]],
    const device float* amplitudes [[buffer(4)]],
    const device float* phase_embedding [[buffer(5)]],
    const device float* time_values [[buffer(6)]],
    constant uint& batch_size [[buffer(7)]],
    constant uint& input_dim [[buffer(8)]],
    constant uint& harmonics [[buffer(9)]],
    constant uint& embedding_dim [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    // Batch implementation...
    // ...
}
