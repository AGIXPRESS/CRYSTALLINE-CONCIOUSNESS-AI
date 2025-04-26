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
kernel void apply_resonance_patterns(
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
    uint2 id [[thread_position_in_grid]])
{
    uint batch_idx = id.x;
    uint dim_idx = id.y;
    
    if (batch_idx >= batch_size || dim_idx >= input_dim) {
        return;
    }
    
    uint index = batch_idx * input_dim + dim_idx;
    float input_value = input[index];
    
    // Get the time value for this batch element (or default to 1.0)
    float t = (time_values != nullptr) ? time_values[batch_idx] : 1.0f;
    
    // Calculate phase based on input pattern projected to phase space
    float phase = compute_phase(input, phase_embedding, batch_idx, input_dim, embedding_dim);
    
    // Initialize resonance output with zeros
    float resonance_value = 0.0f;
    
    // Generate resonance patterns for each harmonic
    for (uint i = 0; i < harmonics; i++) {
        // Convert frequency and decay rate to appropriate range
        float freq = sigmoid(frequencies[i]) * 10.0f;
        float tau = exp(decay_rates[i]);  // Ensure positive decay rates
        
        // Calculate the resonance term:
        // harmonic = amplitude[i] * cos(freq * phi^i * t + phase) * exp(-(t^2) / (tau^2))
        float phi_power = pow(PHI, float(i));
        float harmonic = amplitudes[i] * cos(freq * phi_power * t + phase) * 
                         exp(-(t * t) / (tau * tau));
        
        // Add this harmonic to the total resonance (multiplied by input)
        resonance_value += harmonic * input_value;
    }
    
    // Store the result
    output[index] = resonance_value;
}

// Helper function: sigmoid activation
inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// Half-precision version of the resonance pattern kernel
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
    uint batch_idx = id.x;
    uint dim_idx = id.y;
    
    if (batch_idx >= batch_size || dim_idx >= input_dim) {
        return;
    }
    
    uint index = batch_idx * input_dim + dim_idx;
    float input_value = float(input[index]); // Convert to float for higher precision in calculations
    
    // Get the time value (or default to 1.0)
    float t = (time_values != nullptr) ? float(time_values[batch_idx]) : 1.0f;
    
    // Calculate phase (using float for accuracy)
    float phase = 0.0f;
    uint limit = min(input_dim, embedding_dim);
    
    for (uint i = 0; i < limit; i++) {
        phase += float(input[batch_idx * input_dim + i]) * float(phase_embedding[i]);
    }
    
    // Initialize resonance output
    float resonance_value = 0.0f;
    
    // Generate resonance patterns for each harmonic
    for (uint i = 0; i < harmonics; i++) {
        // Convert to float for calculations
        float freq = sigmoid(float(frequencies[i])) * 10.0f;
        float tau = exp(float(decay_rates[i]));
        float amplitude = float(amplitudes[i]);
        
        // Calculate the resonance term
        float phi_power = pow(PHI, float(i));
        float harmonic = amplitude * cos(freq * phi_power * t + phase) * 
                         exp(-(t * t) / (tau * tau));
        
        // Add this harmonic to the total resonance
        resonance_value += harmonic * input_value;
    }
    
    // Convert back to half and store
    output[index] = half(resonance_value);
}

// Optimized version using thread groups and shared memory
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
    uint2 id [[thread_position_in_grid]],
    uint2 threadgroup_id [[threadgroup_position_in_grid]],
    uint2 local_id [[thread_position_in_threadgroup]])
{
    // Define shared memory for thread group
    threadgroup float local_embeddings[256]; // Assuming max embedding_dim <= 256
    threadgroup float local_frequencies[8];  // Assuming max harmonics <= 8
    threadgroup float local_decay_rates[8];
    threadgroup float local_amplitudes[8];
    
    // Load shared data (phase embeddings and harmonic parameters)
    if (local_id.x == 0 && local_id.y < embedding_dim) {
        local_embeddings[local_id.y] = phase_embedding[local_id.y];
    }
    
    if (local_id.x == 0 && local_id.y < harmonics) {
        local_frequencies[local_id.y] = frequencies[local_id.y];
        local_decay_rates[local_id.y] = decay_rates[local_id.y];
        local_amplitudes[local_id.y] = amplitudes[local_id.y];
    }
    
    // Ensure all threads have loaded data before proceeding
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    uint batch_idx = id.x;
    uint dim_idx = id.y;
    
    if (batch_idx >= batch_size || dim_idx >= input_dim) {
        return;
    }
    
    uint index = batch_idx * input_dim + dim_idx;
    float input_value = input[index];
    
    // Get the time value
    float t = (time_values != nullptr) ? time_values[batch_idx] : 1.0f;
    
    // Calculate phase using shared memory
    float phase = 0.0f;
    uint limit = min(input_dim, embedding_dim);
    
    for (uint i = 0; i < limit; i++) {
        phase += input[batch_idx * input_dim + i] * local_embeddings[i];
    }
    
    // Initialize resonance output
    float resonance_value = 0.0f;
    
    // Generate resonance patterns using shared memory
    for (uint i = 0; i < harmonics; i++) {
        float freq = sigmoid(local_frequencies[i]) * 10.0f;
        float tau = exp(local_decay_rates[i]);
        
        float phi_power = pow(PHI, float(i));
        float harmonic = local_amplitudes[i] * cos(freq * phi_power * t + phase) * 
                         exp(-(t * t) / (tau * tau));
        
        resonance_value += harmonic * input_value;
    }
    
    // Store the result
    output[index] = resonance_value;
}

// Batch resonance application - processes multiple resonance applications in parallel
kernel void batch_apply_resonance(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& input_dim [[buffer(3)]],
    constant uint& num_resonators [[buffer(4)]],
    constant float* resonator_params [[buffer(5)]], // Packed parameters for all resonators
    uint3 id [[thread_position_in_grid]])
{
    uint batch_idx = id.x;
    uint dim_idx = id.y;
    uint resonator_idx = id.z;
    
    if (batch_idx >= batch_size || dim_idx >= input_dim || resonator_idx >= num_resonators) {
        return;
    }
    
    uint input_index = batch_idx * input_dim + dim_idx;
    uint output_index = (batch_idx * num_resonators + resonator_idx) * input_dim + dim_idx;
    
    float input_value = input[input_index];
    
    // Get resonator parameters from the packed array
    // Format: [freq1, decay1, amp1, freq2, decay2, amp2, ...]
    uint param_offset = resonator_idx * 3; // 3 parameters per resonator
    float frequency = resonator_params[param_offset];
    float decay_rate = resonator_params[param_offset + 1];
    float amplitude = resonator_params[param_offset + 2];
    
    // Apply simple resonance formula
    float t = 1.0f; // Default time
    float resonance = amplitude * cos(frequency * t) * exp(-(t * t) / (decay_rate * decay_rate));
    
    output[output_index] = input_value * resonance;
}

