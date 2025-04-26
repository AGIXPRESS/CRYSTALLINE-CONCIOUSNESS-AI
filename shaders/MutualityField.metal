#include <metal_stdlib>
#include <metal_math>
#include <metal_simdgroup>
using namespace metal;

// Constants
constant float PHI = 1.61803398875f; // Golden ratio: (1 + sqrt(5))/2
// Reserved for future use:
// constant float INV_PHI = 0.61803398875f; // Inverse golden ratio: (sqrt(5) - 1)/2
// constant float PI = 3.14159265359f;

// Convolution Kernel Weights
constant float conv_weights_layer1[8][3][3] = {
    // 8 kernels, each 3x3
    {
        {0.1f, 0.2f, 0.1f},
        {0.2f, 0.0f, 0.2f},
        {0.1f, 0.2f, 0.1f}
    },
    {
        {0.2f, 0.3f, 0.2f},
        {0.1f, 0.0f, 0.1f},
        {0.2f, 0.3f, 0.2f}
    },
    {
        {-0.1f, -0.1f, -0.1f},
        {-0.1f, 1.6f, -0.1f},
        {-0.1f, -0.1f, -0.1f}
    },
    {
        {0.0f, -0.2f, 0.0f},
        {-0.2f, 1.6f, -0.2f},
        {0.0f, -0.2f, 0.0f}
    },
    {
        {-0.2f, 0.0f, 0.2f},
        {-0.3f, 0.0f, 0.3f},
        {-0.2f, 0.0f, 0.2f}
    },
    {
        {0.2f, -0.3f, 0.2f},
        {-0.3f, 0.0f, -0.3f},
        {0.2f, -0.3f, 0.2f}
    },
    {
        {0.1f, 0.1f, 0.1f},
        {0.1f, 0.4f, 0.1f},
        {0.1f, 0.1f, 0.1f}
    },
    {
        {0.0f, 0.1f, 0.0f},
        {0.1f, 0.6f, 0.1f},
        {0.0f, 0.1f, 0.0f}
    }
};

// Reserved for future implementations:
/* 
constant float conv_weights_layer2[4][8][3][3] = {
    // 4 kernels, each taking 8 input channels and producing 3x3 output
    // (Simplified for space - actual weights would be tuned)
    {
        {{0.05f, 0.1f, 0.05f}, {0.1f, 0.4f, 0.1f}, {0.05f, 0.1f, 0.05f}},
        {{0.05f, 0.1f, 0.05f}, {0.1f, 0.0f, 0.1f}, {0.05f, 0.1f, 0.05f}},
        {{0.05f, 0.1f, 0.05f}, {0.1f, 0.0f, 0.1f}, {0.05f, 0.1f, 0.05f}},
        {{0.05f, 0.1f, 0.05f}, {0.1f, 0.0f, 0.1f}, {0.05f, 0.1f, 0.05f}},
        {{0.05f, 0.1f, 0.05f}, {0.1f, 0.0f, 0.1f}, {0.05f, 0.1f, 0.05f}},
        {{0.05f, 0.1f, 0.05f}, {0.1f, 0.0f, 0.1f}, {0.05f, 0.1f, 0.05f}},
        {{0.05f, 0.1f, 0.05f}, {0.1f, 0.0f, 0.1f}, {0.05f, 0.1f, 0.05f}},
        {{0.05f, 0.1f, 0.05f}, {0.1f, 0.0f, 0.1f}, {0.05f, 0.1f, 0.05f}}
    },
    // Additional kernels not fully specified for brevity
    {
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}}
    },
    {
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}}
    },
    {
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}}
    }
};

}
*/

// Reserved for future implementations:
/*
constant float conv_weights_final[4][3][3] = {
    // Final layer: 4 input channels -> 1 output channel
    {
        {0.05f, 0.1f, 0.05f},
        {0.1f, 0.4f, 0.1f},
        {0.05f, 0.1f, 0.05f}
    },
    {
        {0.05f, 0.1f, 0.05f},
        {0.1f, 0.1f, 0.1f},
        {0.05f, 0.1f, 0.05f}
    },
    {
        {0.1f, 0.1f, 0.1f},
        {0.1f, 0.1f, 0.1f},
        {0.1f, 0.1f, 0.1f}
    },
    {
        {0.05f, 0.0f, 0.05f},
        {0.0f, 0.6f, 0.0f},
        {0.05f, 0.0f, 0.05f}
    }
};
*/

// Perform 2D convolution for the first layer (2 input channels -> 8 output channels)
void conv2d_layer1(
    const device float* input, // [batch_size, 2, grid_size, grid_size]
    device float* output,      // [batch_size, 8, grid_size, grid_size]
    uint batch_idx,
    uint grid_size)
{
    // For each output channel
    for (uint out_c = 0; out_c < 8; out_c++) {
        // For each spatial position
        for (uint y = 1; y < grid_size - 1; y++) {
            for (uint x = 1; x < grid_size - 1; x++) {
                float sum = 0.0f;
                
                // For each input channel
                for (uint in_c = 0; in_c < 2; in_c++) {
                    // 3x3 convolution
                    for (int ky = -1; ky <= 1; ky++) {
                        for (int kx = -1; kx <= 1; kx++) {
                            uint in_y = y + ky;
                            uint in_x = x + kx;
                            
                            // Input index: [batch, channel, y, x]
                            uint in_idx = ((batch_idx * 2 + in_c) * grid_size + in_y) * grid_size + in_x;
                            
                            // Kernel weight
                            float weight = conv_weights_layer1[out_c][ky+1][kx+1];
                            
                            sum += input[in_idx] * weight;
                        }
                    }
                }
                
                // Output index: [batch, channel, y, x]
                uint out_idx = ((batch_idx * 8 + out_c) * grid_size + y) * grid_size + x;
                output[out_idx] = sum;
            }
        }
    }
}

// Kernel to reshape a linear input into a 2D grid
kernel void reshape_to_grid(
    const device float* input [[buffer(0)]],   // [batch_size, input_dim]
    device float* output [[buffer(1)]],        // [batch_size, 1, grid_size, grid_size]
    constant uint& batch_size [[buffer(2)]],
    constant uint& input_dim [[buffer(3)]],
    constant uint& grid_size [[buffer(4)]],
    uint2 id [[thread_position_in_grid]])
{
    uint batch_idx = id.x;
    uint grid_idx = id.y;
    
    if (batch_idx >= batch_size || grid_idx >= grid_size * grid_size) {
        return;
    }
    
    uint y = grid_idx / grid_size;
    uint x = grid_idx % grid_size;
    
    // If grid_size*grid_size is larger than input_dim, we need to handle padding
    if (grid_idx < input_dim) {
        // Input is [batch_size, input_dim]
        uint input_idx = batch_idx * input_dim + grid_idx;
        
        // Output is [batch_size, 1, grid_size, grid_size]
        uint output_idx = ((batch_idx * 1 + 0) * grid_size + y) * grid_size + x;
        
        output[output_idx] = input[input_idx];
    } else {
        // Pad with zeros if needed
        uint output_idx = ((batch_idx * 1 + 0) * grid_size + y) * grid_size + x;
        output[output_idx] = 0.0f;
    }
}

// Kernel to create shifted versions of the field (for spatial and temporal shifts)
kernel void create_shifted_fields(
    const device float* input [[buffer(0)]],        // [batch_size, 1, grid_size, grid_size]
    device float* r_shifted [[buffer(1)]],          // Spatially shifted (right/x-direction)
    device float* t_shifted [[buffer(2)]],          // Temporally shifted (down/y-direction)
    constant uint& batch_size [[buffer(3)]],
    constant uint& grid_size [[buffer(4)]],
    uint3 id [[thread_position_in_grid]])
{
    uint batch_idx = id.x;
    uint y = id.y;
    uint x = id.z;
    
    if (batch_idx >= batch_size || y >= grid_size || x >= grid_size) {
        return;
    }
    
    // Input index: [batch, channel, y, x]
            // Directly use output_idx instead of creating an unused input_idx
    
    // Shifted indices - handling boundaries with circular padding
    uint x_next = (x + 1) % grid_size;
    uint y_next = (y + 1) % grid_size;
    
    // Create spatially shifted field (shift in x/r direction)
    uint r_shifted_idx = ((batch_idx * 1 + 0) * grid_size + y) * grid_size + x;
    r_shifted[r_shifted_idx] = input[((batch_idx * 1 + 0) * grid_size + y) * grid_size + x_next];
    
    // Create temporally shifted field (shift in y/t direction)
    uint t_shifted_idx = ((batch_idx * 1 + 0) * grid_size + y) * grid_size + x;
    t_shifted[t_shifted_idx] = input[((batch_idx * 1 + 0) * grid_size + y_next) * grid_size + x];
}

// Kernel to combine shifted fields into interference patterns
kernel void create_interference_patterns(
    const device float* original [[buffer(0)]],     // [batch_size, 1, grid_size, grid_size]
    const device float* r_shifted [[buffer(1)]],    // Spatially shifted field
    const device float* t_shifted [[buffer(2)]],    // Temporally shifted field
    device float* r_interference [[buffer(3)]],     // Output r-interference
    device float* t_interference [[buffer(4)]],     // Output t-interference
    constant uint& batch_size [[buffer(5)]],
    constant uint& grid_size [[buffer(6)]],
    uint3 id [[thread_position_in_grid]])
{
    uint batch_idx = id.x;
    uint y = id.y;
    uint x = id.z;
    
    if (batch_idx >= batch_size || y >= grid_size || x >= grid_size) {
        return;
    }
    
    // Calculate indices
    uint idx = ((batch_idx * 1 + 0) * grid_size + y) * grid_size + x;
    
    // Get values from each field
    float orig_val = original[idx];
    float r_shifted_val = r_shifted[idx];
    float t_shifted_val = t_shifted[idx];
    
    // Create interference patterns by combining field with shifts
    // In the PyTorch code, these are combined and processed through convolutional layers
    // Here we'll store the combined fields, with channel 0 = original, channel 1 = shifted
    uint r_out_idx_orig = ((batch_idx * 2 + 0) * grid_size + y) * grid_size + x;
    uint r_out_idx_shifted = ((batch_idx * 2 + 1) * grid_size + y) * grid_size + x;
    
    r_interference[r_out_idx_orig] = orig_val;
    r_interference[r_out_idx_shifted] = r_shifted_val;
    
    uint t_out_idx_orig = ((batch_idx * 2 + 0) * grid_size + y) * grid_size + x;
    uint t_out_idx_shifted = ((batch_idx * 2 + 1) * grid_size + y) * grid_size + x;
    
    t_interference[t_out_idx_orig] = orig_val;
    t_interference[t_out_idx_shifted] = t_shifted_val;
}

// Process the interference fields through the convolutional network
kernel void process_interference_fields(
    const device float* r_interference [[buffer(0)]],    // [batch_size, 2, grid_size, grid_size]
    const device float* t_interference [[buffer(1)]],    // [batch_size, 2, grid_size, grid_size]
    device float* r_processed [[buffer(2)]],            // [batch_size, 1, grid_size, grid_size]
    device float* t_processed [[buffer(3)]],            // [batch_size, 1, grid_size, grid_size]
    device float* layer1_output_r [[buffer(4)]],        // Intermediate buffer for r conv output
    device float* layer1_output_t [[buffer(5)]],        // Intermediate buffer for t conv output
    constant uint& batch_size [[buffer(6)]],
    constant uint& grid_size [[buffer(7)]],
    uint3 id [[thread_position_in_grid]])
{
    uint batch_idx = id.x;
    
    if (batch_idx >= batch_size) {
        return;
    }
    
    // Note: Using direct averaging instead of a second convolutional layer for simplicity
    // No need for intermediate threadgroup arrays for the second layer
    
    // Apply first convolutional layer
    conv2d_layer1(r_interference, layer1_output_r, batch_idx, grid_size);
    conv2d_layer1(t_interference, layer1_output_t, batch_idx, grid_size);
    
    // Apply second convolutional layer (simplified, in full implementation would be more complex)
    // For now, we'll just pool the outputs of the first layer to get a single channel
    for (uint y = 0; y < grid_size; y++) {
        for (uint x = 0; x < grid_size; x++) {
            float r_sum = 0.0f;
            float t_sum = 0.0f;
            
            // Average across all 8 channels
            for (uint c = 0; c < 8; c++) {
                uint idx = (c * grid_size + y) * grid_size + x;
                r_sum += layer1_output_r[idx];
                t_sum += layer1_output_t[idx];
            }
            
            // Store averaged results
            uint out_idx = ((batch_idx * 1 + 0) * grid_size + y) * grid_size + x;
            r_processed[out_idx] = r_sum / 8.0f;
            t_processed[out_idx] = t_sum / 8.0f;
        }
    }
}

// Combine r and t interference patterns and apply golden ratio modulation
kernel void combine_interference_patterns(
    const device float* r_processed [[buffer(0)]],    // [batch_size, 1, grid_size, grid_size]
    const device float* t_processed [[buffer(1)]],    // [batch_size, 1, grid_size, grid_size]
    const device float* original [[buffer(2)]],       // Original field for interference factor
    device float* mutual_field [[buffer(3)]],         // [batch_size, 1, grid_size, grid_size]
    constant uint& batch_size [[buffer(4)]],
    constant uint& grid_size [[buffer(5)]],
    constant float& interference_scale [[buffer(6)]],
    uint3 id [[thread_position_in_grid]])
{
    uint batch_idx = id.x;
    uint y = id.y;
    uint x = id.z;
    
    if (batch_idx >= batch_size || y >= grid_size || x >= grid_size) {
        return;
    }
    
    // Calculate indices
    uint idx = ((batch_idx * 1 + 0) * grid_size + y) * grid_size + x;
    
    // Combine r and t interference patterns (average)
    float mutual_value = (r_processed[idx] + t_processed[idx]) / 2.0f;
    
    // Calculate golden ratio modulation
    // In the PyTorch code: interference_factor = torch.sin(self.phi * torch.mean(field, dim=[2, 3], keepdim=True))
    // We'll approximate by calculating the mean for the entire field per batch
    float field_mean = 0.0f;
    for (uint j = 0; j < grid_size; j++) {
        for (uint i = 0; i < grid_size; i++) {
            uint mean_idx = ((batch_idx * 1 + 0) * grid_size + j) * grid_size + i;
            field_mean += original[mean_idx];
        }
    }
    field_mean /= (grid_size * grid_size);
    
    // Apply golden ratio modulation
    float interference_factor = sin(PHI * field_mean);
    mutual_value = mutual_value * (1.0f + interference_scale * interference_factor);
    
    // Store result
    mutual_field[idx] = mutual_value;
}

// Implement persistence equation with decay
kernel void apply_persistence(
    const device float* mutual_field [[buffer(0)]],    // Current mutual field
    device float* persistence_state [[buffer(1)]],     // Persistent state to update
    constant uint& batch_size [[buffer(2)]],
    constant uint& grid_size [[buffer(3)]],
    constant float& decay_rate [[buffer(4)]],          // λ in equation
    constant float& dt [[buffer(5)]],                  // Time step differential
    uint3 id [[thread_position_in_grid]])
{
    uint batch_idx = id.x;
    uint y = id.y;
    uint x = id.z;
    
    if (batch_idx >= batch_size || y >= grid_size || x >= grid_size) {
        return;
    }
    
    // Calculate index
    uint idx = ((batch_idx * 1 + 0) * grid_size + y) * grid_size + x;
    
    // Apply persistence function
    // P_crystal(r, t → ∞) = ∫₀^∞ Ξ_mutual(r, τ) × e^(-λ(t-τ)) dτ
    // In discretized form, this is:
    // persistence_state = mutual_field + decay_factor * persistence_state
    
    float decay_factor = exp(-decay_rate * dt);
    persistence_state[idx] = mutual_field[idx] + decay_factor * persistence_state[idx];
}

// Flatten grid back to vector format
kernel void flatten_to_vector(
    const device float* grid_input [[buffer(0)]],     // [batch_size, 1, grid_size, grid_size]
    device float* vector_output [[buffer(1)]],        // [batch_size, output_dim]
    constant uint& batch_size [[buffer(2)]],
    constant uint& grid_size [[buffer(3)]],
    constant uint& output_dim [[buffer(4)]],          // May be smaller than grid_size^2
    uint2 id [[thread_position_in_grid]])
{
    uint batch_idx = id.x;
    uint out_idx = id.y;
    
    if (batch_idx >= batch_size || out_idx >= output_dim) {
        return;
    }
    
    // If output_dim is smaller than grid_size^2, we only take the first output_dim elements
    if (out_idx < grid_size * grid_size) {
        uint y = out_idx / grid_size;
        uint x = out_idx % grid_size;
        
        uint grid_idx = ((batch_idx * 1 + 0) * grid_size + y) * grid_size + x;
        vector_output[batch_idx * output_dim + out_idx] = grid_input[grid_idx];
    } else {
        // Pad with zeros if output_dim > grid_size^2
        vector_output[batch_idx * output_dim + out_idx] = 0.0f;
    }
}

// Half-precision versions of the main kernels can be implemented here.
// For example, we could provide half-precision versions of the kernels above
// to improve performance on devices that benefit from half-precision operations.
