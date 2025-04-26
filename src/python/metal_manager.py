#!/usr/bin/env python3
"""
Metal shader management for crystalline_mlx.

This module provides utilities for loading and executing Metal shaders
for GPU acceleration of the crystalline consciousness model operations.
"""

import os
import ctypes
import numpy as np
import warnings
from typing import List, Dict, Tuple, Optional, Union

# Import Metal frameworks
try:
    from Metal import MTLCreateSystemDefaultDevice, MTL
    from Foundation import NSError
    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    warnings.warn("PyObjC Metal framework not found. GPU acceleration unavailable.")

class MetalShaderManager:
    """Manager for Metal shaders and execution."""
    
    def __init__(self, shader_dir: str = None):
        """
        Initialize the Metal shader manager.
        
        Args:
            shader_dir: Directory containing Metal shader files
        """
        self.device = None
        self.command_queue = None
        self.libraries = {}
        self.pipelines = {}
        self.shader_dir = shader_dir
        
        # Initialize Metal if available
        if HAS_METAL:
            self.initialize_metal()
        
    def initialize_metal(self) -> bool:
        """Initialize Metal device and command queue."""
        if not HAS_METAL:
            return False
            
        try:
            # Create Metal device
            self.device = MTLCreateSystemDefaultDevice()
            if self.device is None:
                warnings.warn("Failed to create Metal device")
                return False
                
            # Create command queue
            self.command_queue = self.device.newCommandQueue()
            if self.command_queue is None:
                warnings.warn("Failed to create Metal command queue")
                self.device = None
                return False
                
            return True
        except Exception as e:
            warnings.warn(f"Metal initialization failed: {e}")
            self.device = None
            self.command_queue = None
            return False
    
    def load_shader_library(self, shader_path: str, name: str = None) -> bool:
        """
        Load and compile a Metal shader file.
        
        Args:
            shader_path: Path to the Metal shader file
            name: Name to identify the library (defaults to filename)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not HAS_METAL or self.device is None:
            return False
            
        if name is None:
            name = os.path.basename(shader_path).split('.')[0]
            
        try:
            # Read shader source
            with open(shader_path, 'r') as f:
                shader_source = f.read()
                
            # Create compile options
            options = MTL.CompileOptions.alloc().init()
            
            # Compile shader
            library, error = self.device.newLibraryWithSource_options_error_(
                shader_source, options, None)
                
            if library is None:
                warnings.warn(f"Failed to compile shader {name}: {error}")
                return False
                
            self.libraries[name] = library
            return True
        except Exception as e:
            warnings.warn(f"Failed to load shader {name}: {e}")
            return False
    
    def create_compute_pipeline(self, library_name: str, function_name: str, 
                               pipeline_name: str = None) -> bool:
        """
        Create a compute pipeline for a specific shader function.
        
        Args:
            library_name: Name of the shader library
            function_name: Name of the function in the shader
            pipeline_name: Name to identify the pipeline (defaults to function_name)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not HAS_METAL or self.device is None:
            return False
            
        if library_name not in self.libraries:
            warnings.warn(f"Library {library_name} not found")
            return False
            
        if pipeline_name is None:
            pipeline_name = function_name
            
        try:
            library = self.libraries[library_name]
            function = library.newFunctionWithName_(function_name)
            
            if function is None:
                warnings.warn(f"Function {function_name} not found in library {library_name}")
                return False
                
            pipeline_state, error = self.device.newComputePipelineStateWithFunction_error_(
                function, None)
                
            if pipeline_state is None:
                warnings.warn(f"Failed to create pipeline for {function_name}: {error}")
                return False
                
            self.pipelines[pipeline_name] = pipeline_state
            return True
        except Exception as e:
            warnings.warn(f"Failed to create pipeline for {function_name}: {e}")
            return False
    
    def create_buffer(self, data: np.ndarray, length: int = None) -> Optional[any]:
        """
        Create a Metal buffer from numpy data.
        
        Args:
            data: NumPy array containing the data
            length: Length of the buffer in bytes (optional)
            
        Returns:
            Metal buffer object or None if failed
        """
        if not HAS_METAL or self.device is None:
            return None
            
        try:
            if length is None:
                length = data.nbytes
                
            # Create a shared buffer (accessible by CPU and GPU)
            buffer = self.device.newBufferWithLength_options_(
                length, MTL.ResourceStorageModeShared)
                
            if buffer is None:
                warnings.warn("Failed to create Metal buffer")
                return None
                
            # Copy data to buffer
            buffer_ptr = buffer.contents()
            data_ptr = data.ctypes.data_as(ctypes.c_void_p)
            ctypes.memmove(buffer_ptr, data_ptr, length)
            
            return buffer
        except Exception as e:
            warnings.warn(f"Failed to create buffer: {e}")
            return None
    
    def get_buffer_data(self, buffer: any, shape: tuple, dtype=np.float32) -> Optional[np.ndarray]:
        """
        Get data from a Metal buffer.
        
        Args:
            buffer: Metal buffer object
            shape: Shape of the output array
            dtype: Data type of the output array
            
        Returns:
            NumPy array containing the data or None if failed
        """
        if not HAS_METAL or buffer is None:
            return None
            
        try:
            # Calculate size in bytes
            size = np.prod(shape) * np.dtype(dtype).itemsize
            
            # Create output array
            output = np.empty(shape, dtype=dtype)
            
            # Copy data from buffer
            buffer_ptr = buffer.contents()
            output_ptr = output.ctypes.data_as(ctypes.c_void_p)
            ctypes.memmove(output_ptr, buffer_ptr, size)
            
            return output
        except Exception as e:
            warnings.warn(f"Failed to get buffer data: {e}")
            return None
    
    def execute_shader(self, pipeline_name: str, input_buffers: List[any], 
                      output_buffers: List[any], thread_groups: tuple, 
                      threads_per_group: tuple = None) -> bool:
        """
        Execute a compute shader with the given inputs and outputs.
        
        Args:
            pipeline_name: Name of the compute pipeline to use
            input_buffers: List of input Metal buffers
            output_buffers: List of output Metal buffers
            thread_groups: Tuple of (x, y, z) thread groups
            threads_per_group: Tuple of (x, y, z) threads per group
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not HAS_METAL or self.command_queue is None:
            return False
            
        if pipeline_name not in self.pipelines:
            warnings.warn(f"Pipeline {pipeline_name} not found")
            return False
            
        try:
            pipeline_state = self.pipelines[pipeline_name]
            
            # Create default threads per group if not provided
            if threads_per_group is None:
                max_threads = pipeline_state.maxTotalThreadsPerThreadgroup()
                threads_per_group = (max_threads, 1, 1)
                
            # Create command buffer and encoder
            command_buffer = self.command_queue.commandBuffer()
            encoder = command_buffer.computeCommandEncoder()
            
            # Set pipeline state
            encoder.setComputePipelineState_(pipeline_state)
            
            # Set input buffers
            for i, buffer in enumerate(input_buffers):
                encoder.setBuffer_offset_atIndex_(buffer, 0, i)
                
            # Set output buffers
            for i, buffer in enumerate(output_buffers):
                encoder.setBuffer_offset_atIndex_(buffer, 0, len(input_buffers) + i)
                
            # Dispatch thread groups
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                thread_groups, threads_per_group)
                
            # End encoding and commit
            encoder.endEncoding()
            command_buffer.commit()
            command_buffer.waitUntilCompleted()
            
            return True
        except Exception as e:
            warnings.warn(f"Failed to execute shader {pipeline_name}: {e}")
            return False

# Create global instance of shader manager
shader_manager = None

def get_shader_manager(shader_dir: str = None) -> MetalShaderManager:
    """Get or create the global shader manager instance."""
    global shader_manager
    if shader_manager is None:
        shader_manager = MetalShaderManager(shader_dir)
    return shader_manager

