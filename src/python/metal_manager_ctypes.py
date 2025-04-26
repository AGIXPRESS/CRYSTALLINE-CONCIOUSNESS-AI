#!/usr/bin/env python3
"""
Improved Metal shader management for crystalline_mlx using ctypes.

This module provides utilities for loading and executing Metal shaders
for GPU acceleration of the crystalline consciousness model operations,
using ctypes for direct access to the Metal framework.
"""

import os
import sys
import ctypes
import numpy as np
import warnings
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

# Load required frameworks
try:
    # Load Foundation framework
    foundation = ctypes.cdll.LoadLibrary('/System/Library/Frameworks/Foundation.framework/Foundation')
    
    # Load Metal framework
    metal = ctypes.cdll.LoadLibrary('/System/Library/Frameworks/Metal.framework/Metal')
    
    # Initialize Objective-C runtime
    objc = ctypes.cdll.LoadLibrary('/usr/lib/libobjc.dylib')
    
    # Define required Objective-C functions
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.objc_msgSend.restype = ctypes.c_void_p
    objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    
    HAS_METAL = True
except (OSError, AttributeError) as e:
    warnings.warn(f"Failed to load Metal framework: {e}")
    HAS_METAL = False

class MetalShaderManager:
    """Manager for Metal shaders and execution using ctypes."""
    
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
            # Get MTLCreateSystemDefaultDevice function
            MTLCreateSystemDefaultDevice = ctypes.CDLL(None).MTLCreateSystemDefaultDevice
            MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
            
            # Create Metal device
            self.device = MTLCreateSystemDefaultDevice()
            if not self.device:
                warnings.warn("Failed to create Metal device")
                return False
                
            # Create command queue
            sel_newCommandQueue = objc.sel_registerName(b"newCommandQueue")
            self.command_queue = objc.objc_msgSend(self.device, sel_newCommandQueue)
            if not self.command_queue:
                warnings.warn("Failed to create Metal command queue")
                self.device = None
                return False
                
            print(f"Metal device initialized successfully")
            return True
        except Exception as e:
            warnings.warn(f"Metal initialization failed: {e}")
            self.device = None
            self.command_queue = None
            return False
    
    def get_device_name(self) -> str:
        """Get the name of the Metal device."""
        if not HAS_METAL or not self.device:
            return "No Metal device"
            
        try:
            # Get device name
            sel_name = objc.sel_registerName(b"name")
            name_nsstring = objc.objc_msgSend(self.device, sel_name)
            
            # Convert NSString to Python string
            sel_UTF8String = objc.sel_registerName(b"UTF8String")
            name_ptr = objc.objc_msgSend(name_nsstring, sel_UTF8String)
            
            if name_ptr:
                return ctypes.string_at(name_ptr).decode('utf-8')
            else:
                return "Unknown device"
        except Exception as e:
            warnings.warn(f"Failed to get device name: {e}")
            return "Error getting device name"
    
    def load_shader_library(self, shader_path: str, name: str = None) -> bool:
        """
        Load and compile a Metal shader file.
        
        Args:
            shader_path: Path to the Metal shader file
            name: Name to identify the library (defaults to filename)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not HAS_METAL or not self.device:
            return False
            
        if name is None:
            name = os.path.basename(shader_path).split('.')[0]
            
        try:
            # Read shader source
            with open(shader_path, 'r') as f:
                shader_source = f.read()
                
            # Create NSString from shader source
            NSString = objc.objc_getClass(b"NSString")
            sel_stringWithUTF8String = objc.sel_registerName(b"stringWithUTF8String:")
            source_nsstring = objc.objc_msgSend(NSString, sel_stringWithUTF8String, 
                                              shader_source.encode('utf-8'))
            
            # Create MTLCompileOptions
            MTLCompileOptions = objc.objc_getClass(b"MTLCompileOptions")
            sel_alloc = objc.sel_registerName(b"alloc")
            sel_init = objc.sel_registerName(b"init")
            options = objc.objc_msgSend(objc.objc_msgSend(MTLCompileOptions, sel_alloc), sel_init)
            
            # Create library - this is a simplified approach, in a full implementation
            # we would need to handle errors and use the proper newLibraryWithSource:options:error: method
            sel_newLibraryWithSource = objc.sel_registerName(b"newLibraryWithSource:options:error:")
            library = objc.objc_msgSend(self.device, sel_newLibraryWithSource, 
                                      source_nsstring, options, None)
            
            if not library:
                warnings.warn(f"Failed to compile shader {name}")
                return False
                
            self.libraries[name] = library
            print(f"Successfully loaded {name} shader library")
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
        if not HAS_METAL or not self.device:
            return False
            
        if library_name not in self.libraries:
            warnings.warn(f"Library {library_name} not found")
            return False
            
        if pipeline_name is None:
            pipeline_name = function_name
            
        try:
            library = self.libraries[library_name]
            
            # Get function from library
            sel_newFunctionWithName = objc.sel_registerName(b"newFunctionWithName:")
            function = objc.objc_msgSend(library, sel_newFunctionWithName, 
                                       function_name.encode('utf-8'))
            
            if not function:
                warnings.warn(f"Function {function_name} not found in library {library_name}")
                return False
                
            # Create compute pipeline state
            sel_newComputePipelineStateWithFunction_error = objc.sel_registerName(b"newComputePipelineStateWithFunction:error:")
            
            # Need to properly handle error parameter for production code
            pipeline = objc.objc_msgSend(self.device, sel_newComputePipelineStateWithFunction_error,
                                       function, None)
                
            if not pipeline:
                warnings.warn(f"Failed to create pipeline for {function_name}")
                return False
                
            self.pipelines[pipeline_name] = pipeline
            print(f"Successfully created pipeline for {function_name}")
            return True
        except Exception as e:
            warnings.warn(f"Failed to create pipeline for {function_name}: {e}")
            return False

    def create_buffer(self, data: np.ndarray, length: int = None) -> Optional[int]:
        """
        Create a Metal buffer from numpy data.
        
        Args:
            data: NumPy array containing the data
            length: Length of the buffer in bytes (optional)
            
        Returns:
            Metal buffer object or None if failed
        """
        if not HAS_METAL or not self.device:
            return None
            
        try:
            if length is None:
                length = data.nbytes
            
            # Make sure data is contiguous
            if not data.flags['C_CONTIGUOUS']:
                data = np.ascontiguousarray(data)
            
            # Get data pointer
            data_ptr = data.ctypes.data_as(ctypes.c_void_p)
            
            # Create buffer with shared storage mode
            MTLResourceStorageModeShared = 0  # MTLResourceStorageModeShared enum value
            sel_newBufferWithBytes_length_options = objc.sel_registerName(b"newBufferWithBytes:length:options:")
            
            buffer = objc.objc_msgSend(self.device, sel_newBufferWithBytes_length_options,
                                     data_ptr, ctypes.c_int(length), ctypes.c_int(MTLResourceStorageModeShared))
            
            if not buffer:
                warnings.warn("Failed to create Metal buffer")
                return None
                
            return buffer
        except Exception as e:
            warnings.warn(f"Failed to create buffer: {e}")
            return None

    def get_buffer_data(self, buffer: int, shape: tuple, dtype=np.float32) -> Optional[np.ndarray]:
        """
        Get data from a Metal buffer.
        
        Args:
            buffer: Metal buffer object
            shape: Shape of the output array
            dtype: Data type of the output array
            
        Returns:
            NumPy array containing the data or None if failed
        """
        if not HAS_METAL or not buffer:
            return None
            
        try:
            # Calculate size in bytes
            size = np.prod(shape) * np.dtype(dtype).itemsize
            
            # Get buffer contents
            sel_contents = objc.sel_registerName(b"contents")
            buffer_ptr = objc.objc_msgSend(buffer, sel_contents)
            
            if not buffer_ptr:
                warnings.warn("Failed to get buffer contents")
                return None
            
            # Create numpy array from buffer
            output = np.ctypeslib.as_array(
                ctypes.cast(buffer_ptr, ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))),
                shape=shape
            )
            
            # Make a copy to ensure the data stays valid
            return output.copy()
        except Exception as e:
            warnings.warn(f"Failed to get buffer data: {e}")
            return None

    def execute_shader(self, pipeline_name: str, input_buffers: List[int], 
                      output_buffers: List[int], thread_groups: tuple, 
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
        if not HAS_METAL or not self.command_queue or pipeline_name not in self.pipelines:
            return False
            
        try:
            pipeline_state = self.pipelines[pipeline_name]
            
            # Create command buffer
            sel_commandBuffer = objc.sel_registerName(b"commandBuffer")
            command_buffer = objc.objc_msgSend(self.command_queue, sel_commandBuffer)
            
            # Create compute encoder
            sel_computeCommandEncoder = objc.sel_registerName(b"computeCommandEncoder")
            encoder = objc.objc_msgSend(command_buffer, sel_computeCommandEncoder)
            
            # Set pipeline state
            sel_setComputePipelineState = objc.sel_registerName(b"setComputePipelineState:")
            objc.objc_msgSend(encoder, sel_setComputePipelineState, pipeline_state)
            
            # Set buffers
            sel_setBuffer_offset_atIndex = objc.sel_registerName(b"setBuffer:offset:atIndex:")
            for i, buffer in enumerate(input_buffers):
                objc.objc_msgSend(encoder, sel_setBuffer_offset_atIndex, buffer, 0, i)
                
            for i, buffer in enumerate(output_buffers):
                objc.objc_msgSend(encoder, sel_setBuffer_offset_atIndex, buffer, 0, len(input_buffers) + i)
            
            # Create MTLSize for thread groups
            MTLSizeMake = ctypes.CDLL(None).MTLSizeMake
            MTLSizeMake.restype = ctypes.c_void_p
            MTLSizeMake.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
            
            thread_groups_size = MTLSizeMake(thread_groups[0], thread_groups[1], thread_groups[2])
            
            # Use default threads per group if not provided
            if threads_per_group is None:
                # Get max threads per threadgroup
                sel_maxTotalThreadsPerThreadgroup = objc.sel_registerName(b"maxTotalThreadsPerThreadgroup")
                max_threads = objc.objc_msgSend(pipeline_state, sel_maxTotalThreadsPerThreadgroup)
                threads_per_group = (max_threads, 1, 1)
                
            threads_per_threadgroup_size = MTLSizeMake(threads_per_group[0], threads_per_group[1], threads_per_group[2])
            
            # Dispatch thread groups
            sel_dispatchThreads = objc.sel_registerName(b"dispatchThreadgroups:threadsPerThreadgroup:")
            objc.objc_msgSend(encoder, sel_dispatchThreads, thread_groups_size, threads_per_threadgroup_size)
            
            # End encoding and commit
            sel_endEncoding = objc.sel_registerName(b"endEncoding")
            objc.objc_msgSend(encoder, sel_endEncoding)
            
            sel_commit = objc.sel_registerName(b"commit")
            objc.objc_msgSend(command_buffer, sel_commit)
            
            sel_waitUntilCompleted = objc.sel_registerName

