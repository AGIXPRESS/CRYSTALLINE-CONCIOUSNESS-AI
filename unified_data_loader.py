#!/usr/bin/env python3
"""
Unified Data Loader for Crystalline Consciousness AI Project.

This module provides a versatile data loading system for the Crystalline Consciousness AI project,
capable of processing multiple file types and preparing them for training and inference.
"""

import os
import sys
from pathlib import Path
import logging
import hashlib
import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, BinaryIO

# Try to import optional dependencies
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import mlx
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UnifiedDataLoader:
    """
    A unified data loader for the crystalline consciousness AI project.
    
    This loader handles various file formats including TXT, PDF, SVG, MERMAID, PY, CSV, 
    PNG, MD, TSX, XML, and JSON. It prepares the data for processing with the crystalline
    consciousness framework.
    
    The loader can optionally use caching to speed up data loading and processing,
    and is optimized for Metal/MLX acceleration on Apple Silicon hardware.
    """
    
    SUPPORTED_FORMATS = [
        "txt", "pdf", "svg", "mermaid", "py", "csv", "png", 
        "md", "tsx", "xml", "json"
    ]
    
    def __init__(
        self, 
        data_dir: str = None,
        batch_size: int = 32,
        use_metal: bool = True,
        max_load_size: int = None,
        verbose: bool = False,
        enable_cache: bool = False,
        cache_dir: str = None
    ):
        """
        Initialize the unified data loader.
        
        Args:
            data_dir: Directory containing the training data.
            batch_size: Batch size for data loading.
            use_metal: Whether to utilize Metal/MLX for accelerated processing.
            max_load_size: Maximum size to load (in MB) per file type.
            verbose: Enable verbose logging.
            enable_cache: Whether to enable caching of processed data.
            cache_dir: Directory for cache storage. If None, a 'cache' subdirectory in data_dir is used.
        """
        self.data_dir = data_dir if data_dir else os.path.join(os.getcwd(), 'data')
        self.batch_size = batch_size
        self.use_metal = use_metal and HAS_MLX
        self.max_load_size = max_load_size
        self.verbose = verbose
        self.enable_cache = enable_cache
        
        # Data containers
        self.data_files = {fmt: [] for fmt in self.SUPPORTED_FORMATS}
        self.loaded_data = {}
        self.processed_data = None
        
        # Cache setup
        if self.enable_cache:
            self.cache_dir = cache_dir
            self._initialize_cache()
        else:
            self.cache_dir = None
        
        # Check Metal/MLX availability for GPU acceleration
        if self.use_metal:
            if HAS_MLX and hasattr(mx, 'metal') and mx.metal.is_available():
                logger.info("Metal acceleration enabled for data loading")
            else:
                logger.warning("Metal acceleration requested but not available. Falling back to CPU.")
                self.use_metal = False
        
        if not self.use_metal:
            logger.info("Using CPU for data loading")
            
        # Validate the data directory exists
        if not os.path.exists(self.data_dir):
            logger.warning(f"Data directory not found: {self.data_dir}")
            os.makedirs(self.data_dir, exist_ok=True)
            logger.info(f"Created data directory: {self.data_dir}")

    def _initialize_cache(self) -> None:
        """
        Initialize the cache system.
        
        Sets up the cache directory and basic cache tracking.
        """
        # Determine cache directory location
        if self.cache_dir is None:
            self.cache_dir = os.path.join(self.data_dir, 'cache')
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Created cache directory: {self.cache_dir}")
        
        # Create/check cache metadata file
        self.cache_meta_path = os.path.join(self.cache_dir, 'cache_meta.json')
        if os.path.exists(self.cache_meta_path):
            try:
                with open(self.cache_meta_path, 'r') as f:
                    self.cache_meta = json.load(f)
                logger.info(f"Loaded existing cache metadata with {len(self.cache_meta)} entries")
            except Exception as e:
                logger.warning(f"Error loading cache metadata: {e}")
                self.cache_meta = {}
        else:
            self.cache_meta = {}
            self._save_cache_metadata()
            logger.info("Initialized new cache metadata")
    
    def _cache_key(self, file_path: str, params: Optional[Dict] = None) -> str:
        """
        Generate a unique cache key for a file path and optional parameters.
        
        Args:
            file_path: Path to the file being processed
            params: Optional parameters that affect processing
            
        Returns:
            A hash string to use as cache key
        """
        # Get file modification time for invalidation checking
        try:
            mtime = os.path.getmtime(file_path)
        except OSError:
            mtime = 0
        
        # Create a string with path and modification time
        key_content = f"{file_path}|{mtime}"
        
        # Add parameters if provided
        if params:
            param_str = json.dumps(params, sort_keys=True)
            key_content += f"|{param_str}"
        
        # Hash the key content
        return hashlib.md5(key_content.encode()).hexdigest()
    
    def _save_cache_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            with open(self.cache_meta_path, 'w') as f:
                json.dump(self.cache_meta, f)
        except Exception as e:
            logger.warning(f"Error saving cache metadata: {e}")
    
    def _save_to_cache(self, data: Any, cache_key: str) -> bool:
        """
        Save data to cache.
        
        Args:
            data: The data to cache
            cache_key: Unique identifier for the cached data
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enable_cache:
            return False
        
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.npy")
        
        try:
            # Handle different data types
            if HAS_MLX and isinstance(data, mx.array):
                # Use MLX's save function for MLX arrays
                mx.save(cache_file, data)
            elif HAS_TORCH and isinstance(data, torch.Tensor):
                # Convert PyTorch tensor to numpy and save
                np.savez_compressed(cache_file, data=data.cpu().numpy())
            elif isinstance(data, np.ndarray):
                # Save numpy array directly
                np.save(cache_file, data)
            else:
                # For other types, use numpy's object saving
                np.save(cache_file, data)
            
            # Update metadata
            self.cache_meta[cache_key] = {
                'created': time.time(),
                'file': cache_file,
                'type': str(type(data))
            }
            self._save_cache_metadata()
            
            logger.debug(f"Saved data to cache with key: {cache_key}")
            return True
            
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
            return False
    
    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        """
        Load data from cache.
        
        Args:
            cache_key: Unique identifier for the cached data
            
        Returns:
            The cached data or None if not found or invalid
        """
        # 1. Basic validation - exit immediately if conditions aren't met
        if not self.enable_cache:
            return None
        
        if cache_key not in self.cache_meta:
            logger.debug(f"Cache key not found in metadata: {cache_key}")
            return None
        
        # 2. Critical file existence check
        try:
            # Get cache file path from metadata
            cache_file = self.cache_meta[cache_key]['file']
            # Get absolute path for consistent checking
            cache_file_abs = os.path.abspath(cache_file)
            
            # Add small delay to ensure filesystem sync (helps with test reliability)
            time.sleep(0.05)
            
            # Check if file exists with os.path.exists
            if not os.path.exists(cache_file_abs):
                logger.warning(f"Cache file does not exist: {cache_file_abs}")
                # Clean up metadata for missing file
                if cache_key in self.cache_meta:
                    del self.cache_meta[cache_key]
                    self._save_cache_metadata()
                return None
                
            # Ensure it's actually a file (not directory)
            if not os.path.isfile(cache_file_abs):
                logger.warning(f"Cache path is not a file: {cache_file_abs}")
                # Remove invalid metadata
                if cache_key in self.cache_meta:
                    del self.cache_meta[cache_key]
                    self._save_cache_metadata()
                return None
        except Exception as e:
            logger.warning(f"Error checking cache file existence: {e}")
            # Clean up potentially corrupt metadata
            if cache_key in self.cache_meta:
                del self.cache_meta[cache_key]
                self._save_cache_metadata()
            return None
        
        # 3. File exists at this point - try to load it
        try:
            # Final verification immediately before loading
            # This catches files deleted between our initial check and now
            if not os.path.exists(cache_file_abs):
                logger.warning(f"Cache file disappeared after initial check: {cache_file_abs}")
                if cache_key in self.cache_meta:
                    del self.cache_meta[cache_key]
                    self._save_cache_metadata()
                return None
                
            # For MLX arrays, try to use mx.load first
            if HAS_MLX and hasattr(mx, 'load'):
                try:
                    # Try to load as MLX array
                    result = mx.load(cache_file_abs)
                    logger.debug(f"Loaded MLX data from cache with key: {cache_key}")
                    return result
                except Exception as e:
                    logger.debug(f"MLX load failed, falling back to numpy: {e}")
                    # Fall back to numpy if MLX loading fails
                    
                    # Verify file still exists after MLX load failure
                    if not os.path.exists(cache_file_abs):
                        logger.warning(f"Cache file disappeared during MLX load attempt")
                        if cache_key in self.cache_meta:
                            del self.cache_meta[cache_key]
                            self._save_cache_metadata()
                        return None
            
            # Load with numpy - extract the actual array data
            try:
                # Final check immediately before numpy load attempt
                if not os.path.exists(cache_file_abs):
                    logger.warning(f"Cache file disappeared before numpy load")
                    if cache_key in self.cache_meta:
                        del self.cache_meta[cache_key]
                        self._save_cache_metadata()
                    return None
                    
                # Check file extension to determine how to load it
                if cache_file_abs.endswith('.npy'):
                    # For .npy files, np.load directly returns the array
                    array_data = np.load(cache_file_abs, allow_pickle=True)
                elif cache_file_abs.endswith('.npz'):
                    # For .npz files (backward compatibility), extract the array from the container
                    with np.load(cache_file_abs, allow_pickle=True) as data_dict:
                        # Get the list of array names in the file
                        array_names = data_dict.files
                        
                        if not array_names:
                            logger.warning(f"No arrays found in cached file: {cache_file}")
                            return None
                        
                        # Extract the array data - prioritize our standard naming convention
                        if 'arr_0' in array_names:
                            # Extract the actual array and make a copy
                            array_data = data_dict['arr_0'].copy()
                        elif 'data' in array_names:  # For backward compatibility
                            array_data = data_dict['data'].copy()
                        else:
                            # Take the first array found
                            array_data = data_dict[array_names[0]].copy()
                else:
                    logger.warning(f"Unknown cache file format: {cache_file_abs}")
                    # Clean up metadata for unknown format
                    del self.cache_meta[cache_key]
                    self._save_cache_metadata()
                    return None
            except Exception as e:
                logger.warning(f"Error extracting array from cache file: {e}")
                # Clean up metadata for extraction errors
                del self.cache_meta[cache_key]
                self._save_cache_metadata()
                return None
            
            # If we're using Metal/MLX and have numpy data, convert it
            if self.use_metal and HAS_MLX:
                result = mx.array(array_data)
            else:
                result = array_data
                
            logger.debug(f"Loaded numpy data from cache with key: {cache_key}")
            return result
            
        except Exception as e:
            logger.warning(f"Error loading from cache: {e}")
            # Clean up invalid cache files
            del self.cache_meta[cache_key]
            self._save_cache_metadata()
            if os.path.exists(cache_file_abs):
                try:
                    os.remove(cache_file_abs)
                except OSError:
                    pass
            return None
        
    
    def scan_directory(self, specific_dir: str = None) -> Dict[str, List[str]]:
        """
        Scan the specified directory and catalog files by format.
        
        Args:
            specific_dir: Optional specific directory to scan instead of the default.
            
        Returns:
            Dictionary mapping file formats to lists of file paths.
        """
        target_dir = specific_dir if specific_dir else self.data_dir
        logger.info(f"Scanning directory: {target_dir}")
        
        for root, _, files in os.walk(target_dir):
            for filename in files:
                # Get file extension (lowercase)
                ext = filename.split('.')[-1].lower()
                
                # If it's a supported format, add to the appropriate list
                if ext in self.SUPPORTED_FORMATS:
                    file_path = os.path.join(root, filename)
                    self.data_files[ext].append(file_path)
                    
                    if self.verbose:
                        logger.debug(f"Found {ext} file: {file_path}")
        
        # Log count of files found by type
        for fmt, files in self.data_files.items():
            logger.info(f"Found {len(files)} {fmt.upper()} files")
            
        return self.data_files
    
    def load_data(self, file_types: List[str] = None) -> Dict[str, Any]:
        """
        Load data from the cataloged files.
        
        Args:
            file_types: Optional list of file types to load. If None, loads all supported types.
            
        Returns:
            Dictionary containing loaded data by file type.
        """
        types_to_load = file_types if file_types else self.SUPPORTED_FORMATS
        logger.info(f"Loading data for formats: {', '.join(types_to_load)}")
        
        for fmt in types_to_load:
            if fmt not in self.SUPPORTED_FORMATS:
                logger.warning(f"Unsupported format: {fmt}")
                continue
                
            files = self.data_files.get(fmt, [])
            if not files:
                logger.info(f"No {fmt.upper()} files to load")
                continue
                
            logger.info(f"Loading {len(files)} {fmt.upper()} files...")
            
            # Load based on file type
            if fmt == "txt" or fmt == "md" or fmt == "py":
                self._load_text_files(fmt, files)
            elif fmt == "csv":
                self._load_csv_files(files)
            elif fmt == "json" or fmt == "xml":
                self._load_structured_files(fmt, files)
            elif fmt == "png":
                self._load_image_files(files)
            elif fmt == "pdf":
                self._load_pdf_files(files)
            elif fmt == "svg" or fmt == "mermaid" or fmt == "tsx":
                self._load_graph_files(fmt, files)
            else:
                logger.warning(f"Loader for {fmt} not implemented yet")
        
        return self.loaded_data
    
    def _load_text_files(self, fmt: str, files: List[str]) -> None:
        """Load text-based files (txt, md, py)."""
        loaded_texts = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    loaded_texts.append({
                        'path': file_path,
                        'content': content,
                        'size': len(content)
                    })
                    
                    if self.verbose:
                        logger.debug(f"Loaded {fmt} file: {file_path} ({len(content)} chars)")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        
        self.loaded_data[fmt] = loaded_texts
        logger.info(f"Loaded {len(loaded_texts)} {fmt.upper()} files")
    
    def _load_csv_files(self, files: List[str]) -> None:
        """Load CSV files using pandas if available."""
        if not HAS_PANDAS:
            logger.warning("Pandas not available for CSV loading. Using basic CSV parsing.")
            # Fallback to basic CSV loading
            loaded_csv = []
            
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Basic CSV parsing - split by lines and commas
                        lines = content.strip().split('\n')
                        data = [line.split(',') for line in lines]
                        
                        loaded_csv.append({
                            'path': file_path,
                            'content': content,
                            'data': data,
                            'rows': len(data),
                            'cols': len(data[0]) if data else 0
                        })
                        
                        if self.verbose:
                            logger.debug(f"Loaded CSV file: {file_path} (rows: {len(data)})")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")
            
            self.loaded_data["csv"] = loaded_csv
            logger.info(f"Loaded {len(loaded_csv)} CSV files using basic parsing")
            return
            
        loaded_dataframes = []
        
        for file_path in files:
            try:
                # Use on_bad_lines='skip' to handle malformed rows (instead of error_bad_lines=False)
                df = pd.read_csv(file_path, on_bad_lines='skip')
                loaded_dataframes.append({
                    'path': file_path,
                    'dataframe': df,
                    'shape': df.shape
                })
                
                if self.verbose:
                    logger.debug(f"Loaded CSV file: {file_path} (shape: {df.shape})")
            except pd.errors.ParserError as e:
                # For more severe parsing errors, try with Python engine
                logger.warning(f"CSV parse error with default engine, trying Python engine: {file_path}")
                try:
                    df = pd.read_csv(file_path, engine='python', on_bad_lines='skip')
                    loaded_dataframes.append({
                        'path': file_path,
                        'dataframe': df,
                        'shape': df.shape,
                        'engine': 'python'  # Mark that we used Python engine
                    })
                    logger.info(f"Loaded CSV file with Python engine: {file_path} (shape: {df.shape})")
                except Exception as inner_e:
                    logger.error(f"Failed to load CSV with Python engine {file_path}: {str(inner_e)}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        
        self.loaded_data["csv"] = loaded_dataframes
        logger.info(f"Loaded {len(loaded_dataframes)} CSV files")
    
    def _load_structured_files(self, fmt: str, files: List[str]) -> None:
        """Load structured data files (JSON, XML)."""
        loaded_structures = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse the content according to format if possible
                parsed_data = None
                if fmt == "json":
                    try:
                        import json
                        parsed_data = json.loads(content)
                    except Exception as e:
                        logger.warning(f"Failed to parse JSON in {file_path}: {str(e)}")
                elif fmt == "xml":
                    try:
                        import xml.etree.ElementTree as ET
                        parsed_data = ET.fromstring(content)
                    except Exception as e:
                        logger.warning(f"Failed to parse XML in {file_path}: {str(e)}")
                
                loaded_structures.append({
                    'path': file_path,
                    'content': content,
                    'parsed': parsed_data,
                    'size': len(content)
                })
                
                if self.verbose:
                    logger.debug(f"Loaded {fmt} file: {file_path} ({len(content)} chars)")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        
        self.loaded_data[fmt] = loaded_structures
        logger.info(f"Loaded {len(loaded_structures)} {fmt.upper()} files")
    
    def _load_image_files(self, files: List[str]) -> None:
        """Load image files using PIL if available, otherwise just catalog them."""
        loaded_images = []
        
        try:
            from PIL import Image
            has_pil = True
        except ImportError:
            has_pil = False
            logger.warning("PIL not available for image loading. Images will be cataloged only.")
        
        for file_path in files:
            try:
                image_data = {
                    'path': file_path
                }
                
                if has_pil:
                    img = Image.open(file_path)
                    image_data.update({
                        'image': img,
                        'size': img.size,
                        'mode': img.mode
                    })
                    
                    if self.verbose:
                        logger.debug(f"Loaded PNG file: {file_path} (size: {img.size}, mode: {img.mode})")
                
                loaded_images.append(image_data)
                
            except Exception as e:
                logger.error(f"Error loading image {file_path}: {str(e)}")
        
        self.loaded_data["png"] = loaded_images
        if has_pil:
            logger.info(f"Loaded {len(loaded_images)} PNG files")
        else:
            logger.info(f"Cataloged {len(loaded_images)} PNG files (loading deferred)")
    
    def _load_pdf_files(self, files: List[str]) -> None:
        """Load PDF files using PyPDF2 if available, otherwise just catalog them."""
        loaded_pdfs = []
        
        try:
            import PyPDF2
            has_pypdf2 = True
        except ImportError:
            has_pypdf2 = False
            logger.warning("PyPDF2 not available for PDF loading. PDFs will be cataloged only.")
        
        for file_path in files:
            try:
                pdf_data = {
                    'path': file_path
                }
                
                if has_pypdf2:
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        num_pages = len(pdf_reader.pages)
                        
                        # Extract text from first few pages (as example)
                        first_pages_text = []
                        for i in range(min(3, num_pages)):
                            page = pdf_reader.pages[i]
                            first_pages_text.append(page.extract_text())
                        
                        pdf_data.update({
                            'num_pages': num_pages,
                            'sample_text': first_pages_text
                        })
                        
                        if self.verbose:
                            logger.debug(f"Loaded PDF file: {file_path} ({num_pages} pages)")
                
                loaded_pdfs.append(pdf_data)
                
            except Exception as e:
                logger.error(f"Error loading PDF {file_path}: {str(e)}")
        
        self.loaded_data["pdf"] = loaded_pdfs
        if has_pypdf2:
            logger.info(f"Loaded {len(loaded_pdfs)} PDF files")
        else:
            logger.info(f"Cataloged {len(loaded_pdfs)} PDF files (loading deferred)")
    
    def _load_graph_files(self, fmt: str, files: List[str]) -> None:
        """
        Load graph-based files (SVG, MERMAID, TSX).
        
        For SVG files, we'll try to parse them with the xml library.
        For MERMAID files, we'll load them as specialized text.
        For TSX files, we'll process them as React component templates.
        """
        loaded_graphs = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                graph_data = {
                    'path': file_path,
                    'content': content,
                    'size': len(content)
                }
                
                # Format-specific processing
                if fmt == "svg":
                    try:
                        import xml.etree.ElementTree as ET
                        parsed_svg = ET.fromstring(content)
                        graph_data['parsed'] = parsed_svg
                        # Extract dimensions if available
                        width = parsed_svg.get('width')
                        height = parsed_svg.get('height')
                        if width and height:
                            graph_data['dimensions'] = (width, height)
                    except Exception as e:
                        logger.warning(f"Failed to parse SVG in {file_path}: {str(e)}")
                
                elif fmt == "mermaid":
                    # Identify mermaid diagram type
                    mermaid_type = "unknown"
                    if content.strip().startswith("graph "):
                        mermaid_type = "flowchart"
                    elif content.strip().startswith("sequenceDiagram"):
                        mermaid_type = "sequence"
                    elif content.strip().startswith("classDiagram"):
                        mermaid_type = "class"
                    graph_data['diagram_type'] = mermaid_type
                
                elif fmt == "tsx":
                    # Basic component detection for React/TypeScript
                    is_component = "React.Component" in content or "function" in content and "return (" in content
                    graph_data['is_component'] = is_component
                
                loaded_graphs.append(graph_data)
                
                if self.verbose:
                    logger.debug(f"Loaded {fmt} file: {file_path} ({len(content)} chars)")
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        
        self.loaded_data[fmt] = loaded_graphs
        logger.info(f"Loaded {len(loaded_graphs)} {fmt.upper()} files")

    def process_data(self, file_types: List[str] = None) -> Union[np.ndarray, Any]:
        """
        Process the loaded data and convert it to tensors for model training/inference.
        
        Args:
            file_types: Optional list of file types to process. If None, processes all loaded types.
            
        Returns:
            Processed data in the appropriate tensor format (MLX, PyTorch, or NumPy).
        """
        if not self.loaded_data:
            logger.warning("No data loaded. Call load_data() first.")
            return None

        # Determine which formats to process
        types_to_process = file_types if file_types else list(self.loaded_data.keys())
        logger.info(f"Processing data for formats: {', '.join(types_to_process)}")

        # For text-based formats, combine all content
        all_text = []
        for fmt in types_to_process:
            if fmt not in self.loaded_data:
                continue
                
            if fmt in ["txt", "md", "py", "json", "xml", "svg", "mermaid", "tsx"]:
                # Extract text content from these formats
                for item in self.loaded_data[fmt]:
                    if 'content' in item:
                        all_text.append(item['content'])
            elif fmt == "csv":
                # Handle CSV data - convert to strings for now
                for item in self.loaded_data[fmt]:
                    if 'content' in item:
                        all_text.append(item['content'])
                    elif 'dataframe' in item and HAS_PANDAS:
                        # Convert DataFrame to string representation
                        all_text.append(str(item['dataframe']))
            elif fmt == "pdf":
                # Extract text from PDFs if available
                for item in self.loaded_data[fmt]:
                    if 'sample_text' in item:
                        all_text.append("\n".join(item['sample_text']))
            elif fmt == "png":
                # Handle image data if PIL is available
                for item in self.loaded_data[fmt]:
                    if 'image' in item:
                        # We just note the image dimensions for now
                        all_text.append(f"Image: {item['path']} - Size: {item.get('size', 'unknown')}")

        # Convert to numpy array first
        if all_text:
            # Very simple processing - just convert to character indices
            char_set = set(''.join(all_text))
            char_to_idx = {c: i for i, c in enumerate(sorted(char_set))}
            
            # Convert first text as example
            if all_text[0]:
                indices = [char_to_idx[c] for c in all_text[0][:1000]]  # Limit size for example
                data = np.array(indices, dtype=np.float32).reshape(-1, 1)
                
                # Convert to appropriate format based on backend
                if self.use_metal and HAS_MLX:
                    processed = mx.array(data)
                    logger.info(f"Processed data to MLX array of shape {processed.shape}")
                elif HAS_TORCH:
                    processed = torch.tensor(data)
                    logger.info(f"Processed data to PyTorch tensor of shape {processed.shape}")
                else:
                    processed = data
                    logger.info(f"Processed data to NumPy array of shape {processed.shape}")
                
                self.processed_data = processed
                return processed
        
        logger.warning("No textual data available for processing")
        return None
    
    def get_batch(self, batch_idx: int = 0) -> Union[np.ndarray, Any]:
        """
        Get a specific batch from the processed data.
        
        Args:
            batch_idx: Index of the batch to retrieve.
            
        Returns:
            Batch data in the appropriate format.
        """
        if self.processed_data is None:
            logger.warning("No processed data available. Call process_data() first.")
            return None
        
        # Calculate batch indices
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.processed_data))
        
        # Extract the batch
        if isinstance(self.processed_data, np.ndarray):
            return self.processed_data[start_idx:end_idx]
        elif HAS_TORCH and isinstance(self.processed_data, torch.Tensor):
            return self.processed_data[start_idx:end_idx]
        elif HAS_MLX and isinstance(self.processed_data, mx.array):
            return self.processed_data[start_idx:end_idx]
        else:
            logger.warning(f"Unsupported data type: {type(self.processed_data)}")
            return None
    
    def cleanup(self) -> None:
        """
        Clean up resources and free memory.
        """
        logger.info("Cleaning up data loader resources")
        self.loaded_data = {}
        self.processed_data = None
        self.data_files = {fmt: [] for fmt in self.SUPPORTED_FORMATS}


# Example usage
if __name__ == "__main__":
    # Create a data loader instance
    loader = UnifiedDataLoader(
        data_dir='Training Data (PROJECT HISTORY ALL)',
        batch_size=64,
        use_metal=True,
        verbose=True
    )
    
    # Scan directory for files
    loader.scan_directory()
    
    # Load text and CSV data
    loader.load_data(file_types=["txt", "csv", "py"])
    
    # Process the data
    processed_data = loader.process_data()
    
    # Get a batch
    batch = loader.get_batch(0)
    if batch is not None:
        print(f"Batch shape: {batch.shape}")
    
    # Clean up
    loader.cleanup()

