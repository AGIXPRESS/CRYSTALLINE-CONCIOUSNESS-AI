#!/usr/bin/env python3
"""
Tests for the UnifiedDataLoader class with focus on caching and MLX acceleration.

This test suite verifies:
1. Cache initialization and management
2. MLX/Metal acceleration when available
3. CSV parsing with malformed data
4. Benchmarking loading times with and without cache
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path
import pytest
import numpy as np

# Get the project root directory (parent of 'tests')
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the UnifiedDataLoader
from unified_data_loader import UnifiedDataLoader, logger

# Check for optional dependencies
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


# Create a fixture for temporary test directory
@pytest.fixture
def test_data_dir():
    """Create a temporary directory for test data and cache."""
    temp_dir = tempfile.mkdtemp()
    
    # Create a few test files
    # 1. Regular text file
    with open(os.path.join(temp_dir, "test.txt"), "w") as f:
        f.write("This is a test file for the UnifiedDataLoader.\n")
        f.write("It contains text that will be processed and cached.\n")
        f.write("The cache should speed up subsequent loading operations.\n")
    
    # 2. Python file
    with open(os.path.join(temp_dir, "test.py"), "w") as f:
        f.write("def test_function():\n")
        f.write("    print('This is a test Python file')\n")
        f.write("    return True\n")
    
    # 3. Well-formed CSV file
    with open(os.path.join(temp_dir, "valid.csv"), "w") as f:
        f.write("id,name,value\n")
        f.write("1,item1,100\n")
        f.write("2,item2,200\n")
        f.write("3,item3,300\n")
    
    # 4. Malformed CSV file (missing column)
    with open(os.path.join(temp_dir, "malformed.csv"), "w") as f:
        f.write("id,name,value\n")
        f.write("1,item1,100\n")
        f.write("2,item2\n")  # Missing value
        f.write("3,item3,300\n")
    
    yield temp_dir
    
    # Cleanup after tests
    shutil.rmtree(temp_dir)


def test_cache_initialization(test_data_dir):
    """Test cache directory initialization and metadata creation."""
    # Create loader with caching enabled
    loader = UnifiedDataLoader(
        data_dir=test_data_dir,
        enable_cache=True,
        verbose=True
    )
    
    # Check that cache directory was created
    cache_dir = os.path.join(test_data_dir, "cache")
    assert os.path.exists(cache_dir), "Cache directory was not created"
    
    # Check that metadata file was created
    meta_file = os.path.join(cache_dir, "cache_meta.json")
    assert os.path.exists(meta_file), "Cache metadata file was not created"
    
    # Cleanup
    loader.cleanup()


def test_cache_key_generation(test_data_dir):
    """Test the generation of cache keys."""
    # Create loader with caching enabled
    loader = UnifiedDataLoader(
        data_dir=test_data_dir,
        enable_cache=True
    )
    
    # Test file path
    test_file = os.path.join(test_data_dir, "test.txt")
    
    # Generate key
    key1 = loader._cache_key(test_file)
    assert key1 is not None, "Cache key generation failed"
    assert isinstance(key1, str), "Cache key should be a string"
    assert len(key1) > 0, "Cache key should not be empty"
    
    # Generate key with params
    params = {"batch_size": 32, "use_metal": True}
    key2 = loader._cache_key(test_file, params)
    assert key2 is not None, "Cache key generation with params failed"
    assert key1 != key2, "Keys with and without params should be different"
    
    # Generate same key again to verify determinism
    key3 = loader._cache_key(test_file, params)
    assert key2 == key3, "Cache key generation should be deterministic"
    
    # Cleanup
    loader.cleanup()


def test_cache_save_load(test_data_dir):
    """Test saving and loading data from cache."""
    # Create loader with caching enabled
    loader = UnifiedDataLoader(
        data_dir=test_data_dir,
        enable_cache=True
    )
    
    # Create test data
    test_data = np.array([1, 2, 3, 4, 5])
    test_file = os.path.join(test_data_dir, "test.txt")
    cache_key = loader._cache_key(test_file)
    
    # Save to cache
    result = loader._save_to_cache(test_data, cache_key)
    assert result, "Saving to cache failed"
    
    # Load from cache
    loaded_data = loader._load_from_cache(cache_key)
    assert loaded_data is not None, "Loading from cache failed"
    assert np.array_equal(loaded_data, test_data), "Loaded data does not match original"
    
    # Test loading non-existent key
    non_existent = loader._load_from_cache("non_existent_key")
    assert non_existent is None, "Loading non-existent key should return None"
    
    # Cleanup
    loader.cleanup()


def test_csv_parsing_valid(test_data_dir):
    """Test CSV parsing with valid data."""
    if not HAS_PANDAS:
        pytest.skip("Pandas not available for CSV testing")
    
    # Create loader
    loader = UnifiedDataLoader(
        data_dir=test_data_dir
    )
    
    # Scan directory
    loader.scan_directory()
    
    # Load only CSV files
    loader.load_data(file_types=["csv"])
    
    # Verify CSV data was loaded
    assert "csv" in loader.loaded_data, "CSV data not loaded"
    assert len(loader.loaded_data["csv"]) > 0, "No CSV data was loaded"
    
    # Check if we have a DataFrame for the valid CSV
    found_valid = False
    for item in loader.loaded_data["csv"]:
        if "valid.csv" in item["path"] and "dataframe" in item:
            found_valid = True
            df = item["dataframe"]
            # Check DataFrame content
            assert len(df) == 3, "Valid CSV should have 3 rows"
            assert list(df.columns) == ["id", "name", "value"], "Columns don't match expected"
    
    assert found_valid, "Valid CSV was not properly loaded"
    
    # Cleanup
    loader.cleanup()


def test_csv_parsing_malformed(test_data_dir):
    """Test CSV parsing with malformed data using on_bad_lines parameter."""
    if not HAS_PANDAS:
        pytest.skip("Pandas not available for CSV testing")
    
    # Create a custom load_csv function to inspect how pd.read_csv is called
    original_read_csv = pd.read_csv
    
    # Flag to track if on_bad_lines was used
    on_bad_lines_used = False
    
    def mock_read_csv(*args, **kwargs):
        nonlocal on_bad_lines_used
        if 'on_bad_lines' in kwargs:
            on_bad_lines_used = True
        return original_read_csv(*args, **kwargs)
    
    # Replace pandas read_csv with our mock
    pd.read_csv = mock_read_csv
    
    try:
        # Create loader
        loader = UnifiedDataLoader(
            data_dir=test_data_dir
        )
        
        # Scan directory
        loader.scan_directory()
        
        # Load only CSV files
        loader.load_data(file_types=["csv"])
        
        # Verify the on_bad_lines parameter was used
        assert on_bad_lines_used, "on_bad_lines parameter was not used in pd.read_csv"
        
        # Check if malformed CSV was loaded despite errors
        found_malformed = False
        for item in loader.loaded_data["csv"]:
            if "malformed.csv" in item["path"]:
                found_malformed = True
        
        assert found_malformed, "Malformed CSV was not loaded with error handling"
        
        # Cleanup
        loader.cleanup()
    finally:
        # Restore original pandas function
        pd.read_csv = original_read_csv


def test_metal_acceleration(test_data_dir):
    """Test MLX/Metal acceleration if available."""
    if not HAS_MLX:
        pytest.skip("MLX not available for acceleration testing")
    
    # Create loader with Metal enabled
    loader = UnifiedDataLoader(
        data_dir=test_data_dir,
        use_metal=True
    )
    
    # Check if Metal is actually available (platform dependent)
    metal_available = hasattr(mx, 'metal') and mx.metal.is_available()
    
    # If Metal is available, loader.use_metal should be True
    if metal_available:
        assert loader.use_metal, "Metal should be enabled when available"
    
    # Scan and load data
    loader.scan_directory()
    loader.load_data(file_types=["txt", "py"])
    
    # Process data to create tensors
    processed = loader.process_data()
    
    # If Metal is available and enabled, processed should be an mx.array
    if metal_available and loader.use_metal:
        assert isinstance(processed, mx.array), "Processed data should be an MLX array when Metal is available"
    else:
        # Otherwise it should be numpy array
        assert isinstance(processed, np.ndarray), "Processed data should fall back to NumPy when Metal is not available"
    
    # Cleanup
    loader.cleanup()


def test_performance_with_cache(test_data_dir):
    """Test performance improvement with caching."""
    # Create a larger test file for more noticeable performance difference
    big_file = os.path.join(test_data_dir, "large.txt")
    with open(big_file, "w") as f:
        # Write 1MB of data
        for i in range(10000):
            f.write(f"Line {i}: This is a large file for performance testing of the cache system.\n")
    
    # First run without cache
    start_time = time.time()
    loader_nocache = UnifiedDataLoader(
        data_dir=test_data_dir,
        enable_cache=False
    )
    loader_nocache.scan_directory()
    loader_nocache.load_data(file_types=["txt"])
    loader_nocache.process_data()
    no_cache_time = time.time() - start_time
    loader_nocache.cleanup()
    
    # Second run with cache
    start_time = time.time()
    loader_cache = UnifiedDataLoader(
        data_dir=test_data_dir,
        enable_cache=True
    )
    loader_cache.scan_directory()
    loader_cache.load_data(file_types=["txt"])
    loader_cache.process_data()
    first_cache_time = time.time() - start_time
    
    # Third run with cache (should be faster due to cache hit)
    start_time = time.time()
    loader_cache.load_data(file_types=["txt"])
    loader_cache.process_data()
    second_cache_time = time.time() - start_time
    loader_cache.cleanup()
    
    # Log the times
    print(f"\nPerformance test results:")
    print(f"No cache: {no_cache_time:.4f}s")
    print(f"First run with cache: {first_cache_time:.4f}s")
    print(f"Second run with cache: {second_cache_time:.4f}s")
    
    # The second run with cache should be significantly faster than without cache
    # This might not always be true in CI environments, so we'll just log it
    # rather than asserting it
    if second_cache_time < no_cache_time:
        print("Cache improved performance as expected")
    else:
        print("Warning: Cache did not improve performance as expected")


def test_cleanup_cache(test_data_dir):
    """Test cache cleanup functionality."""
    # Create loader with caching enabled
    loader = UnifiedDataLoader(
        data_dir=test_data_dir,
        enable_cache=True
    )
    
    # Create test data and save to cache
    test_data = np.array([1, 2, 3, 4, 5])
    test_file = os.path.join(test_data_dir, "test.txt")
    cache_key = loader._cache_key(test_file)
    loader._save_to_cache(test_data, cache_key)
    
    # Verify data is in cache
    assert loader._load_from_cache(cache_key) is not None, "Data should be in cache"
    
    # Get the actual cache file path from metadata
    cache_file = loader.cache_meta[cache_key]['file']
    assert os.path.exists(cache_file), f"Cache file does not exist: {cache_file}"
    
    # Delete cache file directly to simulate corruption
    os.remove(cache_file)
    
    # Ensure the file is gone
    assert not os.path.exists(cache_file), "Cache file should be deleted for test"
    
    # Force file system sync to ensure deletion is registered
    import time
    time.sleep(0.1)  # Brief pause to let filesystem operations complete
    
    # Try to load corrupted cache
    loaded = loader._load_from_cache(cache_key)
    assert loaded is None, "Corrupted cache should return None"
    
    # Verify metadata was cleaned up
    assert cache_key not in loader.cache_meta, "Corrupted cache metadata should be removed"
    
    # Cleanup
    loader.cleanup()


if __name__ == "__main__":
    pytest.main(["-v", __file__])

