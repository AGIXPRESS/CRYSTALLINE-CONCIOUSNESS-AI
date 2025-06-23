# Metal Initialization Fix - 2025-04-28

## Issue
Fixed initialization sequence in metal_ops.py to prevent NameError during module import.

### Changes Made
1. Added global variable declaration near other globals:
```python
# Metal initialization status
_is_metal_initialized = False
```

2. Added global declaration in _initialize_metal():
```python
def _initialize_metal():
    """Initialize Metal device and load shader libraries."""
    global _is_metal_initialized
    
    with _metal_init_lock:
        ...
```

3. Fixed initialization at module import:
```python
# Initialize Metal on module import
_initialize_metal()
```

### Impact
- Prevents NameError when the module is imported
- Maintains thread-safety with proper lock usage
- Ensures consistent initialization state

### Related Files
- src/python/metal_ops.py

