# MLX Component-Specific Test Guide
Detailed testing procedures for each component

## Octahedron Activation Tests

### 1. Quarter Processing Tests
```bash
# Test first quarter calculations
python tools/run_tests.py --component=quarters --section=first

# Test last quarter calculations
python tools/run_tests.py --component=quarters --section=last

# Test middle section
python tools/run_tests.py --component=quarters --section=middle
```

### 2. Wave Component Tests
```bash
# Test individual wave components
python tools/run_tests.py --component=waves --verify-components

# Test harmonic interactions
python tools/run_tests.py --component=waves --verify-harmonics

# Full wave validation
python tools/run_tests.py --component=waves --full-validation
```

### 3. Memory Pattern Tests
```bash
# Test array initialization
python tools/run_tests.py --component=memory --test-init

# Test allocation patterns
python tools/run_tests.py --component=memory --test-allocation

# Test deallocation
python tools/run_tests.py --component=memory --test-cleanup
```

## Common Test Patterns

### 1. Array Initialization
```python
def test_specific_initialization():
    """
    Test specific array initialization patterns
    Validates:
    - Order of initialization
    - Memory usage
    - Type correctness
    """
    commands = [
        "python tools/test_arrays.py --verify-order",
        "python tools/test_arrays.py --verify-memory",
        "python tools/test_arrays.py --verify-types"
    ]
    return run_test_sequence(commands)
```

### 2. Geometric Validation
```python
def test_specific_geometry():
    """
    Test geometric properties
    Validates:
    - Ratio preservation
    - Symmetry maintenance
    - Boundary behavior
    """
    commands = [
        "python tools/test_geometry.py --verify-ratios",
        "python tools/test_geometry.py --verify-symmetry",
        "python tools/test_geometry.py --verify-boundaries"
    ]
    return run_test_sequence(commands)
```

### 3. Quantum Properties
```python
def test_specific_quantum():
    """
    Test quantum properties
    Validates:
    - State coherence
    - Phase relationships
    - Boundary transitions
    """
    commands = [
        "python tools/test_quantum.py --verify-coherence",
        "python tools/test_quantum.py --verify-phase",
        "python tools/test_quantum.py --verify-transitions"
    ]
    return run_test_sequence(commands)
```

## Test Data Requirements

### 1. Quarter-Specific Data
```python
QUARTER_TEST_DATA = {
    'first_quarter': {
        'size': 256,
        'expected_ratio': 9/8,
        'trend': 'increasing'
    },
    'last_quarter': {
        'size': 256,
        'expected_ratio': 6/5,
        'trend': 'decreasing'
    },
    'middle_section': {
        'size': 512,
        'expected_symmetry': True,
        'balance': 'neutral'
    }
}
```

### 2. Wave Component Data
```python
WAVE_TEST_DATA = {
    'primary': {
        'frequency': 4.0 * math.pi,
        'amplitude': 5.0,
        'phase': 0.0
    },
    'secondary': {
        'frequency': 8.0 * math.pi,
        'amplitude': 4.0,
        'phase': 0.0
    },
    'tertiary': {
        'frequency': 12.0 * math.pi,
        'amplitude': 3.0,
        'phase': 0.5 * math.pi
    }
}
```

## Component-Specific Thresholds

### 1. Quarter Processing
```python
QUARTER_THRESHOLDS = {
    'first': {
        'trend_tolerance': 1e-6,
        'ratio_tolerance': 1e-5
    },
    'last': {
        'correlation_tolerance': 1e-6,
        'ratio_tolerance': 1e-5
    },
    'middle': {
        'symmetry_tolerance': 1e-6,
        'balance_tolerance': 1e-5
    }
}
```

### 2. Wave Generation
```python
WAVE_THRESHOLDS = {
    'amplitude_tolerance': 1e-6,
    'phase_tolerance': 1e-6,
    'frequency_tolerance': 1e-6,
    'harmonic_tolerance': 1e-5
}
```

## Result Verification

### 1. Component-Specific Checks
```bash
# Quarter processing
python tools/verify_quarters.py --full-check

# Wave generation
python tools/verify_waves.py --full-check

# Memory patterns
python tools/verify_memory.py --full-check
```

### 2. Integration Checks
```bash
# Verify component interactions
python tools/verify_integration.py --all-components

# Check boundary conditions
python tools/verify_boundaries.py --all-transitions
```

## Troubleshooting Guide

### Common Component Issues
1. Quarter Processing
   - Check ratio calculations
   - Verify boundary transitions
   - Validate trend calculations

2. Wave Generation
   - Verify frequency components
   - Check phase relationships
   - Validate amplitude scaling

3. Memory Patterns
   - Check initialization order
   - Verify deallocation
   - Monitor fragmentation

### Emergency Response
```bash
# Component-specific recovery
python tools/recover.py --component=quarters
python tools/recover.py --component=waves
python tools/recover.py --component=memory
```

