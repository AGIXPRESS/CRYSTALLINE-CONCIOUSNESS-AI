# MLX Testing Quick Reference Guide
Quick reference for test execution and troubleshooting

## Quick Start Commands

### Basic Test Execution
```bash
# Full test suite
python tools/run_tests.py --suite=full

# Individual components
python tools/run_tests.py --suite=geometric  # Geometric tests
python tools/run_tests.py --suite=quantum    # Quantum coherence
python tools/run_tests.py --suite=performance # Performance tests
```

### Critical Thresholds
```python
CRITICAL_THRESHOLDS = {
    'memory': {
        'reduction': 0.40,    # 40% reduction
        'max_usage': '400MB'  # Maximum memory usage
    },
    'performance': {
        'speedup': 0.25,      # 25% speedup
        'cache_hit': 0.95     # Cache hit rate
    },
    'quantum': {
        'coherence': 0.99,    # State coherence
        'phase': 0.95         # Phase preservation
    }
}
```

### Common Issues & Fixes

1. Memory Issues
   ```bash
   # Check memory usage
   python tools/memory_monitor.py --check
   
   # Enable conservative mode
   python tools/run_tests.py --memory-conservative
   ```

2. Performance Issues
   ```bash
   # Profile execution
   python tools/profile_tests.py --detail
   
   # Check vectorization
   python tools/check_vectorization.py
   ```

3. Quantum Coherence
   ```bash
   # Verify state stability
   python tools/quantum_check.py --verify
   
   # Check phase relationships
   python tools/phase_analysis.py
   ```

## Test Sequence

1. Environment Setup
   ```bash
   source venv/bin/activate
   python tools/verify_environment.py
   ```

2. Basic Validation
   ```bash
   python tools/run_tests.py --suite=basic --quick
   ```

3. Full Test Suite
   ```bash
   python tools/run_tests.py --suite=full --report
   ```

## Emergency Procedures

### 1. Test Failures
```bash
# Revert to last stable
python tools/revert.py --last-stable

# Run diagnostics
python tools/diagnose.py --full
```

### 2. Performance Issues
```bash
# Enable monitoring
python tools/monitor.py --continuous

# Generate report
python tools/report.py --performance
```

### 3. Critical Errors
```bash
# Emergency stop
python tools/emergency_stop.py

# Contact team
python tools/alert.py --team=core
```

## Result Verification

### 1. Quick Check
```bash
# Verify basic metrics
python tools/verify.py --quick

# Check thresholds
python tools/check_thresholds.py
```

### 2. Detailed Analysis
```bash
# Full analysis
python tools/analyze.py --full

# Generate report
python tools/report.py --detailed
```

## Key Files Location

- Main Tests: tests/
- Performance Tests: tests/performance/
- Geometric Tests: tests/geometric/
- Quantum Tests: tests/quantum/
- Tools: tools/
- Reports: reports/

## Support Contacts

- Performance Issues: performance-team@example.com
- Quantum Issues: quantum-team@example.com
- General Support: support@example.com

## Documentation Links

- Full Test Plan: docs/testing/mlx_testing_plan.md
- Test Scenarios: docs/testing/mlx_test_scenarios.md
- Test Specifications: docs/testing/mlx_test_specifications.md
- Execution Guide: docs/testing/mlx_test_execution_guide.md

