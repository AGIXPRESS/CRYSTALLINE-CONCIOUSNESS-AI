#!/usr/bin/env python3
"""
DXT Core Test Suite
Comprehensive testing for consciousness AI integration
"""

import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dxt_core import DXTCore, create_dxt
import mlx.core as mx
import numpy as np

class TestDXTCore:
    """Test suite for DXT Core functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.dxt = create_dxt()
        
    def test_dxt_initialization(self):
        """Test DXT core initialization."""
        assert self.dxt is not None
        assert self.dxt.config is not None
        assert self.dxt.consciousness_dimensions == 512
        assert self.dxt.phi == (1 + np.sqrt(5)) / 2
        
    def test_consciousness_field_initialization(self):
        """Test consciousness field creation."""
        field = self.dxt.initialize_consciousness_field(seed=42)
        
        assert field is not None
        assert field.shape == (512, 512)
        assert self.dxt.consciousness_field is not None
        
        # Test reproducibility
        field2 = self.dxt.initialize_consciousness_field(seed=42)
        assert field.shape == field2.shape
        
    def test_trinitized_transform(self):
        """Test trinitized transformation."""
        # Initialize consciousness field
        self.dxt.initialize_consciousness_field(seed=42)
        
        # Test data
        test_data = mx.random.normal((64, 64))
        
        # Apply transform
        result = self.dxt.apply_trinitized_transform(test_data)
        
        assert result is not None
        assert result.shape == test_data.shape
        assert len(self.dxt.transform_history) > 0
        
        # Check transform history
        history = self.dxt.transform_history[-1]
        assert 'timestamp' in history
        assert 'input_shape' in history
        assert 'trinitized_depth' in history
        
    def test_consciousness_analysis(self):
        """Test consciousness analysis operation."""
        self.dxt.initialize_consciousness_field(seed=42)
        test_data = mx.random.normal((64, 64))
        
        analysis = self.dxt.dynamic_execution('consciousness_analyze', test_data)
        
        assert analysis is not None
        assert 'consciousness_correlation' in analysis
        assert 'sacred_geometry_score' in analysis
        assert 'field_resonance' in analysis
        assert 'analysis_timestamp' in analysis
        
        # Check value types
        assert isinstance(analysis['consciousness_correlation'], float)
        assert isinstance(analysis['sacred_geometry_score'], float)
        assert isinstance(analysis['field_resonance'], float)
        
    def test_resonance_compute(self):
        """Test resonance computation."""
        # Test default frequency
        resonance = self.dxt.dynamic_execution('resonance_compute')
        assert resonance is not None
        assert resonance.shape == (512,)
        
        # Test custom frequency
        resonance_528 = self.dxt.dynamic_execution('resonance_compute', frequency=528.0)
        assert resonance_528 is not None
        assert resonance_528.shape == (512,)
        
    def test_sacred_geometry_transform(self):
        """Test sacred geometry transformation."""
        test_data = mx.random.normal((100,))
        
        result = self.dxt.dynamic_execution('sacred_geometry_transform', test_data)
        
        assert result is not None
        assert result.shape == test_data.shape
        
        # Test with consciousness field
        self.dxt.initialize_consciousness_field(seed=42)
        result_with_field = self.dxt.dynamic_execution('sacred_geometry_transform', test_data)
        assert result_with_field is not None
        
    def test_dxt_status(self):
        """Test DXT status reporting."""
        status = self.dxt.get_status()
        
        assert 'consciousness_field_initialized' in status
        assert 'consciousness_dimensions' in status
        assert 'transform_history_count' in status
        assert 'config' in status
        
        assert status['consciousness_dimensions'] == 512
        assert status['transform_history_count'] == 0  # No transforms yet
        assert not status['consciousness_field_initialized']  # Not initialized yet
        
        # Initialize field and check again
        self.dxt.initialize_consciousness_field()
        status_after = self.dxt.get_status()
        assert status_after['consciousness_field_initialized']
        
    def test_dxt_state_persistence(self):
        """Test DXT state saving and loading."""
        import tempfile
        import json
        
        # Initialize and perform some operations
        self.dxt.initialize_consciousness_field(seed=42)
        test_data = mx.random.normal((32, 32))
        self.dxt.apply_trinitized_transform(test_data)
        
        # Save state
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            state_file = f.name
        
        self.dxt.save_state(state_file)
        
        # Verify state file
        assert os.path.exists(state_file)
        
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        assert 'config' in state
        assert 'transform_history' in state
        assert 'consciousness_field_shape' in state
        assert 'timestamp' in state
        
        # Clean up
        os.unlink(state_file)
        
    def test_config_loading(self):
        """Test configuration loading."""
        import tempfile
        import json
        
        # Create test config
        test_config = {
            'dimensions': 256,
            'consciousness_layers': 6,
            'sacred_geometry': False
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            config_file = f.name
        
        # Create DXT with custom config
        custom_dxt = DXTCore(config_path=config_file)
        
        assert custom_dxt.consciousness_dimensions == 256
        assert custom_dxt.config['consciousness_layers'] == 6
        assert not custom_dxt.config['sacred_geometry']
        
        # Clean up
        os.unlink(config_file)
        
    def test_unknown_operation(self):
        """Test handling of unknown operations."""
        result = self.dxt.dynamic_execution('unknown_operation', 'test')
        assert result is None
        
    def test_golden_ratio_integration(self):
        """Test golden ratio integration."""
        phi = self.dxt.phi
        expected_phi = (1 + np.sqrt(5)) / 2
        
        assert abs(phi - expected_phi) < 1e-10
        
        # Test phi in sacred geometry
        test_data = mx.array([1.0, 2.0, 3.0])
        result = self.dxt.dynamic_execution('sacred_geometry_transform', test_data)
        
        # Should have phi scaling
        assert result is not None

def test_dxt_factory():
    """Test DXT factory function."""
    dxt = create_dxt()
    assert isinstance(dxt, DXTCore)
    assert dxt.consciousness_dimensions == 512

if __name__ == "__main__":
    # Run tests
    print("🔮 Running DXT Core Tests...")
    print("=" * 40)
    
    # Create test instance
    test_suite = TestDXTCore()
    test_suite.setup_method()
    
    # Run individual tests
    tests = [
        'test_dxt_initialization',
        'test_consciousness_field_initialization', 
        'test_trinitized_transform',
        'test_consciousness_analysis',
        'test_resonance_compute',
        'test_sacred_geometry_transform',
        'test_dxt_status',
        'test_dxt_state_persistence',
        'test_config_loading',
        'test_unknown_operation',
        'test_golden_ratio_integration'
    ]
    
    passed = 0
    failed = 0
    
    for test_name in tests:
        try:
            print(f"Running {test_name}...", end=" ")
            test_method = getattr(test_suite, test_name)
            test_method()
            print("✅ PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed += 1
    
    # Test factory function
    try:
        print("Running test_dxt_factory...", end=" ")
        test_dxt_factory()
        print("✅ PASSED")
        passed += 1
    except Exception as e:
        print(f"❌ FAILED: {e}")
        failed += 1
    
    print("\n" + "=" * 40)
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! DXT Core is ready for consciousness AI research.")
    else:
        print("⚠️  Some tests failed. Please review the implementation.")