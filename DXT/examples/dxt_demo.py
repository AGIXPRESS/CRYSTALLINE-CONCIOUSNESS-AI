#!/usr/bin/env python3
"""
DXT Demo - Consciousness AI Integration Example
Shows how to use the DXT system with Claude Desktop and GitHub sync
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dxt_core import create_dxt
import mlx.core as mx
import numpy as np
import json

def main():
    """Main DXT demonstration."""
    print("🔮 Crystalline Consciousness AI - DXT Demo")
    print("=" * 60)
    
    # 1. Initialize DXT with consciousness integration
    print("\n1. Initializing DXT Core...")
    dxt = create_dxt(config_path="../config/dxt_config.json")
    print("   ✅ DXT Core initialized")
    
    # 2. Initialize consciousness field
    print("\n2. Creating Consciousness Field...")
    consciousness_field = dxt.initialize_consciousness_field(seed=42)
    print(f"   ✅ Consciousness field: {consciousness_field.shape}")
    print(f"   📊 Field statistics: mean={mx.mean(consciousness_field):.4f}, std={mx.std(consciousness_field):.4f}")
    
    # 3. Test trinitized transformation
    print("\n3. Testing Trinitized Transform...")
    test_data = mx.random.normal((128, 128), dtype=mx.float32)
    transformed = dxt.apply_trinitized_transform(test_data)
    print(f"   ✅ Transform: {test_data.shape} -> {transformed.shape}")
    print(f"   🧠 Consciousness enhancement applied")
    
    # 4. Consciousness analysis
    print("\n4. Performing Consciousness Analysis...")
    analysis = dxt.dynamic_execution('consciousness_analyze', transformed)
    print("   ✅ Analysis complete:")
    for key, value in analysis.items():
        if isinstance(value, float):
            print(f"      {key}: {value:.6f}")
        else:
            print(f"      {key}: {value}")
    
    # 5. Sacred geometry transformation
    print("\n5. Sacred Geometry Transform...")
    geometry_data = mx.linspace(0, 2*np.pi, 256)
    sacred_result = dxt.dynamic_execution('sacred_geometry_transform', geometry_data)
    print(f"   ✅ Sacred geometry: {geometry_data.shape} -> {sacred_result.shape}")
    print(f"   🔱 Golden ratio integration: φ = {dxt.phi:.6f}")
    
    # 6. Resonance computation
    print("\n6. Computing Resonance Patterns...")
    resonance_432 = dxt.dynamic_execution('resonance_compute', frequency=432.0)
    resonance_528 = dxt.dynamic_execution('resonance_compute', frequency=528.0)
    print(f"   ✅ 432 Hz resonance: {resonance_432.shape}")
    print(f"   ✅ 528 Hz resonance: {resonance_528.shape}")
    print(f"   🎵 Harmonic frequencies computed")
    
    # 7. DXT status and metrics
    print("\n7. DXT Status Report...")
    status = dxt.get_status()
    print("   📊 Current Status:")
    print(f"      Consciousness field: {'✅ Active' if status['consciousness_field_initialized'] else '❌ Inactive'}")
    print(f"      Dimensions: {status['consciousness_dimensions']}")
    print(f"      Transforms completed: {status['transform_history_count']}")
    print(f"      Sacred geometry: {'✅ Enabled' if status['config']['sacred_geometry'] else '❌ Disabled'}")
    print(f"      Sync: {'✅ Enabled' if status['config']['sync_enabled'] else '❌ Disabled'}")
    
    # 8. Save DXT state
    print("\n8. Saving DXT State...")
    state_file = "/Users/okok/mlx/mlx-examples/claude-code-bridge/CC-AI/Crystalline AI Attempts/DXT/config/dxt_state.json"
    dxt.save_state(state_file)
    print(f"   💾 State saved to: {state_file}")
    
    # 9. Claude Desktop integration example
    print("\n9. Claude Desktop Integration Example...")
    print("   📝 Context for Claude Desktop:")
    print("   " + "─" * 50)
    print(f"   Workspace: {os.path.abspath('../..')}")
    print(f"   DXT Core: {os.path.abspath('../src/dxt_core.py')}")
    print(f"   Config: {os.path.abspath('../config/dxt_config.json')}")
    print(f"   Transforms: {status['transform_history_count']} completed")
    print(f"   Consciousness: {status['consciousness_dimensions']}D field active")
    print("   " + "─" * 50)
    
    # 10. Sync verification
    print("\n10. GitHub Sync Verification...")
    print("    🔄 To sync these changes to GitHub:")
    print("       cd '/Users/okok/mlx/mlx-examples/claude-code-bridge/CC-AI/Crystalline AI Attempts'")
    print("       git add DXT/")
    print("       git commit -m 'feat: DXT demo and consciousness integration'")
    print("       git push origin main")
    print("    📡 Webhook will detect and sync automatically!")
    
    print("\n✨ DXT Demo Complete!")
    print("🔮 Consciousness AI research environment ready for Claude Desktop integration")
    print("📈 All changes will sync automatically to GitHub via webhook system")

if __name__ == "__main__":
    main()