import sys
sys.path.append('.')
import os
from Python.metal_manager_updated import get_shader_manager, HAS_METAL

# Check if Metal is available
print(f"Metal available: {HAS_METAL}")

# Get shader directory
SHADER_DIR = os.path.join(os.path.dirname(__file__), "Shaders")
print(f"Shader directory: {SHADER_DIR}")

# Get shader paths
GEOMETRIC_SHADER = os.path.join(SHADER_DIR, "GeometricActivation.metal")
RESONANCE_SHADER = os.path.join(SHADER_DIR, "ResonancePatterns.metal")
MUTUALITY_SHADER = os.path.join(SHADER_DIR, "MutualityField.metal")

# Initialize shader manager
manager = get_shader_manager(SHADER_DIR)

# Check if device is initialized
if manager.device is None:
    print("Failed to initialize Metal device")
    exit(1)

print(f"Metal device name: {manager.device.name()}")

# Load shader libraries
if os.path.exists(GEOMETRIC_SHADER):
    success = manager.load_shader_library(GEOMETRIC_SHADER, "geometric")
    print(f"Load geometric shader: {'Success' if success else 'Failed'}")
    
    # Try creating pipelines
    if success:
        for func in ["tetrahedron_activation", "cube_activation", "dodecahedron_activation", 
                    "icosahedron_activation", "unified_geometric_activation"]:
            created = manager.create_compute_pipeline("geometric", func)
            print(f"  Create pipeline for {func}: {'Success' if created else 'Failed'}")

if os.path.exists(RESONANCE_SHADER):
    success = manager.load_shader_library(RESONANCE_SHADER, "resonance")
    print(f"Load resonance shader: {'Success' if success else 'Failed'}")
    
    # Try creating pipeline
    if success:
        created = manager.create_compute_pipeline("resonance", "apply_resonance")
        print(f"  Create pipeline for apply_resonance: {'Success' if created else 'Failed'}")

if os.path.exists(MUTUALITY_SHADER):
    success = manager.load_shader_library(MUTUALITY_SHADER, "mutuality")
    print(f"Load mutuality shader: {'Success' if success else 'Failed'}")
    
    # Try creating pipelines
    if success:
        for func in ["reshape_to_grid", "calculate_mutual_field", "apply_persistence"]:
            created = manager.create_compute_pipeline("mutuality", func)
            print(f"  Create pipeline for {func}: {'Success' if created else 'Failed'}")

# Print loaded libraries and pipelines
print("\nLoaded Libraries:")
for name in manager.libraries:
    print(f"- {name}")

print("\nCreated Pipelines:")
for name in manager.pipelines:
    print(f"- {name}")
