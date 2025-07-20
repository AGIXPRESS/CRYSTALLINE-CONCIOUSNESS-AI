# Setting up MLX/MPS for CrystallineConsciousnessAI

This guide provides instructions on setting up and verifying MLX/MPS (Metal Performance Shaders) for GPU acceleration in the CrystallineConsciousnessAI project. Ensure you have the correct environment to leverage your Apple Silicon GPU.

## Prerequisites

*   **macOS:** This project is optimized for macOS with Apple Silicon (M1, M2, M3) chips.
*   **Python:** Ensure you have Python 3.9 or higher installed.
*   **Homebrew (Recommended):** Homebrew is a package manager that simplifies the installation of dependencies. If you don't have it installed, you can install it from [https://brew.sh/](https://brew.sh/).

## Installation Steps

1.  **Install MLX:**
    *   Using pip:
        ```bash
        pip install mlx
        ```
    *   This command installs the MLX framework.

2.  **Verify MLX Installation:**
    *   Open a Python terminal and run the following code:
        ```python
        import mlx.core as mx
        print(mx.Device.default_device())
        ```
    *   This should output either `gpu` or `cpu`. If it outputs `gpu`, MLX is using your Apple Silicon GPU. If it outputs `cpu`, then it is using the CPU.

3.  **Troubleshooting:**
    *   If MLX is not using the GPU, ensure that your macOS is up to date.
    *   If you are having issues with pip, consider using a virtual environment:
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        pip install mlx
        ```
    *   If you still face problems, consult the official MLX documentation: [https://github.com/ml-explore/mlx](https://github.com/ml-explore/mlx)

## Verifying Metal (MPS) Configuration

1.  **Check Metal Availability:**
    *   Add the following code snippet to your `unified_data_loader.py` file:
        ```python
        import mlx.core as mx
        metal_available = mx.metal.is_available() if hasattr(mx, 'metal') else False
        print(f"Metal available: {metal_available}")
        ```
    *   This will give you the status of the metal acceleration on your machine.

2.  **Run the `unified_data_loader.py`:**
    *   Execute the script and check the output. If Metal is correctly configured, you should see `Metal available: True`.

## Additional Notes

*   If you encounter issues with specific operations or large models, ensure that you have enough available memory on your GPU.
*   Keep your MLX and macOS versions updated to benefit from the latest performance improvements and bug fixes.

By following these steps, you can ensure that MLX/MPS is correctly set up and verified, allowing you to fully utilize your Apple Silicon GPU for accelerated computations in your projects.

