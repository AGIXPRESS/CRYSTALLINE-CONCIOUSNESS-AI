# Guide to Basic Training Loop in `unified_data_loader.py`

This document provides guidance on how to use the basic training loop implemented in the `unified_data_loader.py` script. This loop is primarily intended for validating data loading, preprocessing, and batching pipelines, not for production-quality training.

## 1. The `TRAIN_MODEL` Flag

The `TRAIN_MODEL` flag (located near the end of the script, at the bottom of the test code) controls whether the basic training loop is executed.

```python
TRAIN_MODEL = False
```

*   Set this flag to `True` to run the training loop.
*   When set to `False`, the training loop and all associated code will be skipped. This helps prevent accidental execution of the test training loop.

## 2. Dummy Model Architecture

A simple "dummy" model is defined within the `train_model` function. This model is used solely to verify data flow and training loop functionality.

```python
class DummyModel(mx.nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.linear = mx.nn.Linear(size, size)

    def __call__(self, x):
        return self.linear(x)
```

This dummy model consists of:

*   A single linear layer (`mx.nn.Linear`) which performs a linear transformation of the input.
*   The `size` parameter controls the input and output dimensions of the linear layer.

It's instantiated in the training loop function:
```python
model = DummyModel(input_size)
```

*`input_size` is set to a value of 10 in this example, can be altered to better reflect the feature dim of a real world implementation, which will allow the dimensions of your model to conform.*

## 3. The Training Loop

The training loop is designed to be a minimal example for verifying data flow.

```python
print("\nStarting Basic Training loop...")
        
    if not hasattr(loader, "train_data") or loader.train_data is None:
            try:
                # First retrieve a processed batch for sizing, and confirm batch size
                if loader.processed_data is not None:
                    initial_batch = loader.get_batch(0)  # Sample initial batch to get size
                    batch_size = len(initial_batch)
                    
                    # Check to see the batch is properly formatted and a good size
                    if batch_size > 0:
                        train_shape = (batch_size,)
                    else:
                        raise AttributeError("Cannot train on a zero-sized batch")

                    for epoch in range(num_epochs):
                        epoch_loss = 0
                        batch_count = 0
                        start_time = time.time()

                        # Set random seed to make batch size consistent
                        np.random.seed(42)
                        # Random input and output in numpy
                        x = np.random.rand(batch_size).astype(np.float32)
                        y = np.random.rand(batch_size).astype(np.float32)

                        # Send all data processing to occur in the backend
                        for batch_i in range(5): # Set iterations to 5 for demonstration
                            batch = loader.get_batch(batch_i)
                            if batch is None:
                                print("No Batch to train")
                                break
```

*   **What it does**:
    *   Iterates for a fixed number of epochs (controlled by `num_epochs`). The test case keeps this at 3.
    *   Retrieves batches using `loader.get_batch(batch_i)`.
    *   The loop generates random `x` and `y` and does not process the model's real output.
*   **Optimizer**: The optimizer is `mx.optimizers.Adam`.
*   **Loss Function**: The loss function is Mean Squared Error (MSE), calculated as:

    ```
    mx.mean((model(x) - y) ** 2)
    ```

    Where:

    *   `model(x)`: Is the model being trained.
    *   `y`: are the labels.
*   **`is_metal` argument:**
    *   This argument enables or disables the use of MLX/MPS (Metal Performance Shaders) for GPU acceleration.
    *   When `is_metal=True`, the code will attempt to use the GPU if MLX is available and properly configured. Otherwise, it will fall back to CPU processing.

## 4. Training Metrics Logging

The training loop provides minimal logging for verification purposes.

*   At the beginning of each epoch, the epoch number is printed.
*   The loss value is intended to be printed each step, but is currently not calculated based on anything that is retrieved from the data loader.
*   It prints a completion message upon finishing.

For proper implementation the batch will have to properly feed in for both the inputs and the labels, and a loss function should also be defined that reflects the true output of the model. A function that implements back propagation will also be neeed to improve the effectiveness of training.

To make any of the above improvements, please see [QUICKSTART.md](./QUICKSTART.md), for a quick look at what has already been done and what is happening.

