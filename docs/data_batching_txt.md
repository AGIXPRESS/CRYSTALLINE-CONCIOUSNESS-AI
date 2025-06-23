# Data Batching for .txt Files

This document describes the data batching process for `.txt` files within the `unified_data_loader.py` script. Data batching is essential for efficient training of machine learning models by dividing the dataset into smaller, manageable chunks.

## 1. Numerical Representation

Before batching, the text data has already been processed into a numerical representation within the `process_data()` function. As a result, each text entry has been converted into a sequence of integer indices, where each index corresponds to a token in the vocabulary.

## 2. Batching Process (`get_batch` Function)

The `get_batch()` function implements the data batching logic. It takes a batch index as input and returns a batch of data corresponding to that index.

```python
    def get_batch(self, batch_idx: int) -> Union[np.ndarray, 'mx.array', None]:
        """
        Get a specific batch of data.
        
        Args:
            batch_idx: Index of the batch to retrieve.
            
        Returns:
            A NumPy array or MLX array representing the batch, or None if no data is available.
        """
        if self.processed_data is None:
            logger.warning("No processed data available. Call process_data() first.")
            return None

        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.processed_data))  # Clip at the end
        
        # Get the batch from numpy
        batch = self.processed_data[start_idx:end_idx]

        # Handle empty batches if start_idx is out of range
        if len(batch) == 0:
            logger.warning(f"Empty batch returned for batch_idx: {batch_idx}")
            return None
```

Key steps in this function:

1.  **Calculate Start and End Indices**: Calculates the start and end indices for the batch based on the `batch_idx` and `batch_size`.
2.  **Extract Batch**: Extracts the data for the batch from `self.processed_data` using array slicing. It is important that the `end_idx` is not greater than the size of `self.processed_data`.
3.  **Empty Batch Handling**: Checks for empty batches, which can occur if the `batch_idx` is out of range. If an empty batch is detected, a warning is logged, and `None` is returned.

## 3. MLX/MPS Integration

The `get_batch` function handles the conversion to an MLX array when the `use_metal` flag is enabled and MLX is available. This enables GPU acceleration for model training.

```python
        try:
        # Check if conversion to mlx has been requested and mlx is available
            if HAS_MLX and self.use_metal:
                # Convert to MLX array
                try:
                    batch = mx.array(batch, dtype=mx.int32)
                    logger.debug("Converted batch to MLX array")
                except Exception as e:
                    logger.warning(f"Failed to convert batch to MLX array: {e}")
                    
                    # Fallback to numpy, but continue using only the CPU
                    batch = np.array(batch, dtype=np.int32)
            return batch
```

If the conversion to an MLX array fails, the code falls back to using a NumPy array to ensure that the program does not crash and runs on the CPU instead.

## 4. Data Types and Shapes

*   The data in the batches is represented as a NumPy array with `dtype=np.int32`, which is suitable for integer-based token indices.
*   The shape of each batch is `(batch_size,)` or smaller if the end of the dataset is reached.

### Adjustments

*   Adjust the `batch_size` to control the size of each batch. Consider the available GPU memory and the complexity of the model when determining the batch size.
*   Modify the data type (`dtype`) of the NumPy array based on the model's requirements.

## Additional Considerations

*   **Batch Size Tuning**: Experiment with different batch sizes to find the optimal balance between memory usage and training speed.
*   **Memory Management**: Monitor memory usage during training to ensure that the data batches and model do not exceed available memory.
*  **GPU/CPU Fallback** This batch loader is capable of defaulting back to a CPU when GPU operations are unavailable. To take full advantage of this, make sure all downstream operations support both mx and np dtypes.

This documentation provides a comprehensive overview of the data batching process for `.txt` files. It explains how the data is converted to a numerical representation, describes the batch sizes used, and outlines how these batches are created using both NumPy arrays, and MLX arrays to leverage hardware acceleration with Metal if available.

