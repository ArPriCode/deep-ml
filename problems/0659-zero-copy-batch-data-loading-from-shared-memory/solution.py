import numpy as np

class ZeroCopyBatchLoader:
    def __init__(self, data: np.ndarray, batch_size: int):
        """Store data in a flat contiguous buffer simulating shared memory."""
        self.n_samples, self.n_features = data.shape
        self.batch_size = batch_size
        self.buffer = np.ascontiguousarray(data.flatten())

    def num_batches(self) -> int:
        """Return total number of batches."""
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def get_batch(self, batch_idx: int) -> np.ndarray:
        """Return batch as a zero-copy view into the buffer."""
        total_batches = self.num_batches()
        if batch_idx < 0 or batch_idx >= total_batches:
            raise IndexError("Batch index out of range.")
        
        start_row = batch_idx * self.batch_size
        end_row = min(start_row + self.batch_size, self.n_samples)
        
        start_idx = start_row * self.n_features
        end_idx = end_row * self.n_features
        
        batch_flat = self.buffer[start_idx:end_idx]
        batch_rows = end_row - start_row
        return batch_flat.reshape(batch_rows, self.n_features)

    def is_zero_copy(self, batch_idx: int) -> bool:
        """Check whether the batch shares memory with the internal buffer."""
        batch = self.get_batch(batch_idx)
        return np.shares_memory(batch, self.buffer)

    def get_batch_means(self) -> list:
        """Return a list of per-batch mean values rounded to 4 decimal places."""
        means = []
        for i in range(self.num_batches()):
            batch = self.get_batch(i)
            mean_val = float(np.mean(batch))
            means.append(round(mean_val, 4))
        return means

    def write_to_buffer(self, row: int, col: int, value: float):
        """Write a value directly into the internal flat buffer."""
        flat_idx = row * self.n_features + col
        self.buffer[flat_idx] = value