import torch
import math
from tensorvid import VideoLoader

class FastVideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, video_path: str, batch_size: int, height: int, width: int):
        super().__init__()
        # Initialize the Rust core. The Rust core spins up its native thread pool immediately.
        self.loader = VideoLoader(video_path, batch_size, height, width)

    def __iter__(self):
        return self

    def __next__(self):
        # Fetch the zero-copy NumPy view from Rust
        # The Rust side explicitly releases the GIL (`py.allow_threads`) while waiting for the lock-free queue.
        np_batch = self.loader.next_batch()
        
        if np_batch is None:
            raise StopIteration
            
        # Convert numpy array to PyTorch tensor with ZERO COPY
        # torch.from_numpy preserves the underlying C data pointer and stride information
        tensor = torch.from_numpy(np_batch)
        
        # The tensor is directly accessible and is contiguous in memory.
        # It can be transparently transferred to the GPU asynchronously without host-side bottlenecks.
        # tensor = tensor.to('cuda', non_blocking=True)
        
        return tensor

if __name__ == "__main__":
    dataset = FastVideoDataset("sample.mp4", batch_size=32, height=224, width=224)
    
    # CRITICAL: Do not use multiprocessing workers in PyTorch DataLoader (`num_workers>0`)
    # Our Rust core already handles parallel decoding, memory pooling, and background pipelining internally.
    # Using python multiprocessing here would just create IPC overhead and duplicate the Rust context.
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0)

    for i, batch in enumerate(loader):
        print(f"Batch {i}: Shape {batch.shape}, Type {batch.dtype}, Device {batch.device}")
