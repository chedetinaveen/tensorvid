# API Reference

Using TensorVid in your standard PyTorch training loop is designed to be completely drop-in.

## `FastVideoDataset`
TensorVid provides an iterable dataset wrapper out-of-the-box that seamlessly inherits from `torch.utils.data.IterableDataset`.

```python
import torch
from loader import FastVideoDataset

# 1. Instantiate the dataset backend
dataset = FastVideoDataset(
    video_path="sample.mp4", 
    batch_size=16, 
    height=224, 
    width=224
)

# 2. Wrap it in a standard PyTorch DataLoader
loader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=None, # batching is handled natively in Rust for speed
    num_workers=0    # Rust handles threading internally
)

# 3. Iterate as normal!
for batch in loader:
    # batch is a raw torch.Tensor of shape [16, H, W, C]
    # Perform standard operations like permute:
    batch = batch.permute(0, 3, 1, 2) # [16, C, H, W]
    
    # Send directly to GPU
    batch = batch.to('cuda', non_blocking=True)
```

## `tensorvid.VideoLoader`
For low-level control, you can import the raw PyO3 Rust object directly.

```python
import tensorvid

# Instantiates the Rust background decoder thread immediately
decoder = tensorvid.VideoLoader("sample.mp4", 16, 3, 224, 224)

# Blocking call to pull the next chunk out of the Rust ring buffer
numpy_array = decoder.next_batch() 
```
