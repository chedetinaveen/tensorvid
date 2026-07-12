# TensorVid

[![Documentation](https://img.shields.io/badge/docs-github%20pages-blue.svg)](https://chedetinaveen.github.io/tensorvid/)
[![Build Status](https://github.com/chedetinaveen/tensorvid/actions/workflows/benchmark.yml/badge.svg)](https://github.com/chedetinaveen/tensorvid/actions)

TensorVid is an ultra-fast, zero-copy video dataloader for PyTorch, written in Rust.

By bridging `ffmpeg-next` (C/Rust) directly to `ndarray` and `PyO3`, TensorVid bypasses the Python Global Interpreter Lock (GIL) and avoids expensive temporary heap allocations in numpy. It streams decoded video frames directly into PyTorch C++ memory space, unlocking massive throughput scaling on standard CPUs.

## Installation

You can install TensorVid directly from PyPI. Note that because this is a source distribution wrapping C-libraries, you **must** have the native FFmpeg C-headers and the Rust compiler installed on your system before running `pip install`.

```bash
pip install tensorvid
```

### Build from Source
```bash
# Clone the repository
git clone https://github.com/chedetinaveen/tensorvid.git
cd tensorvid

# Create virtual environment and install requirements
python -m venv .venv
source .venv/bin/activate
pip install maturin torch numpy

# Build and install the Python bindings via Maturin
maturin develop --release
```

## Quick Start

```python
import torch
from tensorvid import FastVideoDataset

dataset = FastVideoDataset(
    video_path="sample.mp4", 
    batch_size=16, 
    height=224, 
    width=224
)

loader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=None, # batching is handled natively in Rust for speed
    num_workers=0    # Rust handles threading internally
)

for batch in loader:
    batch = batch.permute(0, 3, 1, 2) # [16, C, H, W]
    batch = batch.to('cuda', non_blocking=True)
```

## Benchmarks

In a 100-video scale test spanning 32,000 frames from UCF-101 (on a single thread):

| Architecture | Language Backend | Frames Per Second (FPS) |
|--------------|-----------------|--------------------------|
| **PyAV**     | Python + C      | ~1,108 FPS                |
| **TensorVid**| Rust + C        | **~5,498 FPS**            |

Read the [full documentation](https://chedetinaveen.github.io/tensorvid/) for API details.
