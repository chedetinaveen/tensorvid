# Installation

TensorVid is built with Rust and packaged as a native Python extension using `maturin`. 

## Prerequisites
1. **Rust**: Install via [rustup](https://rustup.rs/).
2. **FFmpeg Libraries**: You need the native C-headers for FFmpeg.

**On Ubuntu/Debian (Linux):**
```bash
sudo apt-get update && sudo apt-get install -y \
    clang pkg-config libavcodec-dev libavformat-dev \
    libavutil-dev libswscale-dev libavdevice-dev
```

**On macOS (Homebrew):**
```bash
brew install ffmpeg pkg-config
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig"
## The Easiest Way: Conda-Forge (Coming Soon)

The absolute most robust way to install TensorVid on any operating system without worrying about C-compilers or Homebrew bugs is via Conda. This downloads a pre-compiled binary that already contains the correct FFmpeg bindings for your exact operating system:

```bash
conda install -c conda-forge tensorvid
```

## Quick Install (PyPI Source)

If you already have the prerequisites (Rust + FFmpeg headers) installed, you can compile the source distribution from PyPI:

```bash
pip install tensorvid
```

## Build from Source
Ensure you are in an active Python virtual environment (`venv` or `conda`), then compile the extension:

```bash
pip install maturin torch
maturin develop --release
```

This compiles the Rust backend and seamlessly exposes the `tensorvid` module to your Python environment.

!!! warning "macOS Homebrew Bug"
    If `pip` crashes with a `pyexpat` or `libexpat` error, it is a known bug in Homebrew Python. You must prioritize the Homebrew `expat` library by running this before your Python commands:
    
    `export DYLD_LIBRARY_PATH="/opt/homebrew/opt/expat/lib:$DYLD_LIBRARY_PATH"`

## Docker Setup
For guaranteed compatibility, use the provided Dockerfile which isolates the Linux FFmpeg headers:

```bash
docker build -t tensorvid .
docker run --rm tensorvid
```
