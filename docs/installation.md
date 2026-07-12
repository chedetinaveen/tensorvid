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
```

## Build from Source
Ensure you are in an active Python virtual environment (`venv` or `conda`), then compile the extension:

```bash
pip install maturin torch
maturin develop --release
```

This compiles the Rust backend and seamlessly exposes the `tensorvid` module to your Python environment.

## Docker Setup
For guaranteed compatibility, use the provided Dockerfile which isolates the Linux FFmpeg headers:

```bash
docker build -t tensorvid .
docker run --rm tensorvid
```
