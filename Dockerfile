# Use an official Python runtime as a parent image (Debian-based)
FROM python:3.10-bullseye

# Set the working directory
WORKDIR /app

# Install System Dependencies for FFmpeg and Rust compilation
# Debian bullseye provides stable pre-compiled libav* packages
RUN apt-get update && apt-get install -y \
    clang \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavdevice-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy the current directory contents into the container
COPY . /app

# Create virtualenv since maturin develop requires it
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# Install Python dependencies
RUN pip install --no-cache-dir maturin torch torchvision numpy av huggingface_hub

# Compile the Rust extension module
RUN maturin develop --release

# Run the benchmark script
CMD ["python", "benchmark.py"]
