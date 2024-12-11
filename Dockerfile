# To build the image: docker build --platform linux/amd64 -t mu-llama:latest .

# Start with NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install system dependencies in smaller groups
RUN apt-get update && \
    apt-get install -y --no-install-recommends sudo wget curl && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends jq ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends git git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

# Install Miniconda - with directory cleanup
RUN rm -rf /opt/conda && \
    curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -afy

# Initialize conda in shell
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete

# Set up working directory and clone repositories
WORKDIR /app

# Set the default command
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/opt/conda/bin/conda", "run", "-n", "mu-llama"]
