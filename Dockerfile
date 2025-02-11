FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV PYTHONPATH=/app

# Install system dependencies in a single layer to minimize image size
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo \
        wget \
        curl \
        jq \
        ffmpeg \
        git \
        git-lfs \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda with proper cleanup
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean --all --force-pkgs-dirs -y

# Create conda environment and install PyTorch separately
RUN conda create -n mu-llama python=3.9 -y && \
    conda clean --all --force-pkgs-dirs -y

# Install PyTorch and CUDA first
RUN conda run -n mu-llama conda install -y pytorch==2.5.0 pytorch-cuda=11.8 -c pytorch -c nvidia && \
    conda clean --all --force-pkgs-dirs -y

# Install vision and audio packages
RUN conda run -n mu-llama conda install -y torchvision==0.20.0 torchaudio==2.0.2 -c pytorch -c nvidia && \
    conda clean --all --force-pkgs-dirs -y

# Set up working directory and copy application files
WORKDIR /app/MU-LLaMA/MU-LLaMA
COPY . /app/MU-LLaMA
RUN conda run -n mu-llama pip install -r /app/MU-LLaMA/requirements.txt

# Only install git-lfs at build time, ckpts clone will happen at runtime
RUN git lfs install

RUN chmod +x /app/MU-LLaMA/entrypoint.sh

# Set the entrypoint to our script
ENTRYPOINT ["/app/MU-LLaMA/entrypoint.sh"]