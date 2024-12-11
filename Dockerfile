# Building the image: docker build -t mu-llama:latest .

# Running with GPU support docker run --gpus all -it mu-llama:latest

# Start with NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"
ENV CONDA_AUTO_UPDATE_CONDA=false

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo \
        wget \
        curl \
        jq \
        ffmpeg \
        git \
        git-lfs \
        && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /root/miniconda3 && \
    rm ~/miniconda.sh

# Initialize conda in bash
RUN conda init bash && \
    echo "conda activate mu-llama" >> ~/.bashrc

# Set up working directory
WORKDIR /app

# Copy local files first (if you have any local modifications)
COPY . /app/

# Clone MU-LLaMA repository
WORKDIR /app
RUN git clone https://github.com/SahilJ97/MU-LLaMA.git && \
    cd MU-LLaMA/MU-LLaMA && \
    git clone https://huggingface.co/mu-llama/MU-LLaMA ckpts

# Create and activate conda environment
RUN conda create -n mu-llama python=3.9 -y && \
    conda run -n mu-llama conda install -y \
        pytorch==2.1.0 \
        torchvision==0.16.0 \
        torchaudio==2.1.0 \
        pytorch-cuda=11.8 \
        'ffmpeg<5' \
        -c pytorch -c nvidia

# Install Python dependencies
RUN conda run -n mu-llama pip install \
        pytorchvideo==0.1.5 \
        ftfy \
        timm \
        einops

# Install requirements.txt
WORKDIR /app/MU-LLaMA
RUN conda run -n mu-llama pip install -r requirements.txt

# Set working directory to the main code directory
WORKDIR /app/MU-LLaMA/MU-LLaMA

# Set the default command to activate conda environment
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["conda", "run", "-n", "mu-llama"]
CMD ["python"]  # Replace with your actual script