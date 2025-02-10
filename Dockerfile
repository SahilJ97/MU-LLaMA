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

# Create new conda environment and install dependencies in one layer
RUN conda create -n mu-llama python=3.9 -y && \
    conda run -n mu-llama conda install -y \
        pytorch==2.5.0 \
        torchvision==0.20.0 \
        torchaudio==2.5.0 \
        pytorch-cuda=11.8 \
        -c pytorch -c nvidia && \
    conda run -n mu-llama pip install flash-attn --no-build-isolation && \
    conda clean --all --force-pkgs-dirs -y

# Set up working directory and copy application files
WORKDIR /app
COPY . .
RUN conda run -n visual pip install -r requirements.txt

# Initialize git-lfs and install checkpoints directory
RUN git lfs install && \
    git clone https://huggingface.co/mu-llama/MU-LLaMA MU-LLaMA/ckpts

# Set the default command to activate conda environment and launch service
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["conda", "run", "-n", "mu-llama"]
CMD ["python", "MU-LLaMA/worker.py"]