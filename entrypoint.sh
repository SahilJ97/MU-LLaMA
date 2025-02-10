#!/bin/bash
set -e

echo "Downloading model checkpoints..."
cd /app/MU-LLaMA/MU-LLaMA
git clone https://huggingface.co/mu-llama/MU-LLaMA ckpts

# Execute the main application
exec conda run -n mu-llama python worker.py