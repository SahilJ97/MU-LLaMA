import argparse
import sys
from pathlib import Path
import torch.cuda
import os
import llama
from util.misc import *
from data.utils import load_and_transform_audio_data
import json
import gc

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:21,garbage_collection_threshold:0.6'

MERT_PATH = "m-a-p/MERT-v1-330M"
MODEL_PATH = "./ckpts/checkpoint.pth"
LLAMA_DIR = "./ckpts/LLaMA"
LLAMA_TYPE = "7B"
KNN_DIR = "./ckpts"

print("Loading model...")
model = llama.load(MODEL_PATH, LLAMA_DIR, mert_path=MERT_PATH, knn=True, knn_dir=KNN_DIR, llama_type=LLAMA_TYPE)
with torch.cuda.amp.autocast():
    model.eval()
torch.cuda.empty_cache()

def multimodal_generate(
        audio_path,
        audio_weight,
        prompts,
        cache_size,
        cache_t,
        cache_weight,
        max_gen_len,
        gen_t,
        top_p
):
    inputs = {}
    audio = load_and_transform_audio_data([audio_path])
    inputs['Audio'] = [audio, audio_weight]
    outputs = []
    for prompt in prompts:
        formatted_prompt = llama.format_prompt(prompt)
        encoded_prompts = [model.tokenizer.encode(formatted_prompt, bos=True, eos=False)]
        with torch.cuda.amp.autocast():
            results = model.generate(inputs, encoded_prompts, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p,
                                         cache_size=cache_size, cache_t=cache_t, cache_weight=cache_weight)
        text_output = results[0].strip()
        outputs.append(text_output)
    return outputs
