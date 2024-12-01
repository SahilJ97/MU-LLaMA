#Usage: python audio_analysis.py --input-clips "$INPUT_AUDIO_CLIPS" \
#    --output-files "$CLIP_AUDIO_ANALYSIS_FILES" --analysis-types general,role \
#    $(for role in "${CREATIVE_ROLES[@]}"; do echo "--role $role"; done


import argparse
import sys
from pathlib import Path
import torch.cuda
import os
import llama
from util.misc import *
from data.utils import load_and_transform_audio_data
import json

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:21,garbage_collection_threshold:0.6'

MERT_PATH = "m-a-p/MERT-v1-330M"
MODEL_PATH = "./ckpts/checkpoint.pth"
LLAMA_DIR = "./ckpts/LLaMA"
LLAMA_TYPE = "7B"
KNN_DIR = "./ckpts"



def parse_comma_separated_paths(paths_str):
    """Parse comma-separated path strings into a list of Path objects."""
    if not paths_str:
        return []
    return [Path(path.strip()) for path in paths_str.split(',')]


def validate_args(input_clips, output_files):
    """Validate command-line arguments according to specified constraints."""

    # Check that the number of input and output paths match
    if len(input_clips) != len(output_files):
        raise ValueError(
            f"Number of input paths ({len(input_clips)}) must match "
            f"number of output paths ({len(output_files)})"
        )
    
    for analysis_type in args.analysis_types:
        if analysis_type not in ["general", "role"]:
            raise ValueError("Invalid analysis type. Must be 'general' or 'role'")

    if "role" in args.analysis_types and len(args.role) == 0:
        raise ValueError("Role analysis requires at least one role to be specified")


parser = argparse.ArgumentParser(description='Process multiple audio clips')

# Input group to ensure mutual exclusivity
input_group = parser.add_mutually_exclusive_group()
input_group.add_argument(
    '--input-clips',
    type=parse_comma_separated_paths,
    help='Comma-separated list of input audio clip paths'
)

parser.add_argument(
    '--output-files',
    type=parse_comma_separated_paths,
    required=True,
    help='Comma-separated list of output file paths'
)

parser.add_argument(
    '--analysis-types',
    type=lambda x: [t.strip() for t in x.split(',')],
    required=True,
    help='Comma-separated list of analysis types (recognized types: general, role)'
)

parser.add_argument(
    '--role',
    action='append',
    default=[],
    help='Creative role to analyze for. Can be specified multiple times.'
)

parser.add_argument(
    '--from-video',
    action='store_true',
    help='Indicate if the input clips are extracted from video'
)


args = parser.parse_args()
validate_args(
    args.input_clips,
    args.output_files
)

first_sentence = "Analyze this audio in detail"
if args.from_video:
    first_sentence = "This audio was extracted from a video. Analyze it in detail"

general_prompt = ".\n".join([
    first_sentence,
    "It may contain any types of content: music, dialogue, narration, or other sounds",
    "Start with descriptive, surface-level observations about what's happening in the clip",
    "Then delve into deeper aspects: artistic styles, underlying sentiments, themes, etc.",
    "Avoid inaccuracies and making guesses about intentions or meaning, but do provide claims and interpretations that are well-supported by observed details",
])

role_prompt = ".\n".join([
    first_sentence,
    "It may contain any types of content: music, dialogue, narration, or other sounds",
    f"Focus exclusively on contributions from the following role(s): {', '.join(args.role)}",
    "Use the appropriate technical terms and language; your response should befit a creative professional",
    "Your observations should cover all notable styles, specific techniques, and creative decisions that fall under the above role(s)",
    "Avoid inaccuracies and making guesses about intentions or meaning, but do provide claims and interpretations that are well-supported by observed details",
])

prompts = []
if "general" in args.analysis_types:
    prompts.append((general_prompt, "general"))
if "role" in args.analysis_types:
    prompts.append((role_prompt, "role"))

model = llama.load(MODEL_PATH, LLAMA_DIR, mert_path=MERT_PATH, knn=True, knn_dir=KNN_DIR, llama_type=LLAMA_TYPE)
with torch.cuda.amp.autocast():
    model.eval()
torch.cuda.empty_cache()

def multimodal_generate(
        audio_path,
        audio_weight,
        prompt,
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
    encoded_prompts = [model.tokenizer.encode(prompt, bos=True, eos=False)]
    with torch.cuda.amp.autocast():
        results = model.generate(inputs, encoded_prompts, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p,
                                     cache_size=cache_size, cache_t=cache_t, cache_weight=cache_weight)
    text_output = results[0].strip()
    return text_output

for input_clip, out_file in zip(args.input_clips, args.output_files):
    input_clip = str(input_clip)
    output_data = {"input_file": input_clip}
    for prompt, prompt_type in prompts:
        prompt = "Describe this audio:"  # temp!
        output_text = multimodal_generate(input_clip, 1, prompt, 100, 20.0, 0.0, 256, 0.6, 0.8)
        output_data[prompt_type] = {
            "prompt": prompt,
            "output": output_text,
        }
        if prompt_type == "role":
            output_data[prompt_type]["roles"] = args.role
    
    with open(f"{out_file}", "w") as f:
        json.dump(output_data, f)
