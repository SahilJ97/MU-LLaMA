#Usage: python audio_analysis.py --input-clips "$INPUT_AUDIO_CLIPS" \
#    --output-files "$CLIP_AUDIO_ANALYSIS_FILES" --analysis-types general,role \
#    $(for role in "${CREATIVE_ROLES[@]}"; do echo "--role $role"; done


import argparse
import sys
from pathlib import Path


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
    type=parse_comma_separated_paths,
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
    "The audio may contain any types of content: music, dialogue, narration, or other sounds",
    "Start with descriptive, surface-level observations about what's happening in the clip",
    "Then delve into deeper aspects, such as artistic styles, underlying sentiments, and themes",
    "Avoid making guesses about intentions or meaning, but do provide claims and interpretations that are well-supported by observed details",
])

role_prompt = ".\n".join([
    first_sentence,
    "The audio may contain any types of content: music, dialogue, narration, or other sounds",
    "Focus exclusively on contributions from the following role(s): {', '.join(args.role)}",
    "Your observations should cover all notable styles, techniques, and creative decisions that fall under the above role(s)",
    "Avoid making guesses about intentions or meaning, but do provide claims and interpretations that are well-supported by observed details",
])

