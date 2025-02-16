from dotenv import load_dotenv
import os
from pydub import AudioSegment
import math
import tempfile
import glob
from analyze_audio_file import multimodal_generate
from openai import OpenAI
import logging
import sys
from typing import Optional
import requests
import time
import redis
from redis_layer import RedisQueue
from speech_segment_extractor import AudioSegmenter

segmenter = AudioSegmenter()

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

load_dotenv()

# Configure Redis client and queues
ENVIRONMENT = os.environ.get('ENVIRONMENT')
REDIS_HOST = os.environ.get('REDIS_HOST')
REDIS_PORT = os.environ.get('REDIS_PORT')
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD')
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True,  # Automatically decode responses to strings instead of bytes
    socket_timeout=5,  # Add timeout to prevent hanging
    retry_on_timeout=True,  # Automatically retry on timeout
)
audio_redis_q = RedisQueue(redis_client, f'{ENVIRONMENT}_audio')

ML_BACKEND_API_KEY = os.environ.get("ML_BACKEND_API_KEY")
if not ML_BACKEND_API_KEY:
    raise ValueError("ML_BACKEND_API_KEY is not set in environment variables")

logger.info("Configuring OpenAI client...")
OPEN_AI_ORG = os.environ.get("OPEN_AI_ORG")
OPEN_AI_KEY = os.environ.get("OPEN_AI_KEY")
open_ai_client = OpenAI(
    organization=OPEN_AI_ORG,
    api_key=OPEN_AI_KEY
)


def split_audio(input_file, output_dir, clip_duration_minutes=1):
    """
    Parameters:
    input_file (str): Path to the input MP3 file
    output_dir (str): Directory where the output clips will be saved

    Returns:
    tuple: (clip_duration_minutes, number_of_clips_created)
    """
    # Load the audio file
    audio = AudioSegment.from_mp3(input_file)
    total_duration_ms = len(audio)
    total_duration_minutes = total_duration_ms / 60000

    # If clip_duration_minutes isn't provided, calculate it based on the total duration
    if clip_duration_minutes is None:
        print(f"Total duration of {input_file}: {total_duration_minutes} minutes")
        clip_duration_minutes = math.ceil(0.73 * math.log(total_duration_minutes) + 0.85)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Split the audio into clips
    num_clips = math.ceil(total_duration_minutes / clip_duration_minutes)
    clip_duration_ms = clip_duration_minutes * 60000
    print(f"Number of clips: {num_clips}")
    for i in range(num_clips):
        start_ms = i * clip_duration_ms
        end_ms = min((i + 1) * clip_duration_ms, total_duration_ms)

        # Extract the clip
        clip = audio[start_ms:end_ms]

        # Generate output filename
        output_filename = os.path.join(
            output_dir,
            f"clip_{i + 1:03d}.mp3"
        )

        # Export the clip
        clip.export(output_filename, format="mp3")

    return clip_duration_minutes, num_clips


def transcribe_audio(clip_path):
    with open(clip_path, "rb") as audio_file:
        transcript = open_ai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcript.text

def download_file(
        download_url: Optional[str],
        tmp_dir_name: str,
        filename: str
):
    download_path = os.path.join(tmp_dir_name, filename)

    headers = {'X-API-Key': ML_BACKEND_API_KEY}
    response = requests.get(download_url, headers=headers, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes

    with open(download_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return download_path

def analyze_audio(job_data: dict):
    download_url = job_data.get('download_url')
    prompts = job_data.get('prompts')
    logger.info(f"Analyzing audio with data {job_data}")
    
    if not prompts:
        raise ValueError("Data for job has no value for 'prompts'")

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        logger.info(f'Created temporary directory {tmp_dir_name}')

        # Download file
        if download_url.lower().endswith(".mp3"):
            filename = "audio.mp3"
        else:
            filename = "video.mp4"
        download_path = download_file(download_url, tmp_dir_name, filename)

        # Convert mp4 to mp3 if needed
        if filename == "video.mp4":
            audio = AudioSegment.from_file(download_path, format="mp4")
            mp3_path = os.path.join(tmp_dir_name, "audio.mp3")
            audio.export(mp3_path, format="mp3")
            download_path = mp3_path

        # Split the audio into clips
        clip_duration_minutes, num_clips = split_audio(download_path, tmp_dir_name)

        # Iterate through the clips in chronological order, collecting analysis data
        all_analysis_data = []
        ordered_clips = sorted(glob.glob(f"{tmp_dir_name}/clip_*.mp3"))
        for clip_path in ordered_clips:

            logger.info(f"Processing clip {clip_path}")
            # Transcribe audio
            speech_only_clip_path = clip_path.replace(".mp3", "_linguistic_only.mp3")
            segmenter.process_audio_file(clip_path, speech_only_clip_path)
            # speech_only_hash = hash_file(speech_only_clip_path)  # TODO: locally cache transcriptions by clip hash
            clip_transcription = transcribe_audio(speech_only_clip_path)
            logger.info(f"Obtained transcription: {clip_transcription}")

            for prompt in prompts:
                clip_analyses = []
                output = multimodal_generate(
                    clip_path,
                    1,
                    prompt,
                    100,
                    20.0,
                    0.0,
                    512,
                    0.6,
                    0.8
                )
                logger.info(f"Generated output for prompt {prompt}: {output}")
                if isinstance(output, list) and isinstance(output[0], str):
                    output = " ".join(output)
                elif isinstance(output, str):
                    pass
                else:
                    logger.info(f"Uh oh...output for prompt {prompt} is not a string or list of strings!")
                    pass
                clip_analyses.append({
                    "prompt": prompt,
                    "output": output
                })
            all_analysis_data.append({
                "clip_analyses": clip_analyses,
                "clip_transcription": clip_transcription
            })

    return {
        "clip_duration_minutes": clip_duration_minutes,
        "analysis_data": all_analysis_data,
    }

def main_loop():
    logger.info("Initializing worker...")
    while True:
        # Check the audio task queue
        audio_job = audio_redis_q.get_next_job()
        if audio_job is not None:
            logger.info(f"Processing audio job {audio_job.id}")
            try:
                result = analyze_audio(audio_job.data)
                audio_redis_q.complete_job(audio_job.id, result)
            except Exception as e:
                audio_redis_q.report_job_failure(audio_job.id, error=str(e))
                raise e
        else:
            logger.info("No jobs found; sleeping for 90s...")
            time.sleep(90)


if __name__ == "__main__":
    main_loop()  # Any unhandled exception will crash the worker