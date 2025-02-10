from flask import Flask, request
from functools import wraps
from dotenv import load_dotenv
import os
from pydub import AudioSegment
import math
import tempfile
import glob
from analyze_audio_file import multimodal_generate
from supabase import create_client, Client
from openai import OpenAI
from flask import jsonify

with tempfile.TemporaryDirectory() as tmpdirname:
     print('Created temporary directory', tmpdirname)

load_dotenv()

app = Flask(__name__)

# Client credentials for verification
ACCESS_KEY = os.environ.get("ACCESS_KEY")
SECRET_KEY = os.environ.get("SECRET_KEY")

# Configure Supabase client
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_SERVICE_ROLE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Configure OpenAI client
OPEN_AI_ORG = os.environ.get("OPEN_AI_ORG")
OPEN_AI_KEY = os.environ.get("OPEN_AI_KEY")
open_ai_client = OpenAI(
    organization=OPEN_AI_ORG,
    api_key=OPEN_AI_KEY
)


def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or ACCESS_KEY != auth.username or SECRET_KEY != auth.password:
            return "Unauthorized", 403
        return f(*args, **kwargs)
    return decorated


def split_audio(input_file, output_dir):
    """
    Parameters:
    input_file (str): Path to the input MP3 file
    output_dir (str): Directory where the output clips will be saved

    Returns:
    tuple: (clip_duration_minutes, number_of_clips_created)
    """
    # Load the audio file
    audio = AudioSegment.from_mp3(input_file)

    # Get the duration in minutes
    total_duration_ms = len(audio)
    total_duration_minutes = total_duration_ms / 60000
    print(f"Total duration of {input_file}: {total_duration_minutes} minutes")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Calculate number of clips
    clip_duration_minutes = math.ceil(0.73 * math.log(total_duration_minutes) + 0.85)
    clip_duration_ms = clip_duration_minutes * 60000
    num_clips = math.ceil(total_duration_minutes / clip_duration_minutes)
    print(f"Number of clips: {num_clips}")

    # Split the audio into clips
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

@app.route('/analyze-audio')
@require_auth
def hello():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json()
    storage_bucket = data.get('storage_bucket')
    storage_relative_path = data.get('storage_relative_path')
    prompts = data.get('prompts')

    if not all([storage_bucket, storage_relative_path, prompts]):
        return jsonify({"error": "storage_bucket, storage_relative_path, and prompts must all be specified"}), 400

    # Convert prompts to a list of (prompt_type, prompt) tuples
    prompts = list(prompts.items())

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        print('Created temporary directory', tmp_dir_name)

        # Download the audio file to ./audio.mp3
        download_path = os.path.join(tmp_dir_name, "audio.mp3")
        with open(download_path, "wb+") as f:
            response = supabase.storage.from_(storage_bucket).download(
                storage_relative_path
            )
            f.write(response)

        # Split the audio into clips
        clip_duration_minutes, num_clips = split_audio(download_path, tmp_dir_name)

        # Iterate through the clips in chronological order, collecting analysis data
        all_analysis_data = []
        ordered_clips = sorted(glob.glob(f"{tmp_dir_name}/clip_*.mp3"))
        for clip_path in ordered_clips:
            clip_transcription = transcribe_audio(clip_path)
            clip_analyses = {}
            outputs = multimodal_generate(
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
            for (prompt_type, prompt), output in zip(prompts, outputs):
                clip_analyses[prompt_type] = {
                    "prompt": prompt,
                    "output": output
                }
            all_analysis_data.append({
                "clip_analyses": clip_analyses,
                "clip_transcription": clip_transcription
            })

    return jsonify({
        "clip_duration_minutes": clip_duration_minutes,
        "analysis_data": all_analysis_data
    })

if __name__ == '__main__':
    app.run(debug=True)