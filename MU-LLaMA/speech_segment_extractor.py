import librosa
import numpy as np
from pydub import AudioSegment
from pathlib import Path


class AudioSegmenter:
    def __init__(self):
        self.window_size = 2048
        self.hop_length = 512

    def load_audio(self, audio_path):
        """Load audio file using librosa."""
        y, sr = librosa.load(str(audio_path), sr=None)
        return y, sr

    def detect_linguistic_content(self, y, sr):
        """
        Detect segments likely to contain linguistic content using spectral features.
        Returns a boolean mask of frames.
        """
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr,
                                                     n_fft=self.window_size,
                                                     hop_length=self.hop_length)

        flatness = librosa.feature.spectral_flatness(y=y,
                                                     n_fft=self.window_size,
                                                     hop_length=self.hop_length)

        rms = librosa.feature.rms(y=y, frame_length=self.window_size,
                                  hop_length=self.hop_length)

        zcr = librosa.feature.zero_crossing_rate(y,
                                                 frame_length=self.window_size,
                                                 hop_length=self.hop_length)

        # Normalize features
        centroid = (centroid - np.mean(centroid)) / np.std(centroid)
        flatness = (flatness - np.mean(flatness)) / np.std(flatness)
        rms = (rms - np.mean(rms)) / np.std(rms)
        zcr = (zcr - np.mean(zcr)) / np.std(zcr)

        # Combined heuristic for speech detection
        is_speech = np.logical_and.reduce([
            centroid[0] > -2.0,  # Was -1.5
            centroid[0] < 2.0,  # Was 1.5
            flatness[0] < 1.3,  # Was 1.0
            rms[0] > -1.8,  # Was -1.5
            zcr[0] < 1.5
        ])

        return is_speech

    def smooth_mask(self, mask, min_length_sec=0.7, sr=22050):  # Was 0.5
        """Smooth the detection mask to avoid rapid switching."""
        min_length = int(min_length_sec * sr / self.hop_length)

        from scipy.ndimage import binary_closing, binary_opening

        # Remove short non-speech segments
        mask = binary_closing(mask, structure=np.ones(min_length))

        # Remove short speech segments
        mask = binary_opening(mask, structure=np.ones(min_length))

        return mask

    def find_speech_segments(self, mask, hop_length):
        """Convert mask to list of segments [(start_sample, end_sample),...]."""
        # Add sentinels to handle edges
        mask_padded = np.concatenate([[False], mask, [False]])

        # Find rising and falling edges
        rises = np.where(np.diff(mask_padded.astype(int)) == 1)[0]
        falls = np.where(np.diff(mask_padded.astype(int)) == -1)[0]

        # Convert frame indices to sample indices
        segments = [(start * hop_length, end * hop_length)
                    for start, end in zip(rises, falls)]

        return segments

    def process_audio_file(self, input_path, output_path):
        """Process audio file to keep segments with linguistic content."""
        print(f"Processing {input_path}...")

        # Load audio
        y, sr = self.load_audio(input_path)

        # Detect speech segments
        is_speech = self.detect_linguistic_content(y, sr)

        # Smooth the detection mask
        is_speech = self.smooth_mask(is_speech, min_length_sec=0.5, sr=sr)

        # Find speech segments
        segments = self.find_speech_segments(is_speech, self.hop_length)
        print(f"Found {len(segments)} segments with linguistic content: {[(float(s[0]/sr), float(s[1]/sr)) for s in segments]}")

        # Create output audio by concatenating speech segments
        import soundfile as sf
        import io

        # First save as WAV in memory
        wav_io = io.BytesIO()
        output_audio = np.concatenate([y[start:end] for start, end in segments])
        sf.write(wav_io, output_audio, sr, format='WAV')

        # Convert to MP3 using pydub
        wav_io.seek(0)
        audio_segment = AudioSegment.from_wav(wav_io)
        audio_segment.export(output_path, format="mp3")

        # Calculate statistics
        original_duration = len(y) / sr
        processed_duration = len(output_audio) / sr
        reduction = (1 - processed_duration / original_duration) * 100

        print(f"\nOriginal duration: {original_duration:.1f}s")
        print(f"Processed duration: {processed_duration:.1f}s")
        print(f"Reduction: {reduction:.1f}%")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract segments with linguistic content from audio files")
    parser.add_argument("input_path", help="Path to input audio file")
    parser.add_argument("--output_path", help="Path to output audio file (optional)")

    args = parser.parse_args()
    input_path = Path(args.input_path)

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = input_path.parent / f"{input_path.stem}_linguistic_only.mp3"

    segmenter = AudioSegmenter()
    segmenter.process_audio_file(input_path, output_path)


if __name__ == "__main__":
    main()