import logging
import math
import torch
import torchaudio

def load_and_transform_audio_data(
    audio_paths,
    sample_rate=24000
):
    audios = []
    for path in audio_paths:
        format = path.split(".")[-1]
        print("Loading audio with format:", format)
        waveform, sr = torchaudio.load(path, format=format)
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=sample_rate)
        waveform = torch.mean(waveform, 0)
        audios.append(waveform)
    return torch.stack(audios, dim=0)

# Commented out as it's unused in audio processing workflow
# def load_and_transform_text(text, device):
#     if text is None:
#         return None
#     tokenizer = SimpleTokenizer(bpe_path=BPE_PATH)
#     tokens = [tokenizer(t).unsqueeze(0).to(device) for t in text]
#     tokens = torch.cat(tokens, dim=0)
#     return tokens