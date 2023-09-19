import os
from whisperx import load_model, load_audio, load_align_model, align, DiarizationPipeline
from dotenv import load_dotenv
import gc
import torch

load_dotenv()
TOKEN = os.getenv("HF_TOKEN")

device = "cuda"
audio_file = "converted.mp3"
batch_size = 16
compute_type = "float16"
audio = load_audio(r"converted.mp3")

def transcribe():
    model = load_model("large-v2", device, compute_type = compute_type)
    transcript = model.transcribe(audio, batch_size= batch_size)
    model_a, metadata = load_align_model(language_code=transcript["language"], device=device)
    result = align(transcript["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    gc.collect()
    torch.cuda.empty_cache() 
    del model_a
    del model
    return result

def diarize():
    model = DiarizationPipeline(use_auth_token= TOKEN, device = device)
    segments = model(audio)
    return segments

if __name__ == "__main__":
    print(diarize())