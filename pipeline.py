from whisperx import load_model, load_audio, load_align_model, align

device = "cuda"
audio_file = "audio.mp3"
batch_size = 16
compute_type = "float16"

def transcribe():
    model = load_model("large-v2", device, compute_type = compute_type)
    audio = load_audio(r"/content/converted.mp3")
    transcript = model.transcribe(audio, batch_size= batch_size)
    return transcript