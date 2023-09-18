from whisperx import load_model, load_audio, load_align_model, align

device = "cuda"
audio_file = "audio.mp3"
batch_size = 16
compute_type = "float16"

def transcribe():
    model = load_model("large-v2", device, compute_type = compute_type)
    audio = load_audio(r"/content/converted.mp3")
    transcript = model.transcribe(audio, batch_size= batch_size)
    model_a, metadata = load_align_model(language_code=transcript["language"], device=device)
    result = align(transcript["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    return result

if __name__ == "__main__":
    print(transcribe)