import os
import torch
import numpy as np
from pathlib import Path
from transformers import AutoProcessor, WhisperModel

AUDIO_DIR = "/content/drive/MyDrive/Talking Heads/Talking_heads/DATASET/audio_clips"        # change to your folder
OUTPUT_DIR = "audio_embeddings"  # .npy files saved here
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperModel.from_pretrained("openai/whisper-large-v3", torch_dtype=torch.float16).to(DEVICE)
model.eval()

SUPPORTED = {".m4a", ".wav", ".mp3", ".flac", ".ogg"}

def load_audio(path: str) -> np.ndarray:
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", tmp_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )
    import soundfile as sf
    audio, _ = sf.read(tmp_path)
    os.remove(tmp_path)
    return audio.astype(np.float32)

audio_files = [p for p in Path(AUDIO_DIR).iterdir() if p.suffix.lower() in SUPPORTED]
print(f"Found {len(audio_files)} audio files.")

with torch.no_grad():
    for fpath in audio_files:
        out_path = Path(OUTPUT_DIR) / (fpath.stem + ".npy")
        if out_path.exists():
            print(f"Skipping {fpath.name} (already done)")
            continue

        print(f"Processing {fpath.name} ...")
        audio = load_audio(str(fpath))

        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True
        )
        input_features = inputs.input_features.to(DEVICE, dtype=torch.float16)  # (1, 128, T)
        attention_mask = inputs.attention_mask.to(DEVICE, dtype=torch.float16)

        encoder_out = model.encoder(
            input_features,
            attention_mask=attention_mask
        )
        # Shape: (1, T', 1280)  — T' ~ audio_seconds * 50 / 2
        embedding = encoder_out.last_hidden_state.squeeze(0).cpu().numpy()

        np.save(out_path, embedding)
        print(f"  Saved {embedding.shape} -> {out_path}")

print("Done.")