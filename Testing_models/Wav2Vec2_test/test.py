import torch
import pyaudio
import numpy as np
from collections import deque
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}\n")

# Load model
print("Loading Wav2Vec2 model...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
model.eval()
print("✅ Model loaded!\n")

# Settings
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.02

# Setup microphone
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paFloat32,
    channels=1,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=1600
)

print("🎤 Listening... (Press Ctrl+C to stop)\n")

audio_buffer = deque(maxlen=10)

try:
    while True:
        # Read audio
        data = stream.read(1600, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.float32)
        audio_buffer.append(audio_data)
        
        # When buffer is full
        if len(audio_buffer) == 10:
            audio = np.concatenate(list(audio_buffer))
            
            # Check if not silence
            if np.abs(audio).mean() > SILENCE_THRESHOLD:
                print("🔊 Transcribing...")
                
                # Process and transcribe
                inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")["input_values"].to(device)
                
                with torch.no_grad():
                    logits = model(inputs).logits
                
                predicted_ids = torch.argmax(logits, dim=-1)
                text = processor.batch_decode(predicted_ids)[0].strip()
                
                if text:
                    print(f"You said: {text}\n")
            
            audio_buffer.clear()
            
except KeyboardInterrupt:
    print("\n\nStopped listening.")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
