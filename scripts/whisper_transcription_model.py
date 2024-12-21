import whisper

# Load the Whisper model
model = whisper.load_model("base")

# Path to your audio file
audio_path = r"#"  # Replace with your file path

# Transcribe the audio
result = model.transcribe(audio_path)

# Print the transcription result
print(f"Transcription: {result['text']}")
print(f"Language Detected: {result['language']}")
print(f"Duration: {result['duration']} seconds")