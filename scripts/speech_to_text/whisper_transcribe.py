import whisper

# Load the Whisper model
model = whisper.load_model("base")  # You can use "base", "small", "medium", "large"

# Load and transcribe the audio file
audio_file_path = r"data\audio.mp3"  # Update with your audio file path
result = model.transcribe(audio_file_path)

# Print the transcription
print("Transcription:")
print(result['text'])
