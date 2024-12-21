import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Load the audio file (use .wav format for compatibility with speech_recognition)
audio_file_path = r"data\audio.mp3"  # Update with your audio file path

# Load audio file
with sr.AudioFile(audio_file_path) as source:
    print("Listening...")
    audio_data = recognizer.record(source)

# Recognize speech using Google Web Speech API
try:
    print("Recognizing...")
    text = recognizer.recognize_google(audio_data)  # Using Google's API
    print("Transcription:")
    print(text)
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand the audio")
except sr.RequestError as e:
    print(f"Could not request results from Google Speech Recognition service; {e}")
