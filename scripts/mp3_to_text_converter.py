import speech_recognition as sr
from pydub import AudioSegment

def mp3_to_text(mp3_file):
    # Convert MP3 to WAV
    sound = AudioSegment.from_mp3(mp3_file)
    wav_file = "converted_audio.wav"
    sound.export(wav_file, format="wav")

    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Open the converted WAV file for recognition
    with sr.AudioFile(wav_file) as source:
        print("Processing the audio file... Please wait.")
        audio = recognizer.record(source)  # Capture the audio

    try:
        # Recognize speech using Google Web Speech API
        print("Recognizing speech...")
        text = recognizer.recognize_google(audio)  # Converts audio to text
        print("Text from audio:", text)
    except sr.UnknownValueError:
        print("Sorry, could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

# Path to your MP3 file
mp3_file = "your_audio_file.mp3"  # Replace with the path to your MP3 file

# Call the function to convert MP3 to text
mp3_to_text(mp3_file)
