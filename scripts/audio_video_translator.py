from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import pipeline
from moviepy import VideoFileClip, AudioFileClip
from gtts import gTTS
from pydub import AudioSegment

# Load environment variables from the 'config/env' file
load_dotenv(dotenv_path='config/env')

# Initialize Google Generative AI for translation
google_chat_model = ChatGoogleGenerativeAI(
    model='gemini-1.5-flash',
    api_key = os.getenv('GOOGLE_API_KEY')
)

# Load the Whisper model pipeline for speech-to-text
speech_to_text = pipeline(
    "automatic-speech-recognition", 
    model="openai/whisper-large-v3",
    device=-1  # Use GPU if available
)

# Function to process audio file
def transcribe_audio(file_path):
    result = speech_to_text(file_path)
    return result['text']

# Function to translate text using Google Generative AI
def translate_text(text, target_language):
    prompt = f"Translate the following text to, {target_language}:\n\n{text}"
    response = google_chat_model.predict(prompt)
    return response

# Function to convert text to speech using gTTS
def convert_to_audio(text, output_file="output.mp3"):
    tts = gTTS(text, lang='hi')  # Change 'lang' as needed for the target language
    tts.save(output_file)
    return output_file

# Function to adjust audio speed to match duration
def adjust_audio_speed(audio_file, target_duration, output_file="adjusted_audio.mp3"):
    audio = AudioSegment.from_file(audio_file)
    current_duration = audio.duration_seconds
    speed_factor = current_duration / target_duration

    # Adjust speed
    adjusted_audio = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * speed_factor)
    }).set_frame_rate(audio.frame_rate)

    adjusted_audio.export(output_file, format="mp3")
    return output_file

# Function to convert MP4 video to MP3 audio
def convert_video_to_audio(video_file_path, output_audio_file="output_audio.mp3"):
    video = VideoFileClip(video_file_path)
    video.audio.write_audiofile(output_audio_file)
    return output_audio_file

# Function to replace the audio in the original video with translated audio
def replace_audio_in_video(original_video_path, translated_audio_path, output_video_path="output_video.mp4"):
    video = VideoFileClip(original_video_path)
    translated_audio = AudioFileClip(translated_audio_path)
    video_with_new_audio = video.set_audio(translated_audio)
    video_with_new_audio.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
    return output_video_path

# Main program
if __name__ == "__main__":
    # Define the base folder where the video/audio files are stored
    data_folder = 'data/'

    file_type = input("Enter the file type (audio/video): ").strip().lower()
    
    if file_type == "video":
        video_file_name = input("Enter the name of your MP4 file (e.g., my_video.mp4): ").strip()
        video_file_path = os.path.join(data_folder, video_file_name)

        # Check if the video file exists in the data folder
        if not os.path.isfile(video_file_path):
            print(f"Error: The file '{video_file_name}' was not found in the 'data' folder.")
            exit(1)

        audio_file = convert_video_to_audio(video_file_path)
        print(f"\nAudio extracted from video and saved as {audio_file}")
        
    elif file_type == "audio":
        audio_file_name = input("Enter the name of your audio file (e.g., my_audio.mp3): ").strip()
        audio_file_path = os.path.join(data_folder, audio_file_name)

        # Check if the audio file exists in the data folder
        if not os.path.isfile(audio_file_path):
            print(f"Error: The file '{audio_file_name}' was not found in the 'data' folder.")
            exit(1)

        print(f"\nAudio file path provided: {audio_file_path}")
        audio_file = audio_file_path

    else:
        print("Invalid file type! Please enter either 'audio' or 'video'.")
        exit(1)

    transcription = transcribe_audio(audio_file)
    print("\nTranscription:")
    print(transcription)

    target_language = input("\nEnter the target language (e.g., hindi, kannada): ").strip()
    translated_text = translate_text(transcription, target_language)
    print(f"\nTranslated Text ({target_language}):")
    print(translated_text)

    print("\nGenerating audio for the translated text...")

    # Define the output folder for translated audio
    output_folder = os.path.join(data_folder, 'output')
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the translated audio file inside 'data/output/'
    translated_audio_file = os.path.join(output_folder, "translated_audio.mp3")
    convert_to_audio(translated_text, output_file=translated_audio_file)

    if file_type == "video":
        print("\nAdjusting the translated audio duration...")
        original_video = VideoFileClip(video_file_path)
        original_audio_duration = original_video.audio.duration
        adjusted_audio_file = adjust_audio_speed(translated_audio_file, original_audio_duration)

        print("\nReplacing the audio in the video...")
        final_video_file = replace_audio_in_video(video_file_path, adjusted_audio_file, output_video_path="final_video.mp4")
        print(f"\nVideo with translated audio saved as {final_video_file}")
    else:
        print(f"\nTranslated audio saved as {translated_audio_file}")
