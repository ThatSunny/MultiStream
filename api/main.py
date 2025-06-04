from flask import Flask, render_template, request, send_from_directory, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
from groq import Groq
from moviepy.editor import VideoFileClip
from gtts import gTTS
from pydub import AudioSegment
import subprocess
import tempfile

# Load environment variables
load_dotenv(dotenv_path='config/.env')

# Flask app initialization
app = Flask(__name__)
CORS(app) 
app.config['UPLOAD_FOLDER'] = 'data/uploads/'
OUTPUT_FOLDER = 'data/output/'  # Output folder for translated audio and video
TEMP_FOLDER = 'data/temp/'  # Temporary folder for intermediate files
ALLOWED_EXTENSIONS = {'mp4', 'mp3'}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Max file size 50MB

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Initialize Groq client
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it in your config/.env file or as an environment variable.")

client = Groq(
    api_key=groq_api_key
)

# Function to transcribe audio using Groq Whisper
def transcribe_audio(file_path):
    try:
        with open(file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=file,  # Pass the file object directly
                model="whisper-large-v3",
                response_format="text"
            )
        return transcription
    except Exception as e:
        app.logger.error(f"Error during transcription: {str(e)}")
        raise

# Function to translate text using Groq Chat Completion
def translate_text(text, target_language):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a professional translator. Translate the given text to {target_language}. Only return the translated text, no additional comments or explanations."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            model="llama3-8b-8192",  # You can also use "mixtral-8x7b-32768" or other available models
            temperature=0.1,
            max_tokens=1024,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        app.logger.error(f"Error during translation: {str(e)}")
        raise

# Function to convert text to speech using gTTS
def convert_to_audio(text, output_file="output.mp3", target_language="hindi"):
    # Map common language names to gTTS language codes
    language_codes = {
        'hindi': 'hi',
        'spanish': 'es',
        'french': 'fr',
        'german': 'de',
        'italian': 'it',
        'portuguese': 'pt',
        'russian': 'ru',
        'japanese': 'ja',
        'korean': 'ko',
        'chinese': 'zh',
        'arabic': 'ar',
        'english': 'en'
    }
    
    # Get language code, default to 'hi' (Hindi) if not found
    lang_code = language_codes.get(target_language.lower(), 'hi')
    
    try:
        tts = gTTS(text, lang=lang_code)
        tts.save(output_file)
        return output_file
    except Exception as e:
        app.logger.error(f"Error during text-to-speech: {str(e)}")
        # Fallback to English if the target language is not supported
        tts = gTTS(text, lang='en')
        tts.save(output_file)
        return output_file

# Function to adjust audio speed to match duration
def adjust_audio_speed(audio_file, target_duration, output_file="adjusted_audio.mp3"):
    output_file = os.path.join(TEMP_FOLDER, output_file)
    try:
        audio = AudioSegment.from_file(audio_file)
        current_duration = audio.duration_seconds
        
        if current_duration == 0:
            raise ValueError("Audio duration is zero")
            
        speed_factor = current_duration / target_duration

        # Adjust speed using playback_speed method (more reliable)
        if speed_factor != 1.0:
            adjusted_audio = audio.speedup(playback_speed=speed_factor)
        else:
            adjusted_audio = audio

        adjusted_audio.export(output_file, format="mp3")
        return output_file
    except Exception as e:
        app.logger.error(f"Error during audio speed adjustment: {str(e)}")
        print(f"Speed adjustment failed, using original audio: {str(e)}")
        # If speed adjustment fails, return the original audio file
        return audio_file

# Function to convert MP4 video to MP3 audio
def convert_video_to_audio(video_file_path, output_audio_file="output_audio.mp3"):
    try:
        video = VideoFileClip(video_file_path)
        
        if video.audio is None:
            raise ValueError("Video file has no audio track")
            
        audio_path = os.path.join(TEMP_FOLDER, output_audio_file)  # Save to TEMP_FOLDER
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        
        video.close()
        return audio_path
    except Exception as e:
        app.logger.error(f"Error during video to audio conversion: {str(e)}")
        raise

# Function to replace the audio in the original video with translated audio
def replace_audio_in_video(original_video_path, translated_audio_path, output_video_path="output_video.mp4"):
    output_video_path = os.path.join(OUTPUT_FOLDER, output_video_path)
    try:
        # Check if files exist
        if not os.path.exists(original_video_path):
            raise FileNotFoundError(f"Original video not found: {original_video_path}")
        if not os.path.exists(translated_audio_path):
            raise FileNotFoundError(f"Translated audio not found: {translated_audio_path}")
        
        command = [
            "ffmpeg",
            "-y",  # Overwrite output files without asking
            "-i", original_video_path,
            "-i", translated_audio_path,
            "-c:v", "copy",  # Copy video stream without re-encoding
            "-c:a", "aac",   # Encode audio as AAC
            "-map", "0:v:0",  # Use the video stream from the first input
            "-map", "1:a:0",  # Use the audio stream from the second input
            "-shortest",  # End output when the shortest input ends
            output_video_path
        ]
        
        # Run FFmpeg with suppressed output
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        # Verify the output file exists and has content
        if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
            return output_video_path
        else:
            raise RuntimeError("Output video file was not created or is empty")
            
    except subprocess.CalledProcessError as e:
        print("FFmpeg command failed:")
        print(f"Return code: {e.returncode}")
        print(f"STDERR: {e.stderr}")
        raise RuntimeError(f"FFmpeg failed: {e.stderr}")
    except Exception as e:
        app.logger.error(f"Error during audio replacement: {str(e)}")
        raise

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to handle file upload and processing
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        target_language = request.form.get('target_language', 'hindi')

        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Unsupported file type. Please upload an MP4 or MP3 file.'})

        # File size limit check
        if hasattr(file, 'content_length') and file.content_length and file.content_length > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'error': 'File is too large. Maximum size is 50MB.'})

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        print(f"Processing file: {filename}")
        print(f"Target language: {target_language}")

        try:
            if filename.lower().endswith('.mp4'):
                # Process video file
                print("Step 1: Converting video to audio...")
                audio_file = convert_video_to_audio(file_path, "extracted_audio.mp3")
                
                print("Step 2: Transcribing audio...")
                transcription = transcribe_audio(audio_file)
                
                if not transcription or transcription.strip() == "":
                    return jsonify({'error': 'No speech detected in the audio. Please check your video file.'})
                
                print("Step 3: Translating text...")
                translated_text = translate_text(transcription, target_language)
                
                if not translated_text or translated_text.strip() == "":
                    return jsonify({'error': 'Translation failed. Please try again.'})
                
                print("Step 4: Converting translated text to speech...")
                translated_audio_file = os.path.join(TEMP_FOLDER, "translated_audio.mp3")
                convert_to_audio(translated_text, output_file=translated_audio_file, target_language=target_language)
                
                # Verify the TTS audio was created
                if not os.path.exists(translated_audio_file) or os.path.getsize(translated_audio_file) == 0:
                    return jsonify({'error': 'Failed to generate speech from translated text.'})
                
                print("Step 5: Getting original video duration...")
                video = VideoFileClip(file_path)
                if video.audio is None:
                    video.close()
                    return jsonify({'error': 'Video file has no audio track to replace.'})
                    
                original_duration = video.audio.duration
                video.close()
                
                print("Step 6: Adjusting audio speed...")
                adjusted_audio_file = adjust_audio_speed(
                    translated_audio_file, 
                    original_duration,
                    f"adjusted_audio_{filename.split('.')[0]}.mp3"
                )
                
                print("Step 7: Replacing audio in video...")
                final_video_file = replace_audio_in_video(
                    file_path, 
                    adjusted_audio_file, 
                    f"translated_{filename}"
                )

                # Verify final output
                if not os.path.exists(final_video_file) or os.path.getsize(final_video_file) == 0:
                    return jsonify({'error': 'Failed to create final video with translated audio.'})

                video_url = f'/data/output/{os.path.basename(final_video_file)}'
                print(f"Process completed successfully. Video URL: {video_url}")
                
                return jsonify({
                    'success': True, 
                    'video_url': video_url,
                    'original_text': transcription,
                    'translated_text': translated_text
                })
                
            elif filename.lower().endswith('.mp3'):
                # Process audio file
                print("Processing MP3 file...")
                transcription = transcribe_audio(file_path)
                
                if not transcription or transcription.strip() == "":
                    return jsonify({'error': 'No speech detected in the audio file.'})
                
                translated_text = translate_text(transcription, target_language)
                
                if not translated_text or translated_text.strip() == "":
                    return jsonify({'error': 'Translation failed. Please try again.'})
                
                translated_audio_file = os.path.join(OUTPUT_FOLDER, f"translated_{filename}")
                convert_to_audio(translated_text, output_file=translated_audio_file, target_language=target_language)
                
                audio_url = f'/data/output/{os.path.basename(translated_audio_file)}'
                return jsonify({
                    'success': True,
                    'audio_url': audio_url,
                    'original_text': transcription,
                    'translated_text': translated_text
                })
            else:
                return jsonify({'error': 'Unsupported file type. Please upload an MP4 or MP3 file.'})
                
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            app.logger.error(f"Error during file processing: {str(e)}")
            return jsonify({'error': f'Error during processing: {str(e)}'})

    return jsonify({'status': 'backend online'})

# Route to serve output files
@app.route('/data/output/<filename>')
def serve_output(filename):
    try:
        return send_from_directory(OUTPUT_FOLDER, filename)
    except Exception as e:
        app.logger.error(f"Error serving file {filename}: {str(e)}")
        return jsonify({'error': f'File not found: {filename}'}), 404

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'service': 'multilingual-translator'})

# Main program entry point
if __name__ == "__main__":
    # Check if GROQ_API_KEY is set
    if not os.getenv('GROQ_API_KEY'):
        print("Warning: GROQ_API_KEY not found in environment variables!")
        print("Please set your Groq API key in the config/.env file or as an environment variable.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)