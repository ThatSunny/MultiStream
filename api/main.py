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
import uuid
import time

# Load environment variables
load_dotenv(dotenv_path='config/.env')

# Flask app initialization
app = Flask(__name__, static_folder='../client/build', static_url_path='/')
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

# Function to generate unique filename
def generate_unique_filename(original_filename, target_language):
    """Generate a unique filename to prevent conflicts"""
    name, ext = os.path.splitext(original_filename)
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    return f"{name}_{target_language}_{timestamp}_{unique_id}{ext}"

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

# Function to convert text to speech using gTTS with improved quality
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
        print(f"Converting text to speech in {target_language} ({lang_code})")
        print(f"Text length: {len(text)} characters")
        
        # Use slower speech for better quality
        tts = gTTS(text=text, lang=lang_code, slow=False)
        
        # Save to temporary file first
        temp_file = output_file + ".temp"
        tts.save(temp_file)
        
        # Convert to higher quality MP3 using FFmpeg if available
        try:
            command = [
                "ffmpeg",
                "-y",
                "-i", temp_file,
                "-acodec", "libmp3lame",
                "-b:a", "192k",  # Higher bitrate for better quality
                "-ar", "44100",  # Standard sample rate
                "-ac", "2",      # Stereo
                output_file
            ]
            
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            
            # Remove temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            print("TTS audio converted to high quality MP3")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If FFmpeg is not available, use the original file
            print("FFmpeg not available, using original TTS quality")
            if os.path.exists(temp_file):
                os.rename(temp_file, output_file)
        
        return output_file
        
    except Exception as e:
        app.logger.error(f"Error during text-to-speech: {str(e)}")
        print(f"TTS error: {str(e)}")
        
        # Fallback to English if the target language is not supported
        try:
            print("Attempting fallback to English TTS")
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(output_file)
            print("Fallback TTS successful")
            return output_file
        except Exception as fallback_error:
            print(f"Fallback TTS also failed: {fallback_error}")
            raise fallback_error

# Function to adjust audio to match video duration using FFmpeg (better quality)
def adjust_audio_to_video_duration(audio_file, target_duration, output_file="adjusted_audio.mp3"):
    output_file = os.path.join(TEMP_FOLDER, output_file)
    try:
        # Get current audio duration using pydub
        audio = AudioSegment.from_file(audio_file)
        current_duration = audio.duration_seconds
        
        if current_duration == 0:
            raise ValueError("Audio duration is zero")
        
        print(f"Original audio duration: {current_duration:.2f}s, Target duration: {target_duration:.2f}s")
        
        # Calculate speed factor
        speed_factor = current_duration / target_duration
        print(f"Speed adjustment factor: {speed_factor:.2f}")
        
        # If the speed factor is too extreme (>2x or <0.5x), use padding/trimming instead
        if speed_factor > 2.0 or speed_factor < 0.5:
            print(f"Extreme speed factor detected ({speed_factor:.2f}). Using padding/trimming instead.")
            
            if current_duration > target_duration:
                # Trim audio to match target duration
                trimmed_audio = audio[:int(target_duration * 1000)]  # pydub uses milliseconds
                trimmed_audio.export(output_file, format="mp3")
                print("Audio trimmed to match video duration")
            else:
                # Pad audio with silence to match target duration
                silence_duration = (target_duration - current_duration) * 1000  # milliseconds
                silence = AudioSegment.silent(duration=int(silence_duration))
                padded_audio = audio + silence
                padded_audio.export(output_file, format="mp3")
                print("Audio padded with silence to match video duration")
            
            return output_file
        
        # For moderate speed adjustments, use FFmpeg for better quality
        try:
            # Use FFmpeg's atempo filter for high-quality speed adjustment
            # atempo can only handle 0.5-2.0 range, so we may need multiple passes
            tempo_value = 1.0 / speed_factor  # Invert because atempo works opposite to our speed_factor
            
            # Clamp tempo_value to valid range
            if tempo_value > 2.0:
                tempo_value = 2.0
            elif tempo_value < 0.5:
                tempo_value = 0.5
            
            print(f"Using FFmpeg atempo filter with value: {tempo_value:.2f}")
            
            command = [
                "ffmpeg",
                "-y",  # Overwrite output files
                "-i", audio_file,
                "-filter:a", f"atempo={tempo_value}",
                "-acodec", "libmp3lame",
                "-b:a", "192k",  # Maintain good audio quality
                output_file
            ]
            
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            
            # Verify the output file was created successfully
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                print("Audio speed adjusted successfully using FFmpeg")
                return output_file
            else:
                raise RuntimeError("FFmpeg did not create output file")
                
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg speed adjustment failed: {e.stderr}")
            # Fallback to pydub with conservative adjustment
            print("Falling back to pydub with conservative speed adjustment")
            
            # Use a more conservative speed adjustment
            conservative_factor = max(0.7, min(1.5, speed_factor))
            adjusted_audio = audio.speedup(playback_speed=conservative_factor)
            
            # If still not the right duration, pad or trim
            new_duration = adjusted_audio.duration_seconds
            if new_duration < target_duration:
                silence_duration = (target_duration - new_duration) * 1000
                silence = AudioSegment.silent(duration=int(silence_duration))
                adjusted_audio = adjusted_audio + silence
            elif new_duration > target_duration:
                adjusted_audio = adjusted_audio[:int(target_duration * 1000)]
            
            adjusted_audio.export(output_file, format="mp3")
            return output_file
            
    except Exception as e:
        app.logger.error(f"Error during audio duration adjustment: {str(e)}")
        print(f"Audio adjustment failed completely, using original: {str(e)}")
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

# Function to replace the audio in the original video with translated audio (improved)
def replace_audio_in_video(original_video_path, translated_audio_path, output_video_path="output_video.mp4"):
    output_video_path = os.path.join(OUTPUT_FOLDER, output_video_path)
    try:
        # Check if files exist
        if not os.path.exists(original_video_path):
            raise FileNotFoundError(f"Original video not found: {original_video_path}")
        if not os.path.exists(translated_audio_path):
            raise FileNotFoundError(f"Translated audio not found: {translated_audio_path}")
        
        print(f"Replacing audio in video...")
        print(f"Video: {original_video_path}")
        print(f"Audio: {translated_audio_path}")
        print(f"Output: {output_video_path}")
        
        command = [
            "ffmpeg",
            "-y",  # Overwrite output files without asking
            "-i", original_video_path,  # Video input
            "-i", translated_audio_path,  # Audio input
            "-c:v", "copy",  # Copy video stream without re-encoding (preserves quality)
            "-c:a", "aac",   # Encode audio as AAC
            "-b:a", "192k",  # High quality audio bitrate
            "-ar", "44100",  # Standard sample rate
            "-ac", "2",      # Stereo audio
            "-map", "0:v:0",  # Use the video stream from the first input
            "-map", "1:a:0",  # Use the audio stream from the second input
            "-shortest",     # End output when the shortest input ends
            "-avoid_negative_ts", "make_zero",  # Handle timing issues
            output_video_path
        ]
        
        print("Running FFmpeg command...")
        # Run FFmpeg with captured output for better error handling
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        print("FFmpeg completed successfully")
        
        # Verify the output file exists and has content
        if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
            print(f"Output video created successfully: {os.path.getsize(output_video_path)} bytes")
            return output_video_path
        else:
            raise RuntimeError("Output video file was not created or is empty")
            
    except subprocess.CalledProcessError as e:
        print("FFmpeg command failed:")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise RuntimeError(f"FFmpeg failed: {e.stderr}")
    except Exception as e:
        app.logger.error(f"Error during audio replacement: {str(e)}")
        print(f"Audio replacement error: {str(e)}")
        raise

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to cleanup old files to prevent disk space issues
def cleanup_old_files():
    """Clean up files older than 1 hour in temp and output folders"""
    try:
        current_time = time.time()
        one_hour = 3600  # 3600 seconds = 1 hour
        
        for folder in [TEMP_FOLDER, OUTPUT_FOLDER]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path):
                        file_age = current_time - os.path.getctime(file_path)
                        if file_age > one_hour:
                            try:
                                os.remove(file_path)
                                print(f"Cleaned up old file: {file_path}")
                            except Exception as e:
                                print(f"Failed to cleanup file {file_path}: {e}")
    except Exception as e:
        print(f"Error during cleanup: {e}")

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

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

        # Generate unique filename to prevent conflicts
        original_filename = secure_filename(file.filename)
        unique_filename = generate_unique_filename(original_filename, target_language)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        print(f"Processing file: {original_filename} -> {unique_filename}")
        print(f"Target language: {target_language}")

        # Cleanup old files before processing
        cleanup_old_files()

        try:
            if unique_filename.lower().endswith('.mp4'):
                # Process video file
                print("Step 1: Converting video to audio...")
                base_name = os.path.splitext(unique_filename)[0]
                audio_file = convert_video_to_audio(file_path, f"extracted_audio_{base_name}.mp3")
                
                print("Step 2: Transcribing audio...")
                transcription = transcribe_audio(audio_file)
                
                if not transcription or transcription.strip() == "":
                    return jsonify({'error': 'No speech detected in the audio. Please check your video file.'})
                
                print("Step 3: Translating text...")
                translated_text = translate_text(transcription, target_language)
                
                if not translated_text or translated_text.strip() == "":
                    return jsonify({'error': 'Translation failed. Please try again.'})
                
                print("Step 4: Converting translated text to speech...")
                translated_audio_file = os.path.join(TEMP_FOLDER, f"translated_audio_{base_name}.mp3")
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
                
                print("Step 6: Adjusting audio to match video duration...")
                adjusted_audio_file = adjust_audio_to_video_duration(
                    translated_audio_file, 
                    original_duration,
                    f"adjusted_audio_{base_name}.mp3"
                )
                
                print("Step 7: Replacing audio in video...")
                output_video_name = f"translated_{target_language}_{base_name}.mp4"
                final_video_file = replace_audio_in_video(
                    file_path, 
                    adjusted_audio_file, 
                    output_video_name
                )

                # Verify final output
                if not os.path.exists(final_video_file) or os.path.getsize(final_video_file) == 0:
                    return jsonify({'error': 'Failed to create final video with translated audio.'})

                video_url = f'/output/{os.path.basename(final_video_file)}'
                print(f"Process completed successfully. Video URL: {video_url}")
                
                return jsonify({
                    'success': True, 
                    'video_url': video_url,
                    'original_text': transcription,
                    'translated_text': translated_text
                })
                
            elif unique_filename.lower().endswith('.mp3'):
                # Process audio file
                print("Processing MP3 file...")
                transcription = transcribe_audio(file_path)
                
                if not transcription or transcription.strip() == "":
                    return jsonify({'error': 'No speech detected in the audio file.'})
                
                translated_text = translate_text(transcription, target_language)
                
                if not translated_text or translated_text.strip() == "":
                    return jsonify({'error': 'Translation failed. Please try again.'})
                
                base_name = os.path.splitext(unique_filename)[0]
                output_audio_name = f"translated_{target_language}_{base_name}.mp3"
                translated_audio_file = os.path.join(OUTPUT_FOLDER, output_audio_name)
                convert_to_audio(translated_text, output_file=translated_audio_file, target_language=target_language)
                
                audio_url = f'/output/{os.path.basename(translated_audio_file)}'
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
        finally:
            # Clean up uploaded file after processing
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Cleaned up uploaded file: {file_path}")
            except Exception as e:
                print(f"Failed to cleanup uploaded file: {e}")

    return jsonify({'status': 'backend online'})

# FIXED: Single route to serve output files with proper path handling
@app.route('/output/<path:filename>')
def serve_output_file(filename):
    try:
        # Get the absolute path to the output directory
        output_dir = os.path.abspath(OUTPUT_FOLDER)
        file_path = os.path.join(output_dir, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            app.logger.error(f"File not found: {file_path}")
            return jsonify({'error': f'File not found: {filename}'}), 404
            
        # Check file size
        if os.path.getsize(file_path) == 0:
            app.logger.error(f"File is empty: {file_path}")
            return jsonify({'error': f'File is empty: {filename}'}), 404
            
        print(f"Serving file: {file_path}")
        return send_from_directory(output_dir, filename)
        
    except Exception as e:
        app.logger.error(f"Error serving file {filename}: {str(e)}")
        return jsonify({'error': f'Error serving file: {str(e)}'}), 500

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'service': 'multilingual-translator'})

# Debug endpoint to list files in output directory
@app.route('/debug/files')
def debug_files():
    try:
        output_dir = os.path.abspath(OUTPUT_FOLDER)
        files = []
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                file_path = os.path.join(output_dir, filename)
                file_info = {
                    'name': filename,
                    'size': os.path.getsize(file_path),
                    'exists': os.path.exists(file_path)
                }
                files.append(file_info)
        return jsonify({
            'output_directory': output_dir,
            'files': files
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# Main program entry point
if __name__ == "__main__":
    # Check if GROQ_API_KEY is set
    if not os.getenv('GROQ_API_KEY'):
        print("Warning: GROQ_API_KEY not found in environment variables!")
        print("Please set your Groq API key in the config/.env file or as an environment variable.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)