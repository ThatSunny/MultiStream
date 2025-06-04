from flask import Flask, render_template, request, send_from_directory, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
from transformers import pipeline
from moviepy.editor import VideoFileClip
from gtts import gTTS
from pydub import AudioSegment
import subprocess
import openai

# Load environment variables
load_dotenv(dotenv_path='config/env')

# Set up Groq OpenAI-compatible API
openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_base = "https://api.groq.com/openai/v1"

# Flask app initialization
app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)
app.config['UPLOAD_FOLDER'] = 'data/uploads/'
OUTPUT_FOLDER = 'data/output/'
TEMP_FOLDER = 'data/temp/'
ALLOWED_EXTENSIONS = {'mp4', 'mp3'}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Load Whisper ASR model
speech_to_text = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    device=-1
)

# Replace: Google Gemini -> Groq for translation
def translate_text(text, target_language):
    prompt = f"Translate the following text to {target_language}:\n\n{text}"
    try:
        response = openai.ChatCompletion.create(
            model="llama3-70b-8192",  # Or "mixtral-8x7b-32768"
            messages=[
                {"role": "system", "content": "You are a professional translation assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error] Translation failed: {str(e)}"

def transcribe_audio(file_path):
    result = speech_to_text(file_path)
    return result['text']

def convert_to_audio(text, output_file="output.mp3"):
    tts = gTTS(text)
    tts.save(output_file)
    return output_file

def adjust_audio_speed(audio_file, target_duration, output_file="adjusted_audio.mp3"):
    output_file = os.path.join(TEMP_FOLDER, output_file)
    audio = AudioSegment.from_file(audio_file)
    current_duration = audio.duration_seconds
    speed_factor = current_duration / target_duration

    adjusted_audio = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * speed_factor)
    }).set_frame_rate(audio.frame_rate)

    adjusted_audio.export(output_file, format="mp3")
    return output_file

def convert_video_to_audio(video_file_path, output_audio_file="output_audio.mp3"):
    video = VideoFileClip(video_file_path)
    video.audio.write_audiofile(output_audio_file)
    return output_audio_file

def replace_audio_in_video(original_video_path, translated_audio_path, output_video_path="output_video.mp4"):
    output_video_path = os.path.join(OUTPUT_FOLDER, output_video_path)
    command = [
        "ffmpeg",
        "-y",
        "-i", original_video_path,
        "-i", translated_audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        output_video_path
    ]
    subprocess.run(command, check=True)
    return output_video_path

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        target_language = request.form.get('target_language')

        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Unsupported file type. Please upload an MP4 or MP3 file.'})

        file_content = file.read()
        if len(file_content) > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'error': 'File is too large. Maximum size is 50MB.'})

        file.seek(0)
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            if filename.endswith('.mp4'):
                audio_file = convert_video_to_audio(file_path, os.path.join(OUTPUT_FOLDER, "output_audio.mp3"))
                transcription = transcribe_audio(audio_file)
                
                translated_text = translate_text(transcription, target_language)
                if translated_text.startswith("[Error]"):
                    raise Exception(translated_text)

                translated_audio_file = os.path.join(OUTPUT_FOLDER, "translated_audio.mp3")
                convert_to_audio(translated_text, output_file=translated_audio_file)

                video = VideoFileClip(file_path)
                adjusted_audio_file = adjust_audio_speed(translated_audio_file, video.audio.duration)

                final_video_file = replace_audio_in_video(file_path, adjusted_audio_file, "final_video.mp4")

                video_url = f'/data/output/{os.path.basename(final_video_file)}'
                return jsonify({'success': True, 'video_url': video_url})
            else:
                return jsonify({'error': 'Unsupported file type. Only MP4 files are supported for now.'})
        except Exception as e:
            app.logger.error(f"Error during file processing: {str(e)}")
            return jsonify({'error': f'Error during processing: {str(e)}'})

    return jsonify({'message': 'Use POST request to upload files.'})

@app.route('/data/output/<filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react_app(path):
    if path != "" and os.path.exists(os.path.join("frontend", path)):
        return send_from_directory("frontend", path)
    else:
        return send_from_directory("frontend", "index.html")

if __name__ == "__main__":
    app.run(debug=True)

# For Vercel
handler = app