from moviepy import VideoFileClip

# Load the video file (using raw string or double backslashes)
video = VideoFileClip(r"#")

# Extract audio and write it to an MP3 file (using raw string or double backslashes)
video.audio.write_audiofile(r"#")