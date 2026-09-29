"""
Chinese Talking Avatar Video Generator — Browser (Flask) Version
====================================================================
Same pipeline as the desktop version, but the UI runs in your web browser
instead of a Tkinter window.

PIPELINE:
  1. Chinese text (typed in the browser)
  2. -> edge-tts converts text to spoken Chinese audio
  3. -> Wav2Lip syncs a face image/video's lips to that audio
  4. -> Final .mp4 is saved and playable directly in the browser

REQUIRED SETUP (same as before, plus Flask):
------------------------------------------------------------
1. Install Python packages:
     pip install flask edge-tts pillow

2. Clone and set up Wav2Lip:
     git clone https://github.com/Rudrabha/Wav2Lip.git
     cd Wav2Lip
     pip install -r requirements.txt

3. Download the Wav2Lip checkpoint "wav2lip_gan.pth" (link in Wav2Lip's
   README) and place it at Wav2Lip/checkpoints/wav2lip_gan.pth

4. Provide a face source, saved as "avatar.mp4" or "avatar.jpg" next to
   this script (a short front-facing video works best).

5. Update WAV2LIP_DIR below if you cloned Wav2Lip somewhere else.

RUN:
    python chinese_avatar_generator_web.py

Then open http://127.0.0.1:5000 in your browser.
"""

import asyncio
import os
import subprocess
from datetime import datetime

import edge_tts
from flask import Flask, render_template_string, request, send_from_directory

# -----------------------------
# CONFIG — adjust for your machine
# -----------------------------
WAV2LIP_DIR = "./Wav2Lip"
CHECKPOINT_PATH = os.path.join(WAV2LIP_DIR, "checkpoints", "wav2lip_gan.pth")
# AVATAR_FACE_PATH = "./avatar.jpg"
AVATAR_FACE_PATH = "./avatar.jpg"  # <- you can use a short video or a single image
OUTPUT_DIR = os.path.join(WAV2LIP_DIR, "output_videos")

VOICE_OPTIONS = {
    "Female (Mainland)": "zh-CN-XiaoxiaoNeural",
    "Male (Mainland)": "zh-CN-YunxiNeural",
    "Female (Taiwan)": "zh-TW-HsiaoChenNeural",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
app = Flask(__name__)


# -----------------------------
# STEP 1: Text -> Speech
# -----------------------------
async def generate_speech(text: str, voice: str, output_audio_path: str):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_audio_path)


# -----------------------------
# STEP 2: Speech + Face -> Lip-synced video
# -----------------------------
def run_wav2lip(audio_path: str, output_video_path: str):
    # Convert everything to ABSOLUTE paths first, since we're about to change
    # the subprocess's working directory to WAV2LIP_DIR — any relative paths
    # passed in would otherwise be looked up relative to WAV2LIP_DIR instead
    # of wherever they actually are.
    abs_checkpoint = os.path.abspath(CHECKPOINT_PATH)
    abs_face = os.path.abspath(AVATAR_FACE_PATH)
    abs_audio = os.path.abspath(audio_path)
    abs_output = os.path.abspath(output_video_path)

    # Wav2Lip's own temp/ folder needs to exist inside WAV2LIP_DIR
    os.makedirs(os.path.join(WAV2LIP_DIR, "temp"), exist_ok=True)

    command = [
        "python3", "inference.py",  # <- now relative to cwd=WAV2LIP_DIR, not the old joined path
        "--checkpoint_path", abs_checkpoint,
        "--face", abs_face,
        "--audio", abs_audio,
        "--outfile", abs_output,
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=os.path.abspath(WAV2LIP_DIR),  # <- run FROM inside Wav2Lip's folder
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)


# -----------------------------
# HTML page (kept in this file for simplicity — no separate templates folder needed)
# -----------------------------
PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <title>Chinese Talking Avatar Generator</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; }
    textarea { width: 100%; font-size: 16px; padding: 10px; box-sizing: border-box; }
    select, button { font-size: 15px; padding: 8px; margin-top: 10px; }
    button { background: #2563eb; color: white; border: none; border-radius: 6px; cursor: pointer; }
    button:hover { background: #1d4ed8; }
    .status { margin-top: 15px; color: #444; }
    .error { color: #b91c1c; }
    video { margin-top: 20px; width: 100%; border-radius: 8px; }
  </style>
</head>
<body>
  <h2>Chinese Talking Avatar Generator</h2>
  <form method="POST">
    <label>Enter Chinese text:</label><br>
    <textarea name="chinese_text" rows="5" placeholder="在这里输入中文文本...">{{ text or "" }}</textarea><br>

    <label>Voice:</label><br>
    <select name="voice">
      {% for label, value in voices.items() %}
        <option value="{{ value }}" {% if value == selected_voice %}selected{% endif %}>{{ label }}</option>
      {% endfor %}
    </select><br>

    <button type="submit">Generate Video</button>
  </form>

  {% if status %}
    <p class="status {{ 'error' if error else '' }}">{{ status }}</p>
  {% endif %}

  {% if video_filename %}
    <video controls>
      <source src="{{ url_for('serve_video', filename=video_filename) }}" type="video/mp4">
    </video>
  {% endif %}
</body>
</html>
"""


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    status = None
    error = False
    video_filename = None
    text = ""
    selected_voice = list(VOICE_OPTIONS.values())[0]

    if request.method == "POST":
        text = request.form.get("chinese_text", "").strip()
        selected_voice = request.form.get("voice", selected_voice)

        if not text:
            status, error = "Please enter some Chinese text.", True
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_path = os.path.join(OUTPUT_DIR, f"speech_{timestamp}.mp3")
            video_path = os.path.join(OUTPUT_DIR, f"avatar_video_{timestamp}.mp4")

            try:
                asyncio.run(generate_speech(text, selected_voice, audio_path))
                run_wav2lip(audio_path, video_path)
                video_filename = os.path.basename(video_path)
                status = "Video generated successfully!"
            except Exception as e:
                status, error = f"Error: {e}", True

    return render_template_string(
        PAGE_TEMPLATE,
        status=status,
        error=error,
        video_filename=video_filename,
        text=text,
        voices=VOICE_OPTIONS,
        selected_voice=selected_voice,
    )


@app.route("/videos/<filename>")
def serve_video(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)