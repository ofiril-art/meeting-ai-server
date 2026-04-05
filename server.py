from flask import Flask, request, jsonify
from openai import OpenAI
import os
import tempfile

app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/")
def home():
    return "Server is running!"

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio_file = request.files["file"]

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        audio_file.save(tmp.name)

        with open(tmp.name, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=f
            )

    return jsonify({
        "text": transcription.text
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
