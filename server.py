from flask import Flask, request, jsonify
from openai import OpenAI
import os
import tempfile

app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/")
def home():
    return jsonify({"status": "ok"})

@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

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

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
