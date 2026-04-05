from flask import Flask, request, jsonify
import os
import tempfile
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        # שמירה זמנית
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_audio:
            file.save(temp_audio.name)
            temp_path = temp_audio.name

        # 🔥 תמלול משופר MULTI-LANGUAGE
        with open(temp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file,

                # 🔥 חשוב מאוד — לא לנעול שפה
                # language="he", ❌ למחוק!

                # 🔥 הנחיה חכמה לשיחה מעורבת
                prompt="""
                This is a business meeting that may include Hebrew and English.
                - Keep Hebrew in Hebrew.
                - Keep English in English.
                - Do NOT translate between languages.
                - Preserve original spoken language exactly.
                """
            )

        os.remove(temp_path)

        return jsonify({
            "text": transcription.text
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/")
def home():
    return "Transcription server is running 🚀"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
