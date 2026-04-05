from flask import Flask, request, jsonify
import os
import re
import tempfile
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def basic_cleanup(text: str) -> str:
    if not text:
        return ""

    result = text.strip()
    result = re.sub(r"[ \t]+", " ", result)
    result = re.sub(r"\s+([,.;:!?])", r"\1", result)
    result = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", result)
    return result.strip()


@app.route("/transcribe", methods=["POST"])
def transcribe():
    temp_path = None

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        uploaded_file = request.files["file"]

        suffix = os.path.splitext(uploaded_file.filename)[1] or ".m4a"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            uploaded_file.save(temp_file.name)
            temp_path = temp_file.name

        # ===== תמלול =====
        with open(temp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file,
                prompt="""
Keep Hebrew in Hebrew.
Keep English in English.
Do not translate.
"""
            )

        raw_text = (transcription.text or "").strip()

        # ===== תיקון סדר משפט בלבד =====
        fix_response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": """
You receive a transcript that may mix Hebrew and English.

Your job:
- Fix ONLY word order if it is broken
- Keep ALL words exactly the same
- Do NOT translate anything
- Do NOT remove words
- Do NOT add words
- Do NOT summarize
- Keep mixed-language sentences natural

Return the corrected sentence only.
"""
                },
                {
                    "role": "user",
                    "content": raw_text
                }
            ]
        )

        fixed_text = (fix_response.output_text or "").strip()
        cleaned_text = basic_cleanup(fixed_text)

        return jsonify({
            "raw_text": raw_text,
            "text": cleaned_text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/", methods=["GET"])
def home():
    return "Transcription server is running 🚀"


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
