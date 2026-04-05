from flask import Flask, request, jsonify
import os
import re
import tempfile
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def basic_cleanup(text: str) -> str:
    """
    ניקוי טכני בלבד.
    חשוב: לא משנה שפה, לא מתרגם, לא משכתב.
    """
    if not text:
        return ""

    result = text.strip()

    # רווחים כפולים
    result = re.sub(r"[ \t]+", " ", result)

    # רווחים לפני סימני פיסוק
    result = re.sub(r"\s+([,.;:!?])", r"\1", result)

    # רווחים אחרי סימני פיסוק אם חסר
    result = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", result)

    # ניקוי שורות ריקות כפולות
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


@app.route("/transcribe", methods=["POST"])
def transcribe():
    temp_path = None

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        uploaded_file = request.files["file"]

        if uploaded_file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        suffix = os.path.splitext(uploaded_file.filename)[1] or ".m4a"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            uploaded_file.save(temp_file.name)
            temp_path = temp_file.name

        with open(temp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file,
                prompt="""
This is a business meeting that may include Hebrew and English.

Rules:
- Keep Hebrew in Hebrew.
- Keep English in English.
- Do NOT translate between languages.
- Preserve the exact spoken words as much as possible.
- Keep mixed-language sentences intact.
"""
            )

        raw_text = (transcription.text or "").strip()
        cleaned_text = basic_cleanup(raw_text)

        return jsonify({
            "raw_text": raw_text,
            "text": cleaned_text
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/", methods=["GET"])
def home():
    return "Transcription server is running 🚀"


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
