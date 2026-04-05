from flask import Flask, request, jsonify
import os
import tempfile
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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
Keep Hebrew in Hebrew.
Keep English in English.
Do not translate between languages.
Preserve the original spoken language exactly.
"""
            )

        raw_text = (transcription.text or "").strip()

        cleanup_response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": """
You clean business meeting transcripts.

Rules:
- Preserve the original spoken languages exactly.
- Keep Hebrew in Hebrew.
- Keep English in English.
- Do not translate.
- Improve punctuation and spacing.
- Remove obvious duplicated fragments if they are clearly accidental.
- Format the transcript into readable sentences and short paragraphs.
- Do not add content that was not spoken.
- Do not summarize.
- Return only the cleaned transcript text.
"""
                },
                {
                    "role": "user",
                    "content": raw_text
                }
            ]
        )

        cleaned_text = cleanup_response.output_text.strip()

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
