from flask import Flask, request, jsonify
import os
import tempfile
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def fix_bidirectional_text(text: str) -> str:
    """
    מוסיף סימני כיווניות כדי שטקסט משולב עברית/אנגלית
    יוצג בצורה יציבה יותר באפליקציות כמו Notes / iOS / macOS.
    """
    LTR = "\u200E"
    RTL = "\u200F"

    result = []
    current = ""
    current_mode = None  # "he" / "non-he"

    def char_mode(ch: str) -> str:
        if "\u0590" <= ch <= "\u05FF":
            return "he"
        return "non-he"

    for ch in text:
        # שומרים רווחים וסימני ירידת שורה כחלק מהקטע הנוכחי
        if ch == "\n":
            if current:
                if current_mode == "he":
                    result.append(RTL + current + RTL)
                else:
                    result.append(LTR + current + LTR)
                current = ""
                current_mode = None
            result.append("\n")
            continue

        mode = char_mode(ch)

        if current_mode is None:
            current_mode = mode
            current = ch
            continue

        if mode == current_mode:
            current += ch
        else:
            if current_mode == "he":
                result.append(RTL + current + RTL)
            else:
                result.append(LTR + current + LTR)

            current = ch
            current_mode = mode

    if current:
        if current_mode == "he":
            result.append(RTL + current + RTL)
        else:
            result.append(LTR + current + LTR)

    return "".join(result)


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
- Preserve original languages exactly.
- Keep Hebrew in Hebrew.
- Keep English in English.
- Do not translate.
- Improve punctuation and spacing.
- Remove obvious duplicated fragments if they are clearly accidental.
- Format the transcript into readable sentences and short paragraphs.
- Do NOT break lines based on language switches.
- Keep mixed-language sentences intact, for example: Hebrew with English words inside the same sentence.
- Only break lines between full sentences or clear topic changes.
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

        cleaned_text = (cleanup_response.output_text or "").strip()
        cleaned_text = fix_bidirectional_text(cleaned_text)

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
