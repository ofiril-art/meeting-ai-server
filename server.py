from flask import Flask, request, jsonify
import os
import re
import json
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
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def fix_mixed_text(text: str) -> str:
    """
    ניקוי עדין לטקסט משולב עברית/אנגלית:
    - מוסיף רווחים כשעברית ואנגלית נצמדות
    - לא מתרגם
    - לא משנה סדר מילים
    """
    if not text:
        return ""

    text = re.sub(r"([א-ת])([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])([א-ת])", r"\1 \2", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def add_readable_paragraphs(text: str) -> str:
    """
    מחלק לפסקאות קריאות יותר בלי לשנות תוכן.
    """
    if not text:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return text.strip()

    paragraphs = []
    current = []

    for sentence in sentences:
        current.append(sentence)

        if len(current) >= 2:
            paragraphs.append(" ".join(current))
            current = []

    if current:
        paragraphs.append(" ".join(current))

    return "\n\n".join(paragraphs).strip()


def json_from_text(text: str):
    text = (text or "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError("No valid JSON found")


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
- Keep English EXACTLY as spoken (do not translate English words).
- If a word was spoken in English, keep it in English.
- Do NOT replace English terms with Hebrew equivalents.
- Preserve natural sentence flow.
- Keep mixed-language sentences intact.
"""
            )

        raw_text = (transcription.text or "").strip()
        cleaned_text = basic_cleanup(raw_text)
        cleaned_text = fix_mixed_text(cleaned_text)
        formatted_text = add_readable_paragraphs(cleaned_text)

        summary_response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": """
You create concise meeting summaries from transcripts.

Rules:
- Respond in Hebrew.
- Keep English technical terms in English when they appeared that way in the transcript.
- Do not invent facts.
- If something is unclear, omit it.
- Return valid JSON only with this exact structure:
{
  "summary": "short paragraph",
  "action_items": ["item 1", "item 2"]
}
- summary must be 2-4 sentences.
- action_items should contain only concrete next steps, and can be empty.
"""
                },
                {
                    "role": "user",
                    "content": formatted_text
                }
            ]
        )

        summary_text = ""
        action_items = []

        try:
            parsed = json_from_text(summary_response.output_text)
            summary_text = parsed.get("summary", "") if isinstance(parsed, dict) else ""
            action_items = parsed.get("action_items", []) if isinstance(parsed, dict) else []

            if not isinstance(action_items, list):
                action_items = []
        except Exception:
            summary_text = ""
            action_items = []

        return jsonify({
            "raw_text": raw_text,
            "text": formatted_text,
            "summary": summary_text,
            "action_items": action_items
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.route("/regenerate-summary", methods=["POST"])
def regenerate_summary():
    try:
        data = request.get_json(silent=True) or {}

        transcript = (data.get("transcript") or "").strip()
        attachments = data.get("attachments") or []

        if not transcript:
            return jsonify({"error": "Missing transcript"}), 400

        attachment_lines = []
        for item in attachments:
            if isinstance(item, dict):
                file_name = (item.get("fileName") or "").strip()
                file_type = (item.get("fileType") or "").strip()

                if file_name or file_type:
                    attachment_lines.append(f"- {file_name} ({file_type})")

        attachment_text = "\n".join(attachment_lines).strip()
        user_content = transcript

        if attachment_text:
            user_content += "\n\nAttachments:\n" + attachment_text

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": """
You create improved meeting summaries using the transcript and any attached files metadata.

Rules:
- Respond in Hebrew.
- Keep English technical terms in English if they are part of the meeting domain.
- Do not invent facts from attachments you did not actually read.
- If only attachment file names/types are available, use them only as context hints.
- Return valid JSON only with this exact structure:
{
  "summary": "short paragraph",
  "action_items": ["item 1", "item 2"]
}
- summary must be 2-4 sentences.
- action_items should contain only concrete next steps, and can be empty.
"""
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        )

        parsed = json_from_text(response.output_text)
        summary_text = parsed.get("summary", "") if isinstance(parsed, dict) else ""
        action_items = parsed.get("action_items", []) if isinstance(parsed, dict) else []

        if not isinstance(action_items, list):
            action_items = []

        return jsonify({
            "summary": summary_text,
            "action_items": action_items
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "Transcription server is running 🚀"


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
