from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import os
import re
import json
import tempfile
import requests
from bs4 import BeautifulSoup
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
    if not text:
        return ""

    text = re.sub(r"([א-ת])([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])([א-ת])", r"\1 \2", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def add_readable_paragraphs(text: str) -> str:
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


# Helper to fetch text from a link, for link attachments
def fetch_link_text(url: str, max_chars: int = 4000) -> str:
    url = (url or "").strip()
    if not url:
        return ""

    if not url.lower().startswith(("http://", "https://")):
        url = f"https://{url}"

    try:
        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            }
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        text = soup.get_text("\n", strip=True)
        text = re.sub(r"\n{2,}", "\n", text)
        text = text.strip()

        if not text and not title:
            return ""

        combined = ""
        if title:
            combined += f"Title: {title}\n"
        if text:
            combined += text

        return combined[:max_chars].strip()
    except Exception as e:
        print(f"❌ Failed fetching link content from {url}: {e}", flush=True)
        return ""


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
        return jsonify({"error": str(e)}), 500

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
        link_context_blocks = []

        for item in attachments:
            if isinstance(item, dict):
                file_name = (item.get("fileName") or "").strip()
                file_type = (item.get("fileType") or "").strip()

                if file_name or file_type:
                    attachment_lines.append(f"- {file_name} ({file_type})")

                if file_type.lower() == "link" and file_name:
                    link_text = fetch_link_text(file_name)
                    if link_text:
                        link_context_blocks.append(
                            f"קישור: {file_name}\nתוכן מהקישור:\n{link_text}"
                        )

        attachment_text = "\n".join(attachment_lines).strip()
        links_context_text = "\n\n".join(link_context_blocks).strip()
        user_content = transcript

        if attachment_text:
            user_content += "\n\nAttachments:\n" + attachment_text

        if links_context_text:
            user_content += "\n\nLink content:\n" + links_context_text

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
- If link content was fetched successfully, use it as additional factual context for the summary and action items.
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


@app.route("/generate-email-summary", methods=["POST"])
def generate_email_summary():
    try:
        data = request.get_json(silent=True) or {}

        transcript = data.get("transcript", "")
        summary = data.get("summary", "")
        action_items = data.get("action_items", [])
        meeting_name = data.get("meeting_name", "")
        meeting_date = data.get("meeting_date", "")

        attachments = data.get("attachments", [])
        print("📎 Email attachments received:", attachments, flush=True)
        print("📎 Email attachments count:", len(attachments), flush=True)

        attachment_lines = []
        link_context_blocks = []

        for item in attachments:
            if isinstance(item, dict):
                file_name = (item.get("file_name") or "").strip()
                file_type = (item.get("file_type") or "").strip()

                if file_name or file_type:
                    attachment_lines.append(f"- {file_name} ({file_type})")

                if file_type.lower() == "link" and file_name:
                    link_text = fetch_link_text(file_name)
                    if link_text:
                        link_context_blocks.append(
                            f"קישור: {file_name}\nתוכן מהקישור:\n{link_text}"
                        )

        attachment_text = "\n".join(attachment_lines).strip()
        links_context_text = "\n\n".join(link_context_blocks).strip()

        attachments_section = ""
        if attachment_text:
            attachments_section = f"\nקבצים וקישורים שקשורים לפגישה:\n{attachment_text}\n"

        links_content_section = ""
        if links_context_text:
            links_content_section = f"\nתוכן שנשלף מהקישורים:\n{links_context_text}\n"

        prompt = f"""
You generate a professional meeting summary email in Hebrew.

Rules:
- Write in Hebrew.
- Keep English terms if they appeared that way.
- Tone: professional, clear, human-like.
- Do not invent details.
- Use the exact structure requested.
- Return ONLY valid JSON.

Return JSON in this format:
{{
  "subject": "...",
  "body": "..."
}}

Subject format:
סיכום פגישה – {meeting_name} – {meeting_date}

Body format:
שלום רב,

בתאריך {meeting_date} התקיימה פגישה בנושא {meeting_name}.

מטרת הפגישה הייתה:
{summary}
{attachments_section}
{links_content_section}
עיקרי הדברים שנדונו:
(Use 3-6 concise bullet points based on transcript and summary)

משימות להמשך:
(Use the provided action_items)

תודה לכולם.
"""

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": "You generate structured professional meeting summary emails in Hebrew."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        parsed = json_from_text(response.output_text)

        return jsonify({
            "subject": parsed.get("subject", ""),
            "body": parsed.get("body", "")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "Transcription server is running 🚀"


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    
    
from datetime import datetime
from flask import request, jsonify

@app.route("/extract-date", methods=["POST"])
def extract_date():
    data = request.get_json()
    text = (data.get("text", "") or "").strip()

    today = datetime.now().date()

    def next_weekday_date(target_weekday: int):
        days_ahead = (target_weekday - today.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        return today + timedelta(days=days_ahead)

    try:
        weekday_map = {
            "יום ראשון": 6,
            "יום שני": 0,
            "יום שלישי": 1,
            "יום רביעי": 2,
            "יום חמישי": 3,
            "יום שישי": 4,
            "יום שבת": 5,
        }

        for phrase, weekday in weekday_map.items():
            if phrase in text:
                found_date = next_weekday_date(weekday)
                return jsonify({"date": found_date.strftime("%Y-%m-%d")})

        if "מחר" in text:
            return jsonify({"date": (today + timedelta(days=1)).strftime("%Y-%m-%d")})

        if "שבוע הבא" in text:
            return jsonify({"date": (today + timedelta(days=7)).strftime("%Y-%m-%d")})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""
Extract a due date from a task.

Today is {today.strftime("%Y-%m-%d")}

Return ONLY:
YYYY-MM-DD or null
"""
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
        )

        result = response.choices[0].message.content.strip()

        if result.lower() == "null":
            return jsonify({"date": None})

        return jsonify({"date": result})

    except Exception as e:
        print("❌ extract_date error:", e)
        return jsonify({"date": None})
