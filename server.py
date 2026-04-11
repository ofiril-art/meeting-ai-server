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
        meeting_language = (data.get("meeting_language") or "אוטומטי").strip()
        print("🌐 Meeting language:", meeting_language, flush=True)

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
            user_content += "\n\nBackground context from related links (use only if relevant, never copy verbatim):\n" + links_context_text

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": """
You create improved meeting summaries using the transcript and any attached files metadata.

Rules:
- Language rules:
  - If meeting_language is "אנגלית" → respond in English.
  - If meeting_language is "עברית" → respond in Hebrew.
  - If meeting_language is "אוטומטי" → detect from transcript and respond accordingly.
- Keep English technical terms in English if they are part of the meeting domain.
- Do not invent facts from attachments you did not actually read.
- If only attachment file names/types are available, use them only as context hints.
- If link content was fetched successfully, use it only as background context.
- Never copy raw website text into the summary or action items; only extract concise, relevant insights if they genuinely help explain the meeting.
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
                    "content": f"Meeting language: {meeting_language}\n\n" + user_content
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

        email_style = (data.get("email_style") or "מקצועי").strip()
        print("✉️ Email style received:", email_style, flush=True)
        email_audience = (data.get("email_audience") or "צוות פנימי").strip()
        print("👥 Email audience received:", email_audience, flush=True)
        meeting_language = (data.get("meeting_language") or "אוטומטי").strip()
        print("🌐 Meeting language:", meeting_language, flush=True)

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

        hidden_context_section = ""
        if attachment_text:
            hidden_context_section += f"\nAttachments related to the meeting:\n{attachment_text}\n"
        if links_context_text:
            hidden_context_section += f"\nLink context extracted from related websites:\n{links_context_text}\n"

        style_instruction = ""
        if email_style == "קצר":
            style_instruction = "Keep the email extremely concise. Maximum 3-4 sentences total. No bullets. Prioritize only the single most important points."
        elif email_style == "ניהולי":
            style_instruction = "Write like a senior executive summary. Focus on decisions, risks, and next steps. Use bullet points. Remove all fluff."
        elif email_style == "ידידותי":
            style_instruction = "Write like a colleague sending a casual follow-up. Use a more natural, flowing tone and less formal structure."
        else:
            style_instruction = "Use a clear, structured, professional tone."

        audience_instruction = ""
        if email_audience == "הנהלה":
            audience_instruction = "Write for senior management. Emphasize business impact, decisions, risks, and next steps."
        elif email_audience == "לקוח":
            audience_instruction = "Write for an external client. Keep the tone polished, representative, and client-safe. Do not sound internal or operational. Avoid internal shorthand, ownership language, or task-tracking phrasing. Focus on outcomes, alignment, and agreed next steps."
        else:
            audience_instruction = "Write for an internal team. Keep it practical, clear, and execution-oriented."

        prompt = f"""
You generate a professional meeting summary email.

Rules:
- Language rules:
  - If meeting_language is "אנגלית" → write in English.
  - If meeting_language is "עברית" → write in Hebrew.
  - If meeting_language is "אוטומטי" → detect from transcript and write accordingly.
- Keep English terms if they appeared that way.
- Tone: adapt to the requested style.
- Style instruction: {style_instruction}
- Audience instruction: {audience_instruction}
- Do not invent details.
- First decide what are the 2-4 MOST important insights from the meeting.
- Prioritize decisions, risks, and next steps over general discussion.
- Do not include everything — include only what matters.
- Use attachments and link content only as background context.
- Do NOT paste raw website text, raw scraped content, long quotes, or lists of site sections into the email.
- Only include information from links if it is clearly relevant to the actual meeting discussion.
- If the links add no real value to the meeting itself, ignore them.
- The final email must read naturally, like a human wrote it after attending the meeting.
- Return ONLY valid JSON.

Hidden meeting context (meeting_language: {meeting_language}):
Meeting name: {meeting_name}
Meeting date: {meeting_date}
Summary of the meeting: {summary}
Transcript:
{transcript}
Action items:
{json.dumps(action_items, ensure_ascii=False)}
{hidden_context_section}

Return JSON in this exact format:
{{
  "subject": "...",
  "body": "..."
}}

Subject requirements:
- Hebrew
- Professional
- Based on the meeting name and date
- Adapt to audience:
  - For "לקוח": more formal and external-facing
  - For "הנהלה": concise and executive-style
  - For "צוות פנימי": more direct and practical

Body requirements:
- Start with: שלום רב,
- Then one short opening sentence about the meeting
- Then ONLY the key insights that truly matter
- Then a short section for follow-up tasks if there are action items
- End naturally and professionally
- Keep it concise and useful
- Each email style must feel clearly different in structure, tone, and length.
- Avoid generating similar outputs across styles.
- For "קצר": no bullets, maximum 3-4 short sentences total.
- For "ניהולי": use bullets and emphasize decisions, risks, and next steps.
- For "ידידותי": prefer a more natural flowing structure and warmer wording.
- For "מקצועי": keep a balanced formal structure.
- For "לקוח": avoid internal operational details and present the meeting as a clear external-facing summary.
- Do not mention "attachments", "links", "scraped content", or "website content" unless absolutely necessary for understanding the meeting
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
