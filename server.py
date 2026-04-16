from datetime import datetime, timedelta, timezone
from flask import Flask, request, jsonify
from urllib.parse import urlparse
import os
import re
import json
import tempfile
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

app = Flask(__name__)

TEAMS_HOST_PATTERNS = {
    "teams.microsoft.com",
    "www.teams.microsoft.com",
}
TEAMS_SESSIONS = {}
TEAMS_BOT_MODE = "mock"
TEAMS_JOB_STATES_ACTIVE = {"created", "start_requested", "booting_bot", "joining_meeting", "recording"}
TEAMS_BOT_EVENT_LOGS = []
TEAMS_STATE_FILE = os.getenv("TEAMS_STATE_FILE", "teams_state.json")

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


# Normalize meeting analysis structure for summary/dynamics/speakers/action items
def normalize_meeting_analysis(parsed):
    empty_dynamics = {
        "communication_style": "",
        "decision_pattern": "",
        "alignment_level": "",
        "key_tensions": []
    }

    if not isinstance(parsed, dict):
        return {
            "summary": "",
            "action_items": [],
            "speakers": [],
            "action_items_by_speaker": [],
            "meeting_dynamics": empty_dynamics
        }

    summary = (parsed.get("summary") or "").strip() if isinstance(parsed.get("summary"), str) else ""

    action_items = [
        i.strip() for i in (parsed.get("action_items") or [])
        if isinstance(i, str) and i.strip()
    ]

    speakers = []
    for item in (parsed.get("speakers") or []):
        if not isinstance(item, dict):
            continue
        name = (item.get("speaker") or item.get("name") or "").strip()
        role = (item.get("role_hint") or item.get("role") or "").strip()
        highlights = item.get("highlights")
        if isinstance(highlights, list):
            highlights = [h.strip() for h in highlights if isinstance(h, str) and h.strip()]
        else:
            contrib = (item.get("contribution") or "").strip()
            highlights = [contrib] if contrib else []
        if name or role or highlights:
            speakers.append({"speaker": name, "role_hint": role, "highlights": highlights})

    action_items_by_speaker = []
    for item in (parsed.get("action_items_by_speaker") or []):
        if not isinstance(item, dict):
            continue
        name = (item.get("speaker") or item.get("name") or "").strip()
        items = [
            a.strip() for a in (item.get("items") or [])
            if isinstance(a, str) and a.strip()
        ]
        if name or items:
            action_items_by_speaker.append({"speaker": name, "items": items})

    md = parsed.get("meeting_dynamics") or {}
    if not isinstance(md, dict):
        md = {}

    meeting_dynamics = {
        "communication_style": (md.get("communication_style") or "").strip() if isinstance(md.get("communication_style"), str) else "",
        "decision_pattern": (md.get("decision_pattern") or "").strip() if isinstance(md.get("decision_pattern"), str) else "",
        "alignment_level": (md.get("alignment_level") or "").strip() if isinstance(md.get("alignment_level"), str) else "",
        "key_tensions": [k.strip() for k in (md.get("key_tensions") or []) if isinstance(k, str) and k.strip()]
    }

    return {
        "summary": summary,
        "action_items": action_items,
        "speakers": speakers,
        "action_items_by_speaker": action_items_by_speaker,
        "meeting_dynamics": meeting_dynamics
    }


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


# Teams URL helpers
def is_valid_teams_url(url: str) -> bool:
    url = (url or "").strip()
    if not url:
        return False

    try:
        parsed = urlparse(url)
    except Exception:
        return False

    host = (parsed.netloc or "").lower().strip()
    scheme = (parsed.scheme or "").lower().strip()

    if scheme not in {"http", "https"}:
        return False

    if host in TEAMS_HOST_PATTERNS:
        return True

    return host.endswith(".teams.microsoft.com")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def load_teams_state():
    global TEAMS_SESSIONS, TEAMS_BOT_EVENT_LOGS

    if not os.path.exists(TEAMS_STATE_FILE):
        return

    try:
        with open(TEAMS_STATE_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)

        sessions = payload.get("sessions") or {}
        events = payload.get("events") or []

        if isinstance(sessions, dict):
            TEAMS_SESSIONS = sessions
        if isinstance(events, list):
            TEAMS_BOT_EVENT_LOGS = events

        print(f"📦 Loaded Teams state: {len(TEAMS_SESSIONS)} sessions, {len(TEAMS_BOT_EVENT_LOGS)} events", flush=True)
    except Exception as e:
        print(f"❌ Failed loading Teams state: {e}", flush=True)


def save_teams_state():
    try:
        payload = {
            "sessions": TEAMS_SESSIONS,
            "events": TEAMS_BOT_EVENT_LOGS[-200:],
            "saved_at": utc_now_iso(),
        }

        with open(TEAMS_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"❌ Failed saving Teams state: {e}", flush=True)


def apply_teams_session_updates(session: dict, **updates):
    for key, value in updates.items():
        session[key] = value
    session["updated_at"] = utc_now_iso()
    save_teams_state()
    return session


def append_teams_bot_event(event_type: str, payload=None, session_id: str | None = None, note: str | None = None):
    entry = {
        "event_id": f"evt_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
        "event_type": (event_type or "unknown").strip(),
        "session_id": (session_id or "").strip() or None,
        "received_at": utc_now_iso(),
        "note": (note or "").strip() or None,
        "payload": payload if isinstance(payload, (dict, list)) else payload,
    }
    TEAMS_BOT_EVENT_LOGS.append(entry)
    if len(TEAMS_BOT_EVENT_LOGS) > 200:
        del TEAMS_BOT_EVENT_LOGS[:-200]
    save_teams_state()
    return entry



def find_teams_session_by_join_url(join_url: str):
    normalized = (join_url or "").strip()
    if not normalized:
        return None

    for session in TEAMS_SESSIONS.values():
        candidates = [
            (session.get("join_url") or "").strip(),
            (session.get("teams_url") or "").strip(),
        ]
        if normalized in candidates:
            return session

    return None



def attach_event_to_matching_session(payload, preferred_session_id: str | None = None):
    if preferred_session_id:
        session = TEAMS_SESSIONS.get(preferred_session_id)
        if session:
            return session

    if isinstance(payload, dict):
        direct_session_id = (payload.get("session_id") or payload.get("sessionId") or "").strip()
        if direct_session_id:
            session = TEAMS_SESSIONS.get(direct_session_id)
            if session:
                return session

        possible_join_urls = [
            payload.get("join_url"),
            payload.get("joinUrl"),
            payload.get("teams_url"),
            payload.get("teamsUrl"),
            payload.get("meeting_url"),
            payload.get("meetingUrl"),
        ]

        for value in possible_join_urls:
            session = find_teams_session_by_join_url((value or "").strip())
            if session:
                return session

    return None



def update_session_from_calling_event(session: dict, payload):
    if not isinstance(payload, dict):
        return session

    event_name = (
        payload.get("event")
        or payload.get("event_type")
        or payload.get("eventType")
        or payload.get("type")
        or ""
    ).strip().lower()

    if event_name in {"incoming_call", "incomingcall", "ringing"}:
        return apply_teams_session_updates(
            session,
            status="bot_preparing",
            job_state="incoming_call",
            last_error=None,
        )

    if event_name in {"joining", "join_requested", "joinrequested", "connecting"}:
        return apply_teams_session_updates(
            session,
            status="bot_joining",
            job_state="joining",
            started_at=session.get("started_at") or utc_now_iso(),
            last_error=None,
        )

    if event_name in {"connected", "call_connected", "callconnected", "recording_started", "recordingstarted"}:
        return apply_teams_session_updates(
            session,
            status="recording",
            job_state="connected",
            started_at=session.get("started_at") or utc_now_iso(),
            last_error=None,
        )

    if event_name in {"recording", "recording_in_progress", "recordinginprogress"}:
        return apply_teams_session_updates(
            session,
            status="recording",
            job_state="recording",
            started_at=session.get("started_at") or utc_now_iso(),
            last_error=None,
        )

    if event_name in {"stopped", "call_ended", "callended", "hangup", "ended"}:
        return stop_teams_bot_for_session(session)

    if event_name in {"failed", "error", "call_failed", "callfailed"}:
        return apply_teams_session_updates(
            session,
            status="failed",
            job_state="failed",
            last_error=(payload.get("error") or payload.get("message") or "Calling event failed").strip(),
        )

    return session


def refresh_teams_session(session: dict):
    if TEAMS_BOT_MODE == "mock":
        return update_mock_teams_session_status(session)
    return session


def start_teams_bot_for_session(session: dict):
    current_status = (session.get("status") or "").strip().lower()
    current_job_state = (session.get("job_state") or "").strip().lower()

    if current_status == "stopped":
        raise ValueError("Session already stopped")

    if current_job_state in {"start_requested", "booting_bot", "joining_meeting", "recording", "joining", "connected"}:
        return session

    append_teams_bot_event(
        event_type="start_bot_requested",
        payload={
            "session_id": session.get("session_id"),
            "meeting_name": session.get("meeting_name"),
            "join_url": session.get("join_url"),
        },
        session_id=session.get("session_id"),
        note="Start bot requested from app",
    )

    if TEAMS_BOT_MODE == "mock":
        return apply_teams_session_updates(
            session,
            job_state="start_requested",
            last_error=None,
        )

    return apply_teams_session_updates(
        session,
        job_state="start_requested",
        last_error=None,
    )


def stop_teams_bot_for_session(session: dict):
    now = utc_now_iso()
    return apply_teams_session_updates(
        session,
        status="stopped",
        job_state="stopped",
        stopped_at=now,
        last_error=None,
    )



def create_teams_session_record(meeting_name: str, teams_url: str):
    now = datetime.utcnow()
    session_id = f"teams_{now.strftime('%Y%m%d%H%M%S%f')}"
    created_at = utc_now_iso()

    session = {
        "session_id": session_id,
        "provider": "teams",
        "join_url": teams_url,
        "teams_url": teams_url,
        "meeting_name": meeting_name,
        "status": "queued",
        "mode": "prepare_only",
        "bot_mode": TEAMS_BOT_MODE,
        "job_state": "created",
        "last_error": None,
        "created_at": created_at,
        "updated_at": created_at,
        "received_at": created_at,
        "started_at": None,
        "stopped_at": None,
        "bot_events": [],
    }

    TEAMS_SESSIONS[session_id] = session
    save_teams_state()
    return session



def build_teams_prepare_response(session):
    return {
        "ok": True,
        "session_id": session["session_id"],
        "status": session["status"],
        "provider": session["provider"],
        "mode": session["mode"],
        "bot_mode": session["bot_mode"],
        "job_state": session["job_state"],
        "message": "Teams meeting received. Bot integration is not connected yet.",
        "meeting_name": session["meeting_name"],
        "teams_url": session["teams_url"],
        "join_url": session["join_url"],
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
        "received_at": session["received_at"],
        "last_error": session["last_error"],
    }


def update_mock_teams_session_status(session):
    current_status = (session.get("status") or "").strip().lower()
    current_job_state = (session.get("job_state") or "").strip().lower()

    if current_status == "stopped":
        return session

    received_at = session.get("received_at") or ""

    try:
        received_dt = datetime.fromisoformat(received_at.replace("Z", "+00:00"))
    except Exception:
        return session

    now = datetime.now(timezone.utc)
    elapsed_seconds = max(0, int((now - received_dt).total_seconds()))

    if current_job_state == "start_requested" and elapsed_seconds < 10:
        return apply_teams_session_updates(
            session,
            status="queued",
            job_state="start_requested",
            last_error=None,
        )
    elif elapsed_seconds < 10:
        return apply_teams_session_updates(
            session,
            status="queued",
            job_state="created",
            last_error=None,
        )
    elif elapsed_seconds < 20:
        return apply_teams_session_updates(
            session,
            status="bot_preparing",
            job_state="booting_bot",
            last_error=None,
        )
    elif elapsed_seconds < 30:
        return apply_teams_session_updates(
            session,
            status="bot_joining",
            job_state="joining_meeting",
            started_at=session.get("started_at") or utc_now_iso(),
            last_error=None,
        )
    else:
        return apply_teams_session_updates(
            session,
            status="recording",
            job_state="recording",
            started_at=session.get("started_at") or utc_now_iso(),
            last_error=None,
        )

load_teams_state()
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

        # 🔒 Guard: avoid AI when transcript too short / weak
        if len(formatted_text.strip()) < 50:
            print("❌ Transcript too short on server — skipping AI", flush=True)
            return jsonify({
                "raw_text": raw_text,
                "text": formatted_text,
                "summary": "",
                "action_items": [],
                "speakers": [],
                "action_items_by_speaker": [],
                "meeting_dynamics": {
                    "communication_style": "",
                    "decision_pattern": "",
                    "alignment_level": "",
                    "key_tensions": []
                }
            })

        summary_response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": """
You analyze a business meeting transcript and return a high-quality structured result.

Rules:
- Respond in Hebrew.
- Keep English technical/product/business terms in English if they appeared that way in the transcript.
- Never invent facts, speakers, decisions, deadlines, or action items.
- If something is unclear, omit it.
- Ignore filler, repetition, and small talk.
- Prioritize what matters: decisions, commitments, blockers, risks, open questions, and next steps.
- If the transcript is weak, partial, or ambiguous, keep the output conservative.
- Return valid JSON ONLY with this exact structure:
{
  "summary": "2-4 sentence management-quality summary",
  "action_items": ["clear action item 1", "clear action item 2"],
  "speakers": [
    {
      "speaker": "Speaker 1",
      "role_hint": "role if known, otherwise empty string",
      "highlights": ["main contribution 1", "main contribution 2"]
    }
  ],
  "action_items_by_speaker": [
    {
      "speaker": "Speaker 1",
      "items": ["assigned action 1", "assigned action 2"]
    }
  ],
  "meeting_dynamics": {
    "communication_style": "short description",
    "decision_pattern": "short description",
    "alignment_level": "high / medium / low with short explanation",
    "key_tensions": ["specific tension 1", "specific tension 2"]
  }
}

Additional instructions:
- summary: should read like a concise update someone would send to a manager after the meeting.
- action_items: include only concrete next steps that someone can actually do.
- speakers: include only meaningful speakers if they can be inferred from the transcript.
- action_items_by_speaker: only assign an item to a speaker if the ownership is reasonably clear.
- meeting_dynamics: describe only what can actually be inferred from the transcript.
- If there are no reliable action items, return an empty list.
- If speaker identity is unclear, still use generic labels like "Speaker 1" but do not invent titles.
"""
                },
                {
                    "role": "user",
                    "content": formatted_text
                }
            ]
        )

        try:
            parsed = json_from_text(summary_response.output_text)
            normalized = normalize_meeting_analysis(parsed)
            summary_text = normalized["summary"]
            action_items = normalized["action_items"]
            speakers = normalized["speakers"]
            action_items_by_speaker = normalized["action_items_by_speaker"]
            meeting_dynamics = normalized["meeting_dynamics"]
        except Exception:
            summary_text = ""
            action_items = []
            speakers = []
            action_items_by_speaker = []
            meeting_dynamics = {
                "communication_style": "",
                "decision_pattern": "",
                "alignment_level": "",
                "key_tensions": []
            }

        return jsonify({
            "raw_text": raw_text,
            "text": formatted_text,
            "summary": summary_text,
            "action_items": action_items,
            "speakers": speakers,
            "action_items_by_speaker": action_items_by_speaker,
            "meeting_dynamics": meeting_dynamics
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

        # 🔒 Guard: avoid hallucinations on weak transcript
        if len(transcript.strip()) < 50:
            print("❌ regenerate_summary: transcript too short (server)", flush=True)
            return jsonify({
                "summary": "",
                "action_items": [],
                "speakers": [],
                "action_items_by_speaker": [],
                "meeting_dynamics": {
                    "communication_style": "",
                    "decision_pattern": "",
                    "alignment_level": "",
                    "key_tensions": []
                }
            })

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
You generate a smart, management-quality meeting analysis from a transcript and optional attachment context.

Rules:
- Language rules:
  - If meeting_language is "אנגלית" -> respond in English.
  - If meeting_language is "עברית" -> respond in Hebrew.
  - If meeting_language is "אוטומטי" -> detect from the transcript and respond accordingly.
- Keep English technical/product/business terms in English when appropriate.
- Never invent facts, names, decisions, ownership, dates, or risks.
- If something is unclear, omit it.
- Ignore filler, repetition, and side chatter.
- Focus on what actually matters: decisions, commitments, blockers, open questions, dependencies, risks, and next steps.
- Use attachments and fetched link context only as supporting context when clearly relevant.
- Never copy raw website text, attachment text, or long quotes into the output.
- If the transcript is partial or weak, keep the output conservative and minimal.
- Return valid JSON ONLY with this exact structure:
{
  "summary": "2-4 sentence management-quality summary",
  "action_items": ["clear action item 1", "clear action item 2"],
  "speakers": [
    {
      "speaker": "Speaker 1",
      "role_hint": "role if known, otherwise empty string",
      "highlights": ["main contribution 1", "main contribution 2"]
    }
  ],
  "action_items_by_speaker": [
    {
      "speaker": "Speaker 1",
      "items": ["assigned action 1", "assigned action 2"]
    }
  ],
  "meeting_dynamics": {
    "communication_style": "short description",
    "decision_pattern": "short description",
    "alignment_level": "high / medium / low with short explanation",
    "key_tensions": ["specific tension 1", "specific tension 2"]
  }
}

Additional instructions:
- summary: should be concise, executive, and useful.
- action_items: include only concrete, actionable next steps.
- speakers: capture only real, meaningful participation patterns.
- action_items_by_speaker: assign ownership only when reasonably supported by the transcript.
- meeting_dynamics: surface the real dynamics of the meeting, but do not over-interpret.
- If there are no reliable action items, return an empty list.
- If the output language is English, write natural professional English.
- If the output language is Hebrew, write natural professional Hebrew.
"""
                },
                {
                    "role": "user",
                    "content": f"Meeting language: {meeting_language}\n\n" + user_content
                }
            ]
        )

        try:
            parsed = json_from_text(response.output_text)
            normalized = normalize_meeting_analysis(parsed)
            summary_text = normalized["summary"]
            action_items = normalized["action_items"]
            speakers = normalized["speakers"]
            action_items_by_speaker = normalized["action_items_by_speaker"]
            meeting_dynamics = normalized["meeting_dynamics"]
        except Exception:
            summary_text = ""
            action_items = []
            speakers = []
            action_items_by_speaker = []
            meeting_dynamics = {
                "communication_style": "",
                "decision_pattern": "",
                "alignment_level": "",
                "key_tensions": []
            }

        return jsonify({
            "summary": summary_text,
            "action_items": action_items,
            "speakers": speakers,
            "action_items_by_speaker": action_items_by_speaker,
            "meeting_dynamics": meeting_dynamics
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
- Match the output language to meeting_language.
- Professional.
- Based on the meeting name and date.
- Adapt to audience:
  - For "לקוח": more formal and external-facing.
  - For "הנהלה": concise and executive-style.
  - For "צוות פנימי": more direct and practical.

Body requirements:
- Greeting rules:
  - If meeting_language is "אנגלית" → start with "Hello,".
  - If meeting_language is "עברית" → start with "שלום רב,".
  - If meeting_language is "אוטומטי" → detect from transcript and choose accordingly.
- Match the greeting tone to the audience:
  - For "לקוח": slightly more formal.
  - For "צוות פנימי": can be a bit lighter.
- Then one short opening sentence about the meeting.
- Then ONLY the key insights that truly matter.
- Then a short section for follow-up tasks if there are action items.
- Closing rules:
  - If meeting_language is "אנגלית" → end naturally in English, for example "Best regards," or another natural professional ending.
  - If meeting_language is "עברית" → end naturally in Hebrew, for example "בברכה," or another natural professional ending.
  - If meeting_language is "אוטומטי" → detect from transcript and choose accordingly.
- Keep it concise and useful.
- Each email style must feel clearly different in structure, tone, and length.
- Avoid generating similar outputs across styles.
- For "קצר": no bullets, maximum 3-4 short sentences total.
- For "ניהולי": use bullets and emphasize decisions, risks, and next steps.
- For "ידידותי": prefer a more natural flowing structure and warmer wording.
- For "מקצועי": keep a balanced formal structure.
- For "לקוח": avoid internal operational details and present the meeting as a clear external-facing summary.
- Do not mention "attachments", "links", "scraped content", or "website content" unless absolutely necessary for understanding the meeting.
"""

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": "You generate structured professional meeting summary emails that strictly follow the requested language, audience, and style."
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




@app.route("/teams/prepare-recording", methods=["POST"])
def teams_prepare_recording():
    try:
        data = request.get_json(silent=True) or {}

        meeting_name = (data.get("meeting_name") or "").strip()
        teams_url = (data.get("teams_url") or "").strip()

        if not meeting_name:
            return jsonify({"ok": False, "error": "Missing meeting_name"}), 400

        if not teams_url:
            return jsonify({"ok": False, "error": "Missing teams_url"}), 400

        if not is_valid_teams_url(teams_url):
            return jsonify({"ok": False, "error": "Invalid Teams URL"}), 400

        print("📅 Teams prepare request received", flush=True)
        print(f"   meeting_name: {meeting_name}", flush=True)
        print(f"   teams_url: {teams_url}", flush=True)

        session = create_teams_session_record(meeting_name, teams_url)
        response = build_teams_prepare_response(session)

        print(f"   session_id: {response['session_id']}", flush=True)
        print(f"   status: {response['status']}", flush=True)
        print(f"   job_state: {response['job_state']}", flush=True)

        return jsonify(response)

    except Exception as e:
        print(f"❌ teams_prepare_recording error: {e}", flush=True)
        return jsonify({"ok": False, "error": str(e)}), 500



@app.route("/teams/session/<session_id>", methods=["GET"])
def get_teams_session(session_id):
    session = TEAMS_SESSIONS.get(session_id)

    if not session:
        return jsonify({"ok": False, "error": "Session not found"}), 404

    session = refresh_teams_session(session)

    print(f"🔄 Teams session status refresh: {session_id} -> {session['status']} ({session.get('job_state')})", flush=True)

    return jsonify({
        "ok": True,
        "session": session
    })


@app.route("/teams/session/<session_id>/start-bot", methods=["POST"])
def start_teams_session_bot(session_id):
    session = TEAMS_SESSIONS.get(session_id)

    if not session:
        return jsonify({"ok": False, "error": "Session not found"}), 404

    try:
        session = start_teams_bot_for_session(session)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    print(f"🚀 Teams start-bot requested: {session_id} ({session.get('job_state')})", flush=True)

    return jsonify({
        "ok": True,
        "message": "Teams bot start request accepted.",
        "session": session
    })



@app.route("/teams/session/<session_id>/stop", methods=["POST"])
def stop_teams_session(session_id):
    session = TEAMS_SESSIONS.get(session_id)

    if not session:
        return jsonify({"ok": False, "error": "Session not found"}), 404

    session = stop_teams_bot_for_session(session)

    print(f"🛑 Teams session stopped: {session_id}", flush=True)

    return jsonify({
        "ok": True,
        "message": "Teams session stopped.",
        "session": session
    })



@app.route("/api/messages", methods=["POST"])
def teams_bot_messages_webhook():
    payload = request.get_json(silent=True)
    raw_body = request.get_data(as_text=True)

    event = append_teams_bot_event(
        event_type="bot_messages_webhook",
        payload=payload if payload is not None else raw_body,
        note="Received Bot Framework messages webhook",
    )

    print("📨 /api/messages webhook received", flush=True)
    if isinstance(payload, dict):
        print(f"   keys: {list(payload.keys())}", flush=True)
    else:
        print("   payload is not valid JSON", flush=True)

    return jsonify({
        "ok": True,
        "accepted": True,
        "event_id": event["event_id"],
    }), 202


@app.route("/api/calling", methods=["POST"])
def teams_bot_calling_webhook():
    payload = request.get_json(silent=True)
    raw_body = request.get_data(as_text=True)

    session = attach_event_to_matching_session(payload)
    event = append_teams_bot_event(
        event_type="bot_calling_webhook",
        payload=payload if payload is not None else raw_body,
        session_id=session.get("session_id") if session else None,
        note="Received Teams calling webhook",
    )

    if session:
        bot_events = session.setdefault("bot_events", [])
                bot_events = session.setdefault("bot_events", [])
        bot_events.append({
            "event_id": event["event_id"],
            "received_at": event["received_at"],
            "event_type": event["event_type"],
        })
        if len(bot_events) > 50:
            del bot_events[:-50]
        save_teams_state()

        update_session_from_calling_event(session, payload if isinstance(payload, dict) else {})
        print(f"📞 /api/calling webhook matched session: {session.get('session_id')}", flush=True)
        print(f"   status -> {session.get('status')} ({session.get('job_state')})", flush=True)
    else:
        print("📞 /api/calling webhook received with no matching session", flush=True)

    return jsonify({
        "ok": True,
        "accepted": True,
        "event_id": event["event_id"],
        "matched_session_id": session.get("session_id") if session else None,
    }), 202



@app.route("/teams/events", methods=["GET"])
def list_teams_bot_events():
    return jsonify({
        "ok": True,
        "events": list(reversed(TEAMS_BOT_EVENT_LOGS[-50:])),
    })


# Per-session Teams events endpoint
@app.route("/teams/session/<session_id>/events", methods=["GET"])
def list_teams_session_events(session_id):
    session = TEAMS_SESSIONS.get(session_id)

    if not session:
        return jsonify({"ok": False, "error": "Session not found"}), 404

    session_event_refs = list(reversed(session.get("bot_events", [])[-50:]))
    session_event_ids = {
        item.get("event_id")
        for item in session_event_refs
        if isinstance(item, dict) and item.get("event_id")
    }

    detailed_events = [
        event for event in reversed(TEAMS_BOT_EVENT_LOGS[-200:])
        if event.get("event_id") in session_event_ids
    ]

    return jsonify({
        "ok": True,
        "session_id": session_id,
        "events": detailed_events,
        "event_refs": session_event_refs,
    })

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


@app.route("/", methods=["GET"])
def home():
    return "Transcription server is running 🚀"


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
