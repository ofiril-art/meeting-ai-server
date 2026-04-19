from datetime import datetime, timedelta, timezone
from flask import Flask, request, jsonify
from urllib.parse import urlparse
import os
import re
import json
import tempfile
import subprocess
import glob
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
TEAMS_JOB_STATES_ACTIVE = {"created", "start_requested", "booting_bot", "joining_meeting", "incoming_call", "joining", "connected", "recording", "failed"}
TEAMS_BOT_EVENT_LOGS = []

TEAMS_STATE_FILE = os.getenv("TEAMS_STATE_FILE", "teams_state.json")

ZOOM_HOST_PATTERNS = {
    "zoom.us",
    "www.zoom.us",
    "us02web.zoom.us",
    "us03web.zoom.us",
    "us04web.zoom.us",
    "us05web.zoom.us",
    "us06web.zoom.us",
}
ZOOM_SESSIONS = {}


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=600.0,
)


TRANSCRIPTION_CHUNK_SECONDS = int(os.getenv("TRANSCRIPTION_CHUNK_SECONDS", "60"))
TRANSCRIPTION_CHUNK_OVERLAP_SECONDS = float(os.getenv("TRANSCRIPTION_CHUNK_OVERLAP_SECONDS", "8"))
FFMPEG_AUDIO_RATE = os.getenv("FFMPEG_AUDIO_RATE", "16000")
FFMPEG_AUDIO_CHANNELS = os.getenv("FFMPEG_AUDIO_CHANNELS", "1")


def run_ffmpeg_command(args):
    completed = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffmpeg command failed")
    return completed


def normalize_audio_for_transcription(input_path: str, output_path: str):
    run_ffmpeg_command([
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vn",
        "-ac", FFMPEG_AUDIO_CHANNELS,
        "-ar", FFMPEG_AUDIO_RATE,
        "-c:a", "pcm_s16le",
        output_path,
    ])


def get_audio_duration_seconds(input_path: str) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffprobe command failed")

    output = (completed.stdout or "").strip()
    if not output:
        raise RuntimeError("ffprobe returned empty duration")

    return float(output)


def split_audio_into_chunks(input_path: str, output_dir: str, chunk_seconds: int):
    total_duration = get_audio_duration_seconds(input_path)
    overlap_seconds = max(0.0, TRANSCRIPTION_CHUNK_OVERLAP_SECONDS)

    if overlap_seconds >= float(chunk_seconds):
        raise RuntimeError("TRANSCRIPTION_CHUNK_OVERLAP_SECONDS must be smaller than TRANSCRIPTION_CHUNK_SECONDS")

    step_seconds = float(chunk_seconds) - overlap_seconds
    chunk_paths = []
    start_seconds = 0.0
    chunk_index = 1

    while start_seconds < total_duration:
        remaining = max(0.0, total_duration - start_seconds)
        current_duration = min(float(chunk_seconds), remaining)
        chunk_path = os.path.join(output_dir, f"chunk_{chunk_index:03d}.wav")

        run_ffmpeg_command([
            "ffmpeg",
            "-y",
            "-ss", f"{start_seconds:.3f}",
            "-t", f"{current_duration:.3f}",
            "-i", input_path,
            "-c:a", "pcm_s16le",
            chunk_path,
        ])

        chunk_paths.append(chunk_path)
        chunk_index += 1

        if current_duration < float(chunk_seconds):
            break

        start_seconds += step_seconds

    return chunk_paths


def transcribe_audio_file_with_openai(audio_path: str):
    with open(audio_path, "rb") as audio_file:
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
    return (transcription.text or "").strip()


# --- Chunk overlap deduplication helpers ---
def normalize_text_for_dedup(text: str) -> str:
    text = (text or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\u200f\u200e\-–—\"'“”׳״,.;:!?()\[\]{}]", "", text)
    return text.strip()


def sentence_split_for_dedup(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [part.strip() for part in parts if part and part.strip()]


def deduplicate_overlap_between_chunks(chunk_texts: list[str], max_overlap_sentences: int = 6) -> str:
    if not chunk_texts:
        return ""

    merged_sentences: list[str] = []

    for chunk_text in chunk_texts:
        chunk_sentences = sentence_split_for_dedup(chunk_text)
        if not chunk_sentences:
            continue

        if not merged_sentences:
            merged_sentences.extend(chunk_sentences)
            continue

        overlap_to_skip = 0
        max_check = min(max_overlap_sentences, len(merged_sentences), len(chunk_sentences))

        for size in range(max_check, 0, -1):
            tail = merged_sentences[-size:]
            head = chunk_sentences[:size]

            tail_norm = [normalize_text_for_dedup(item) for item in tail]
            head_norm = [normalize_text_for_dedup(item) for item in head]

            if tail_norm == head_norm:
                overlap_to_skip = size
                break

        merged_sentences.extend(chunk_sentences[overlap_to_skip:])

    deduped = "\n\n".join(sentence.strip() for sentence in merged_sentences if sentence.strip()).strip()
    return deduped


# --- Remove adjacent near-duplicate passages helper ---
def remove_adjacent_near_duplicate_passages(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    passages = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not passages:
        return text

    cleaned: list[str] = []

    for passage in passages:
        current_norm = normalize_text_for_dedup(passage)

        if not cleaned:
            cleaned.append(passage)
            continue

        previous = cleaned[-1]
        previous_norm = normalize_text_for_dedup(previous)

        if not current_norm:
            continue

        if current_norm == previous_norm:
            continue

        if current_norm in previous_norm:
            continue

        if previous_norm in current_norm:
            cleaned[-1] = passage
            continue

        cleaned.append(passage)

    return "\n\n".join(cleaned).strip()


def transcribe_with_chunking(input_path: str):
    working_dir = tempfile.mkdtemp(prefix="transcribe_chunks_")
    normalized_path = os.path.join(working_dir, "normalized.wav")

    try:
        normalize_audio_for_transcription(input_path, normalized_path)
        chunk_paths = split_audio_into_chunks(
            normalized_path,
            working_dir,
            TRANSCRIPTION_CHUNK_SECONDS,
        )

        if not chunk_paths:
            chunk_paths = [normalized_path]

        chunk_texts = []
        for index, chunk_path in enumerate(chunk_paths, start=1):
            chunk_size = os.path.getsize(chunk_path) if os.path.exists(chunk_path) else 0
            print(
                f"🎙️ Transcribing chunk {index}/{len(chunk_paths)}: {os.path.basename(chunk_path)} ({chunk_size} bytes)",
                flush=True,
            )
            chunk_text = transcribe_audio_file_with_openai(chunk_path)
            if chunk_text:
                chunk_texts.append(chunk_text.strip())

        print(
            f"✅ Completed chunked transcription with {len(chunk_texts)} successful chunks out of {len(chunk_paths)}",
            flush=True,
        )

        combined_text = deduplicate_overlap_between_chunks(chunk_texts)
        combined_text = remove_adjacent_near_duplicate_passages(combined_text)
        print(
            f"🧩 Deduplicated overlapping chunk text. Raw chunks: {len(chunk_texts)}, final chars: {len(combined_text)}",
            flush=True,
        )
        return combined_text
    finally:
        for path in glob.glob(os.path.join(working_dir, "*")):
            try:
                os.remove(path)
            except Exception:
                pass
        try:
            os.rmdir(working_dir)
        except Exception:
            pass


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


# --- Hebrew transcript cleanup, entity correction, and promo/ad stripping helpers ---
def apply_phrase_replacements(text: str) -> str:
    if not text:
        return ""

    replacements = [
        ("שר השיקון", "שר השיכון"),
        ("גולד קנופף", "גולדקנופף"),
        ("גולד קנופ", "גולדקנופף"),
        ("טילה", "תילה"),
        ("חורבת טילה", "חורבת תילה"),
        ("איילד שקד", "איילת שקד"),
        ("הילד שקד", "איילת שקד"),
        ("מערעת", "מרהט"),
        ("דרום הערבי", "דרום־מערבי"),
        ("בר שבע", "באר שבע"),
        ("בער שבע", "באר שבע"),
        ("מודיעין אילית", "מודיעין עילית"),
        ("ביתר אילית", "ביתר עילית"),
        ("היהרות הפיתוח", "עיירות הפיתוח"),
        ("חריגות בנייה", "חריגות בנייה"),
        ("חגוגות בנייה", "חריגות בנייה"),
        ("הר המסע", "הר עמשא"),
        ("לבנה", "ליבנה"),
        ("יאטה", "יאטא"),
        ("הולנטה", "הולנת\"ע"),
        ("המועצה הארצית לתכנון ובנייה", "המועצה הארצית לתכנון ולבנייה"),
        ("תמה 35", 'תמ"א 35'),
        ("מטר רבוע", "מטר רבוע"),
        ("פי שניים", "פי שניים"),
        ("העסוקה", "תעסוקה"),
    ]

    for wrong, correct in replacements:
        text = text.replace(wrong, correct)

    text = re.sub(r"\b20 קילומטרים מערהט\b", "20 קילומטרים מרהט", text)
    text = re.sub(r"\b20 קילומטרים מערעת\b", "20 קילומטרים מרהט", text)
    text = re.sub(r"\b25 קילומטרים מבאר שבע\b", "25 קילומטרים מבאר שבע", text)
    text = re.sub(r"\b50 אלף דירות ו-80 עד 100 אלף תושבים\b", "50 אלף דירות ו־80–100 אלף תושבים", text)
    text = re.sub(r"\b7 אחוזים בשנה\b", "7% בשנה", text)
    text = re.sub(r"\b70 אלף שקלים למטר רבוע\b", "70 אלף שקל למ\"ר", text)
    text = re.sub(r"\b200 אלף יחידות דיור\b", "200 אלף יחידות דיור", text)

    return text


def strip_trailing_promos(text: str) -> str:
    if not text:
        return ""

    promo_markers = [
        "הכותב הוא",
        "פצועי צה\"ל",
        "פצועות צה\"ל",
        "נפש אחת",
        "קו הסיוע והתמיכה",
        "מכל הלב לכל החיים",
    ]

    earliest_index = None
    for marker in promo_markers:
        index = text.find(marker)
        if index != -1 and (earliest_index is None or index < earliest_index):
            earliest_index = index

    if earliest_index is not None:
        text = text[:earliest_index].rstrip()

    return text.strip()


def clean_hebrew_transcript(text: str) -> str:
    if not text:
        return ""

    text = apply_phrase_replacements(text)
    text = strip_trailing_promos(text)
    text = re.sub(r"\bשתיים\.\s*נתחיל בהתחלה\.\b", "2. נתחיל בהתחלה.", text)
    text = re.sub(r"\bשלוש\.\s*החברה החרדית\b", "3. החברה החרדית", text)
    text = re.sub(r"\bארבע\.\s*מצוקת הדיור\b", "4. מצוקת הדיור", text)
    text = re.sub(r"\bחמש\.\s*וזה כבר\b", "5. וזה כבר", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --- Hebrew transcript AI repair helpers ---
def contains_hebrew(text: str) -> bool:
    return bool(text) and re.search(r"[\u0590-\u05FF]", text) is not None



def split_text_for_ai_repair(text: str, max_chars: int = 5000) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
            current = ""

        if len(paragraph) <= max_chars:
            current = paragraph
            continue

        sentence_parts = re.split(r"(?<=[.!?])\s+", paragraph)
        sentence_parts = [s.strip() for s in sentence_parts if s.strip()]

        sentence_chunk = ""
        for sentence in sentence_parts:
            sentence_candidate = f"{sentence_chunk} {sentence}".strip() if sentence_chunk else sentence
            if len(sentence_candidate) <= max_chars:
                sentence_chunk = sentence_candidate
            else:
                if sentence_chunk:
                    chunks.append(sentence_chunk.strip())
                sentence_chunk = sentence
        if sentence_chunk:
            chunks.append(sentence_chunk.strip())
    if current:
        chunks.append(current.strip())

    return [chunk for chunk in chunks if chunk]


def repair_hebrew_transcript_chunk_with_ai(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": """
You repair Hebrew ASR transcript text.

Goals:
- Fix obvious Hebrew transcription mistakes.
- Fix obvious place names, person names, numbers, and planning/policy terms when they are strongly implied by the text.
- Preserve the original meaning and structure.
- Keep the response in Hebrew.
- Keep English terms in English if they were spoken that way.
- Do NOT summarize.
- Do NOT rewrite stylistically.
- Do NOT invent facts, sentences, names, numbers, or sections that are not strongly supported by the original transcript.
- Do NOT add introductions, explanations, markdown, or quotation marks.
- Return only the repaired transcript text.

Be conservative:
- Only fix clear ASR mistakes.
- If unsure, keep the original wording.
"""
            },
            {
                "role": "user",
                "content": text
            }
        ]
    )

    repaired = (response.output_text or "").strip()
    return repaired or text


def repair_hebrew_transcript_with_ai(text: str, max_chars: int = 5000) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    if not contains_hebrew(text):
        return text

    chunks = split_text_for_ai_repair(text, max_chars=max_chars)
    if not chunks:
        return text

    repaired_chunks = []
    for index, chunk in enumerate(chunks, start=1):
        print(f"🛠️ Repairing Hebrew transcript chunk {index}/{len(chunks)}", flush=True)
        repaired_chunks.append(repair_hebrew_transcript_chunk_with_ai(chunk))

    return "\n\n".join(chunk.strip() for chunk in repaired_chunks if chunk.strip()).strip() or text


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


# Helper: build analysis input for meeting analysis (head + tail, with truncation marker)
def build_analysis_input(text: str, max_chars: int = 50000) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text

    head_target = max_chars // 2
    tail_target = max_chars - head_target

    head = text[:head_target].rstrip()
    tail = text[-tail_target:].lstrip()

    return (
        head
        + "\n\n[... transcript truncated for analysis due to length ...]\n\n"
        + tail
    ).strip()


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


# --- Robust meeting-analysis parsing helpers ---
def repair_analysis_json(raw_output: str) -> str:
    repaired = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": """
You receive malformed or noisy meeting-analysis output.
Your job is to convert it into VALID JSON ONLY.

Return ONLY valid JSON with this exact structure:
{
  "summary": "string",
  "action_items": ["string"],
  "speakers": [
    {
      "speaker": "string",
      "role_hint": "string",
      "highlights": ["string"]
    }
  ],
  "action_items_by_speaker": [
    {
      "speaker": "string",
      "items": ["string"]
    }
  ],
  "meeting_dynamics": {
    "communication_style": "string",
    "decision_pattern": "string",
    "alignment_level": "string",
    "key_tensions": ["string"]
  }
}

Rules:
- Preserve the meaning from the original output.
- Do not invent facts.
- If a field is missing, use an empty string or empty list.
- Return JSON only. No markdown. No explanation.
"""
            },
            {
                "role": "user",
                "content": raw_output or ""
            }
        ]
    )
    return (repaired.output_text or "").strip()


def generate_fallback_meeting_analysis_from_transcript(transcript_text: str) -> dict:
    fallback_response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": """
Generate a conservative meeting analysis from a transcript.
Return ONLY valid JSON with this exact structure:
{
  "summary": "string",
  "action_items": ["string"],
  "speakers": [
    {
      "speaker": "string",
      "role_hint": "string",
      "highlights": ["string"]
    }
  ],
  "action_items_by_speaker": [
    {
      "speaker": "string",
      "items": ["string"]
    }
  ],
  "meeting_dynamics": {
    "communication_style": "string",
    "decision_pattern": "string",
    "alignment_level": "string",
    "key_tensions": ["string"]
  }
}

Rules:
- Respond in Hebrew.
- Keep English business and product terms in English if they appear that way in the transcript.
- Be conservative.
- Do not invent facts, owners, or decisions.
- If something is unclear, leave it empty.
- Return JSON only.
"""
            },
            {
                "role": "user",
                "content": transcript_text or ""
            }
        ]
    )
    parsed = json_from_text(fallback_response.output_text)
    return normalize_meeting_analysis(parsed)


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


def is_valid_zoom_url(url: str) -> bool:
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

    if host in ZOOM_HOST_PATTERNS:
        return True

    return host.endswith(".zoom.us")


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


def create_zoom_session_record(meeting_name: str, zoom_url: str):
    now = datetime.utcnow()
    session_id = f"zoom_{now.strftime('%Y%m%d%H%M%S%f')}"
    created_at = utc_now_iso()

    session = {
        "session_id": session_id,
        "provider": "zoom",
        "join_url": zoom_url,
        "zoom_url": zoom_url,
        "meeting_name": meeting_name,
        "status": "queued",
        "mode": "prepare_only",
        "bot_mode": "mock",
        "job_state": "created",
        "last_error": None,
        "created_at": created_at,
        "updated_at": created_at,
        "received_at": created_at,
        "started_at": None,
        "stopped_at": None,
    }

    ZOOM_SESSIONS[session_id] = session
    return session



def build_zoom_prepare_response(session):
    return {
        "ok": True,
        "session_id": session["session_id"],
        "status": session["status"],
        "provider": session["provider"],
        "mode": session["mode"],
        "bot_mode": session["bot_mode"],
        "job_state": session["job_state"],
        "message": "Zoom meeting received. Bot integration is not connected yet.",
        "meeting_name": session["meeting_name"],
        "zoom_url": session["zoom_url"],
        "join_url": session["join_url"],
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
        "received_at": session["received_at"],
        "last_error": session["last_error"],
    }


def apply_zoom_session_updates(session: dict, **updates):
    for key, value in updates.items():
        session[key] = value
    session["updated_at"] = utc_now_iso()
    return session



def refresh_zoom_session(session: dict):
    return update_mock_zoom_session_status(session)



def start_zoom_bot_for_session(session: dict):
    current_status = (session.get("status") or "").strip().lower()
    current_job_state = (session.get("job_state") or "").strip().lower()

    if current_status == "stopped":
        raise ValueError("Session already stopped")

    if current_job_state in {"start_requested", "joining", "recording"}:
        return session

    return apply_zoom_session_updates(
        session,
        job_state="start_requested",
        last_error=None,
    )



def stop_zoom_bot_for_session(session: dict):
    now = utc_now_iso()
    return apply_zoom_session_updates(
        session,
        status="stopped",
        job_state="stopped",
        stopped_at=now,
        last_error=None,
    )



def update_mock_zoom_session_status(session):
    current_status = (session.get("status") or "").strip().lower()
    current_job_state = (session.get("job_state") or "").strip().lower()

    if current_status == "stopped":
        return session

    if current_job_state in {"recording", "failed"}:
        return session

    received_at = session.get("received_at") or ""

    try:
        received_dt = datetime.fromisoformat(received_at.replace("Z", "+00:00"))
    except Exception:
        return session

    now = datetime.now(timezone.utc)
    elapsed_seconds = max(0, int((now - received_dt).total_seconds()))

    if current_job_state == "start_requested" and elapsed_seconds < 10:
        return apply_zoom_session_updates(
            session,
            status="queued",
            job_state="start_requested",
            last_error=None,
        )
    elif elapsed_seconds < 10:
        return apply_zoom_session_updates(
            session,
            status="queued",
            job_state="created",
            last_error=None,
        )
    elif elapsed_seconds < 20:
        return apply_zoom_session_updates(
            session,
            status="bot_joining",
            job_state="joining",
            started_at=session.get("started_at") or utc_now_iso(),
            last_error=None,
        )
    else:
        return apply_zoom_session_updates(
            session,
            status="recording",
            job_state="recording",
            started_at=session.get("started_at") or utc_now_iso(),
            last_error=None,
        )


def update_mock_teams_session_status(session):
    current_status = (session.get("status") or "").strip().lower()
    current_job_state = (session.get("job_state") or "").strip().lower()

    if current_status == "stopped":
        return session

    # Do not let the timer-based mock flow override real calling webhook state.
    # Once calling events moved the session into a live bot/call state, keep it as-is
    # until a new webhook changes it again.
    if current_job_state in {"incoming_call", "joining", "connected", "recording", "failed"}:
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
        print(f"📥 Uploaded file received: {uploaded_file.filename}", flush=True)

        if uploaded_file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        suffix = os.path.splitext(uploaded_file.filename)[1] or ".m4a"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            uploaded_file.save(temp_file.name)
            temp_path = temp_file.name
        try:
            temp_size = os.path.getsize(temp_path)
            print(f"📦 Saved upload to temp file: {temp_path} ({temp_size} bytes)", flush=True)
        except Exception:
            print(f"📦 Saved upload to temp file: {temp_path}", flush=True)

        try:
            raw_text = transcribe_with_chunking(temp_path)
        except Exception as chunk_error:
            print(f"❌ Chunked transcription failed: {chunk_error}", flush=True)
            return jsonify({
                "error": f"Chunked transcription failed: {str(chunk_error)}"
            }), 500
        cleaned_text = basic_cleanup(raw_text)
        cleaned_text = fix_mixed_text(cleaned_text)
        cleaned_text = clean_hebrew_transcript(cleaned_text)

        if contains_hebrew(cleaned_text) and len(cleaned_text) >= 1200:
            print("🛠️ Starting AI Hebrew transcript repair", flush=True)
            try:
                cleaned_text = repair_hebrew_transcript_with_ai(cleaned_text, max_chars=5000)
                cleaned_text = clean_hebrew_transcript(cleaned_text)
                print("✅ AI Hebrew transcript repair completed", flush=True)
            except Exception as transcript_repair_error:
                print(f"❌ AI Hebrew transcript repair failed: {transcript_repair_error}", flush=True)

        formatted_text = add_readable_paragraphs(cleaned_text)
        analysis_input = build_analysis_input(formatted_text)

        print(f"📝 Raw transcript chars: {len(raw_text)}", flush=True)
        print(f"📝 Formatted transcript chars: {len(formatted_text)}", flush=True)
        print(f"📝 Analysis input chars: {len(analysis_input)}", flush=True)
        print(f"🧹 Cleaned transcript chars: {len(cleaned_text)}", flush=True)
        print(f"🧾 Final transcript chars after repair: {len(formatted_text)}", flush=True)

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

        print("🧠 Starting meeting analysis generation", flush=True)
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
                    "content": analysis_input
                }
            ]
        )
        print("✅ Meeting analysis generation completed", flush=True)

        try:
            print("🧩 Parsing meeting analysis JSON", flush=True)
            parsed = json_from_text(summary_response.output_text)
            normalized = normalize_meeting_analysis(parsed)
            summary_text = normalized["summary"]
            action_items = normalized["action_items"]
            speakers = normalized["speakers"]
            action_items_by_speaker = normalized["action_items_by_speaker"]
            meeting_dynamics = normalized["meeting_dynamics"]
            print(f"✅ Parsed meeting analysis successfully. Summary chars: {len(summary_text)}, action items: {len(action_items)}", flush=True)
        except Exception as parse_error:
            print(f"❌ Failed to parse meeting analysis JSON: {parse_error}", flush=True)
            print(f"Raw analysis output: {summary_response.output_text}", flush=True)

            try:
                print("🛠️ Attempting to repair malformed meeting analysis JSON", flush=True)
                repaired_output = repair_analysis_json(summary_response.output_text)
                print(f"🛠️ Repaired analysis output: {repaired_output}", flush=True)
                repaired_parsed = json_from_text(repaired_output)
                normalized = normalize_meeting_analysis(repaired_parsed)
                summary_text = normalized["summary"]
                action_items = normalized["action_items"]
                speakers = normalized["speakers"]
                action_items_by_speaker = normalized["action_items_by_speaker"]
                meeting_dynamics = normalized["meeting_dynamics"]
                print(f"✅ Repaired meeting analysis successfully. Summary chars: {len(summary_text)}, action items: {len(action_items)}", flush=True)
            except Exception as repair_error:
                print(f"❌ Failed to repair meeting analysis JSON: {repair_error}", flush=True)

                try:
                    print("🧯 Generating fallback meeting analysis from transcript", flush=True)
                    normalized = generate_fallback_meeting_analysis_from_transcript(analysis_input)
                    summary_text = normalized["summary"]
                    action_items = normalized["action_items"]
                    speakers = normalized["speakers"]
                    action_items_by_speaker = normalized["action_items_by_speaker"]
                    meeting_dynamics = normalized["meeting_dynamics"]
                    print(f"✅ Fallback meeting analysis completed. Summary chars: {len(summary_text)}, action items: {len(action_items)}", flush=True)
                except Exception as fallback_error:
                    print(f"❌ Fallback meeting analysis also failed: {fallback_error}", flush=True)
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

        print("📤 Returning transcription response to client", flush=True)
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

@app.route("/zoom/prepare-recording", methods=["POST"])
def zoom_prepare_recording():
    try:
        data = request.get_json(silent=True) or {}

        meeting_name = (data.get("meeting_name") or "").strip()
        zoom_url = (data.get("zoom_url") or "").strip()

        if not meeting_name:
            return jsonify({"ok": False, "error": "Missing meeting_name"}), 400

        if not zoom_url:
            return jsonify({"ok": False, "error": "Missing zoom_url"}), 400

        if not is_valid_zoom_url(zoom_url):
            return jsonify({"ok": False, "error": "Invalid Zoom URL"}), 400

        print("📅 Zoom prepare request received", flush=True)
        print(f"   meeting_name: {meeting_name}", flush=True)
        print(f"   zoom_url: {zoom_url}", flush=True)

        session = create_zoom_session_record(meeting_name, zoom_url)
        response = build_zoom_prepare_response(session)

        print(f"   session_id: {response['session_id']}", flush=True)
        print(f"   status: {response['status']}", flush=True)
        print(f"   job_state: {response['job_state']}", flush=True)

        return jsonify(response)

    except Exception as e:
        print(f"❌ zoom_prepare_recording error: {e}", flush=True)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/zoom/session/<session_id>", methods=["GET"])
def get_zoom_session(session_id):
    session = ZOOM_SESSIONS.get(session_id)

    if not session:
        return jsonify({"ok": False, "error": "Session not found"}), 404

    session = refresh_zoom_session(session)

    print(f"🔄 Zoom session status refresh: {session_id} -> {session['status']} ({session.get('job_state')})", flush=True)

    return jsonify({
        "ok": True,
        "session": session
    })

@app.route("/zoom/session/<session_id>/start-bot", methods=["POST"])
def start_zoom_session_bot(session_id):
    session = ZOOM_SESSIONS.get(session_id)

    if not session:
        return jsonify({"ok": False, "error": "Session not found"}), 404

    try:
        session = start_zoom_bot_for_session(session)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    print(f"🚀 Zoom start-bot requested: {session_id} ({session.get('job_state')})", flush=True)

    return jsonify({
        "ok": True,
        "message": "Zoom bot start request accepted.",
        "session": session
    })


@app.route("/zoom/session/<session_id>/stop", methods=["POST"])
def stop_zoom_session(session_id):
    session = ZOOM_SESSIONS.get(session_id)

    if not session:
        return jsonify({"ok": False, "error": "Session not found"}), 404

    session = stop_zoom_bot_for_session(session)

    print(f"🛑 Zoom session stopped: {session_id}", flush=True)

    return jsonify({
        "ok": True,
        "message": "Zoom session stopped.",
        "session": session
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
