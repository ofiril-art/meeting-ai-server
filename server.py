{\rtf1\ansi\ansicpg1252\cocoartf2869
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 from flask import Flask, request, jsonify\
from openai import OpenAI\
import os\
import tempfile\
\
app = Flask(__name__)\
\
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))\
\
\
@app.route("/", methods=["GET"])\
def home():\
    return "Meeting AI transcription server is running"\
\
\
@app.route("/transcribe", methods=["POST"])\
def transcribe():\
    if "file" not in request.files:\
        return jsonify(\{"error": "No file uploaded"\}), 400\
\
    uploaded_file = request.files["file"]\
\
    if uploaded_file.filename == "":\
        return jsonify(\{"error": "Empty filename"\}), 400\
\
    suffix = os.path.splitext(uploaded_file.filename)[1] or ".m4a"\
\
    try:\
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:\
            uploaded_file.save(temp_file.name)\
            temp_path = temp_file.name\
\
        with open(temp_path, "rb") as audio_file:\
            transcript = client.audio.transcriptions.create(\
                model="gpt-4o-mini-transcribe",\
                file=audio_file\
            )\
\
        os.remove(temp_path)\
\
        return jsonify(\{\
            "text": transcript.text\
        \})\
\
    except Exception as e:\
        return jsonify(\{\
            "error": str(e)\
        \}), 500\
\
\
if __name__ == "__main__":\
    app.run(host="0.0.0.0", port=8080)}