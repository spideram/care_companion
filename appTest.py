"""
app.py — Unified Page (Recording + Chat) — Encrypted Streamlit MVP
Now includes a temporary transcript debug tab for testing.
⚠️ DELETE THE DEBUG TAB SECTION BEFORE RELEASE.
"""

import os
import gc
import tempfile
import subprocess
from typing import Optional, List, Dict

import numpy as np
import imageio_ffmpeg
import whisper.audio
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# ========== FFMPEG PATCH ==========
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
whisper.audio.load_audio = lambda path: np.frombuffer(
    subprocess.run(
        [
            ffmpeg_path,
            "-nostdin",
            "-threads",
            "0",
            "-i",
            path,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(whisper.audio.SAMPLE_RATE),
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout,
    np.int16,
).flatten().astype(np.float32) / 32768.0

# ========== SETUP ==========
load_dotenv()
from cryptography.fernet import Fernet

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# ========== WHISPER ==========
@st.cache_resource(show_spinner=False)
def load_whisper(model_size: str = "base"):
    import whisper
    return whisper.load_model(model_size)

def transcribe_audio(tmp_path: str, model_size: str = "base") -> str:
    model = load_whisper(model_size)
    result = model.transcribe(tmp_path)
    return result.get("text", "").strip()

# ========== GEMINI ==========
def init_gemini() -> Optional[object]:
    if not GEMINI_AVAILABLE:
        return None
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")

def run_gemini_chat(model, transcript: str, user_prompt: str, system_preamble: str) -> str:
    if model is None:
        return "[Gemini not configured] Provide GEMINI_API_KEY to enable AI responses."
    prompt = f"""{system_preamble}

---
Encounter transcript (verbatim):
{transcript}

---
User request: {user_prompt}

Return a concise, clinically useful answer with bullet points when appropriate.
"""
    try:
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        return f"[Gemini error] {e}"

# ========== ENCRYPTION ==========
def get_session_fernet() -> Fernet:
    if "fernet_key" not in st.session_state:
        st.session_state.fernet_key = Fernet.generate_key()
    return Fernet(st.session_state.fernet_key)

def encrypt_text(plain_text: str, fernet: Fernet) -> bytes:
    return fernet.encrypt(plain_text.encode("utf-8"))

def decrypt_text(blob: bytes, fernet: Fernet) -> str:
    return fernet.decrypt(blob).decode("utf-8", errors="ignore")

# ========== PROMPTS ==========
with open("prompts/prompt_start.txt", "r", encoding="utf-8") as f:
    DEFAULT_SYSTEM_PROMPT = f.read()
    
TEMPLATES = {
    "Summary": "Summarize main concerns, pertinent positives/negatives, and proposed plan.",
    "SOAP": "Produce a SOAP-style summary (Subjective, Objective, Assessment, Plan).",
    "Follow-ups": "List 3–5 follow-up questions to clarify key uncertainties.",
    "Patient education": "Draft a plain-language explanation and next steps for the patient.",
}

# ========== UI ==========
st.set_page_config(page_title="Provider Comms MVP", page_icon="🎙️", layout="wide")

# ⚠️ BLACK BACKGROUND
st.markdown(
    """
    <style>
    body, .stApp { background-color: #000000; color: #FFFFFF; }
    div[data-testid="stMarkdownContainer"] p, label, span, h1, h2, h3, h4 {
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ========== DISCLAIMER ==========
st.markdown(
    """
    <div style="background-color:#111111; color:#ffffff; border:1px solid #444; padding:15px; border-radius:10px;">
    <strong>⚠️ Disclaimer:</strong><br>
    This application records and transcribes audio for clinical documentation assistance only.
    Please ensure you have consent before recording any other person.
    <br><br>
    • <b>Encrypted transcripts:</b> Your recordings are encrypted in-memory only.<br>
    • <b>Privacy notice:</b> This prototype is not a substitute for professional medical documentation systems.
    </div>
    """,
    unsafe_allow_html=True
)

agree = st.checkbox("I understand and agree to the recording disclaimer above.")
if not agree:
    st.warning("You must agree to the disclaimer before using this app.")
    st.stop()

# ========== TABS ==========
tab_main, tab_debug = st.tabs(["🎙️ Chat & Recording", "🧩 Transcript Debug (DELETE BEFORE RELEASE)"])

# ========== MAIN TAB ==========
with tab_main:
    st.title("🎙️ Provider Communication Assistant — Unified View")
    # st.caption("Record → Auto-transcribe (encrypted) → Chat instantly")

    with st.sidebar:
        st.header("Settings")
        whisper_size = st.selectbox(
            "Whisper model",
            options=["tiny", "base", "small", "medium"],
            index=1,
            help="Smaller = faster, lower accuracy.",
        )
        use_gemini = st.checkbox("Use Gemini for chat", value=True)
        st.write("Encryption: **Always On** (session-scoped Fernet key)")

    if "encrypted_transcript" not in st.session_state:
        st.session_state.encrypted_transcript = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # ========== RECORD & AUTO-TRANSCRIBE ==========
    st.subheader("🎧 Record or Upload Audio")
    audio_data = st.audio_input("Record up to 60 seconds")
    uploaded = st.file_uploader("...or upload a file", type=["wav", "mp3", "m4a", "ogg", "webm"])

    # Track source and last processed audio
    source = None
    raw_bytes = None
    if uploaded is not None:
        raw_bytes = uploaded.read()
        source = uploaded.name
    elif audio_data is not None:
        raw_bytes = audio_data.getvalue()
        source = "Browser recording"

    # Only transcribe if it's *new* audio
    if raw_bytes is not None and st.session_state.get("last_transcribed_source") != source:
        with st.spinner("Transcribing with Whisper..."):
            suffix = ".wav" if (uploaded and uploaded.name.lower().endswith(".wav")) else ".mp3"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(raw_bytes)
                tmp_path = tmp.name
            try:
                text = transcribe_audio(tmp_path, model_size=whisper_size)
            finally:
                os.unlink(tmp_path)

        fernet = get_session_fernet()
        st.session_state.encrypted_transcript = encrypt_text(text, fernet)
        st.session_state.last_transcribed_source = source
        st.success(f"Transcription complete and encrypted from {source}.")

        text = None
        gc.collect()

    # ========== CHAT SECTION ==========
    # st.subheader("💬 Chat with the AI Assistant")
    template = st.selectbox("Prompt template", list(TEMPLATES.keys()), index=0)
    system_preamble = DEFAULT_SYSTEM_PROMPT
    system_goal = f"Template: {template}\n\n{TEMPLATES[template]}"
    gemini_model = init_gemini() if use_gemini else None

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("Ask about this encounter (e.g., 'Summarize main concerns')")

    if user_msg:
        st.session_state.messages.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        if st.session_state.encrypted_transcript is None:
            reply = "Please record or upload audio before chatting."
        else:
            fernet = get_session_fernet()
            transcript = decrypt_text(st.session_state.encrypted_transcript, fernet)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    reply = run_gemini_chat(
                        gemini_model if use_gemini else None,
                        transcript=transcript,
                        user_prompt=user_msg,
                        system_preamble=f"{system_preamble}\n\n{system_goal}",
                    )
                    st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        gc.collect()

    st.divider()
    # st.caption(
    #     "Unified MVP • Whisper ASR • Gemini (optional) • Auto-encryption enabled • For demonstration only."
    # )

# ========== DEBUG TAB ==========
with tab_debug:
    st.title("🧩 Transcript Debug")
    st.warning("⚠️ FOR TESTING ONLY — DELETE THIS TAB BEFORE RELEASE.")
    if "encrypted_transcript" in st.session_state and st.session_state.encrypted_transcript:
        fernet = get_session_fernet()
        decrypted = decrypt_text(st.session_state.encrypted_transcript, fernet)
        st.text_area("Current Transcript (debug only):", value=decrypted, height=300)
    else:
        st.info("No transcript available yet.")
