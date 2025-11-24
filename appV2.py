import os
import gc
import tempfile
import subprocess
import threading
import re
from typing import Optional

import numpy as np
import imageio_ffmpeg
import whisper.audio
import streamlit as st
from dotenv import load_dotenv
from cryptography.fernet import Fernet
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


# ========== TRANSCRIPT MANAGEMENT ==========
def _normalize_text_for_comp(text: str) -> str:
    """Normalize text for comparison to avoid duplicates."""
    if text is None:
        return ""
    t = text.lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _get_last_segment() -> str:
    """Return last appended segment from transcript."""
    if not st.session_state.get("encrypted_transcript"):
        return ""
    try:
        fernet = get_session_fernet()
        full = decrypt_text(st.session_state.encrypted_transcript, fernet)
        parts = [p.strip() for p in full.split("\n\n---\n\n") if p.strip()]
        return parts[-1] if parts else ""
    except Exception:
        return ""


def append_transcript(new_text: str, *, force_append: bool = False):
    """Append new_text safely and skip duplicates."""
    if not new_text:
        return

    last = _get_last_segment()
    if not force_append:
        if _normalize_text_for_comp(last) == _normalize_text_for_comp(new_text):
            return  # skip duplicate

    fernet = get_session_fernet()
    prev_text = ""
    if st.session_state.get("encrypted_transcript"):
        try:
            prev_text = decrypt_text(st.session_state.encrypted_transcript, fernet)
        except Exception:
            prev_text = ""
    combined_text = (prev_text + "\n\n---\n\n" + new_text).strip() if prev_text else new_text.strip()
    st.session_state.encrypted_transcript = encrypt_text(combined_text, fernet)


# ========== PROMPTS ==========
with open("prompts/prompt_start.txt", "r", encoding="utf-8") as f:
    DEFAULT_SYSTEM_PROMPT = f.read()

TEMPLATES = {
    "Summary": "Summarize main concerns, pertinent positives/negatives, and proposed plan.",
    "SOAP": "Produce a SOAP-style summary (Subjective, Objective, Assessment, Plan).",
    "Follow-ups": "List 3–5 follow-up questions to clarify key uncertainties.",
    "Patient education": "Draft a plain-language explanation and next steps for the patient.",
}


# ========== UI CONFIG ==========
st.set_page_config(page_title="Provider Comms MVP", page_icon="🎙️", layout="wide")

st.markdown(
    """
    <style>
    body, .stApp { background-color: #000000; color: #FFFFFF; }
    div[data-testid="stMarkdownContainer"] p, label, span, h1, h2, h3, h4 {
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ========== DISCLAIMER ==========
st.markdown(
    """
    <div style="background-color:#111111; color:#ffffff; border:1px solid #444; padding:15px; border-radius:10px;">
    <strong>⚠️ Disclaimer:</strong><br>
    This application records and transcribes audio for <b>clinical documentation assistance</b> only.
    Please ensure you have obtained explicit consent from all participants before recording.
    <br><br>
    • <b>Data handling:</b> All audio and text data are processed and encrypted in-memory only during this session.<br>
    • <b>Privacy notice:</b> This prototype is intended for demonstration and evaluation purposes and is not a substitute for official electronic medical record (EMR) or documentation systems.<br>
    • <b>HIPAA compliance:</b> Please note that this tool is <b>not yet HIPAA-compliant</b>. We are actively in the process of implementing full HIPAA safeguards and compliance measures, including secure data storage, user authentication, and audit logging.<br><br>
    <em>By continuing, you acknowledge that you understand the data privacy implications and consent to the use of this prototype for testing and evaluation purposes only.</em>
    </div>
    """,
    unsafe_allow_html=True,
)

agree = st.checkbox("✅ I understand and agree to the recording disclaimer above.")
hipaa_ack = st.checkbox("✅ I acknowledge that this tool is not yet HIPAA-compliant and is for testing only.")

if not (agree and hipaa_ack):
    st.warning("You must agree to both the disclaimer and HIPAA acknowledgment before using this app.")
    st.stop()


# ========== TABS ==========
tab_main, tab_debug = st.tabs(["🎙️ Chat & Recording", "🧩 Transcript Debug (DELETE BEFORE RELEASE)"])


# ========== MAIN TAB ==========
with tab_main:
    st.title("🎙️ Provider Communication Assistant — Unified View")

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

    # Initialize session vars
    for key, val in {
        "encrypted_transcript": None,
        "messages": [],
        "recording_thread": None,
        "recording_status": "idle",
    }.items():
        st.session_state.setdefault(key, val)

    st.subheader("🎧 Record or Upload Audio")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎙️ Record Audio")
        audio_data = st.audio_input("Record up to 60 seconds")

    with col2:
        st.markdown("#### 📁 Upload Audio File")
        uploaded = st.file_uploader("Select a file", type=["wav", "mp3", "m4a", "ogg", "webm"])

    source, raw_bytes = None, None
    if uploaded is not None:
        raw_bytes = uploaded.read()
        source = uploaded.name
    elif audio_data is not None:
        raw_bytes = audio_data.getvalue()
        source = "Browser recording"

    # === BACKGROUND THREAD FOR RECORDING ===
    def background_transcribe_recording(raw_bytes, source, whisper_size):
        local_bytes = bytes(raw_bytes)
        st.session_state.recording_status = f"Transcribing {source}..."
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(local_bytes)
            tmp_path = tmp.name
        try:
            new_text = transcribe_audio(tmp_path, model_size=whisper_size)
            append_transcript(new_text)
            st.session_state.last_transcribed_source = f"{source}_{np.random.randint(1e6)}"
            st.session_state.recording_status = f"✅ Recording transcription complete for {source}"
        except Exception as e:
            st.session_state.recording_status = f"[Transcription error: {e}]"
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            st.session_state.recording_thread = None
            gc.collect()
            st.rerun()

    # === HANDLE INPUTS ===
    if raw_bytes is not None:
        if source == "Browser recording":
            if st.session_state.recording_thread is None or not st.session_state.recording_thread.is_alive():
                st.session_state.recording_thread = threading.Thread(
                    target=background_transcribe_recording,
                    args=(raw_bytes, source, whisper_size),
                    daemon=True,
                )
                st.session_state.recording_thread.start()
                st.info("🎙️ Recording is being transcribed in the background.")
            else:
                st.warning("A recording is already being transcribed. Please wait.")
        else:
            # Handle uploads synchronously
            suffix = ".wav" if source.lower().endswith(".wav") else ".mp3"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(raw_bytes)
                tmp_path = tmp.name
            with st.spinner("Transcribing uploaded file..."):
                try:
                    new_text = transcribe_audio(tmp_path, model_size=whisper_size)
                    append_transcript(new_text)
                    st.session_state.last_transcribed_source = source
                    st.success(f"✅ Transcription complete and appended from {source}")
                except Exception as e:
                    st.error(f"[Transcription error] {e}")
                finally:
                    os.unlink(tmp_path)
                    gc.collect()

    if st.session_state.recording_status != "idle":
        st.info(st.session_state.recording_status)

    # === CHAT ===
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


# ========== DEBUG TAB ==========
with tab_debug:
    st.title("🧩 Transcript Debug")
    st.warning("⚠️ FOR TESTING ONLY — DELETE THIS TAB BEFORE RELEASE.")
    if st.session_state.get("encrypted_transcript"):
        fernet = get_session_fernet()
        decrypted = decrypt_text(st.session_state.encrypted_transcript, fernet)
        st.text_area("Current Transcript (debug only):", value=decrypted, height=300)
    else:
        st.info("No transcript available yet.")
