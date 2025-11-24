"""
Streamlit MVP: Patient–Provider Communication Assistant
-------------------------------------------------------
Features (prototype):
- Microphone/uploader to capture encounter audio
- Background transcription using Whisper (local model)
- Prompt-engineered chat over transcript using Gemini (optional; stub if no key)
- Simple at-rest/in-flight demo encryption using Fernet (optional toggle)

Run locally:
  python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\\Scripts\\activate)
  pip install --upgrade pip
  pip install streamlit openai-whisper torch --index-url https://download.pytorch.org/whl/cpu
  pip install google-generativeai cryptography
  # (Optional) If ffmpeg missing, install it via your OS package manager

  # Set env var for Gemini (optional)
  export GEMINI_API_KEY=your_key_here  # (Windows: set GEMINI_API_KEY=...)

  streamlit run app.py

Notes:
- Whisper base model (~142MB) is the default; change in the sidebar for quality/speed.
- "End-to-end" encryption here demonstrates encrypting payloads before API calls. Real HIPAA-grade E2EE,
  logging controls, BAA-backed hosting, and audit trails will be a next phase.
- Simplify UI
- Speed it up
- Hide trancsript
- Put things on one screen
"""

import os
import gc
import tempfile
import subprocess
from typing import Optional, List, Dict
import io
import time
import soundfile as sf
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
    This application records and transcribes audio for <b>clinical documentation assistance</b> only.
    Please ensure you have obtained explicit consent from all participants before recording.
    <br><br>
    • <b>Data handling:</b> All audio and text data are processed and encrypted in-memory only during this session.<br>
    • <b>Privacy notice:</b> This prototype is intended for demonstration and evaluation purposes and is not a substitute for official electronic medical record (EMR) or documentation systems.<br>
    • <b>HIPAA compliance:</b> Please note that this tool is <b>not yet HIPAA-compliant</b>. We are actively in the process of implementing full HIPAA safeguards and compliance measures, including secure data storage, user authentication, and audit logging.<br><br>
    <em>By continuing, you acknowledge that you understand the data privacy implications and consent to the use of this prototype for testing and evaluation purposes only.</em>
    </div>
    """,
    unsafe_allow_html=True
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
    # st.subheader("🎧 Record or Upload Audio")
    # audio_data = st.audio_input("Record up to 60 seconds")
    # uploaded = st.file_uploader("...or upload a file", type=["wav", "mp3", "m4a", "ogg", "webm"])
    st.subheader("🎧 Record or Upload Audio")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎙️ Record Audio")
        audio_data = st.audio_input("Record up to 60 seconds")

    with col2:
        st.markdown("#### 📁 Upload Audio File")
        uploaded = st.file_uploader("Select a file", type=["wav", "mp3", "m4a", "ogg", "webm"])

    # ----------------------------
    # Prepare source + raw_bytes
    # ----------------------------
    source, raw_bytes = None, None
    if uploaded is not None:
        raw_bytes = uploaded.read()
        source = uploaded.name
    elif audio_data is not None:
        # Make each browser recording uniquely identifiable so repeated recordings are processed
        raw_bytes = audio_data.getvalue() if hasattr(audio_data, "getvalue") else bytes(audio_data)
        source = f"Browser recording-{int(time.time()*1000)}"
    # ----------------------------

    # Process new audio (only if source changed)
    if raw_bytes is not None and st.session_state.get("last_transcribed_source") != source:
        # We'll always try to create a valid WAV file for Whisper
        with st.spinner("Preparing audio for transcription..."):
            tmp_wav_path = None
            try:
                # Try to interpret the raw bytes with soundfile (handles many container types)
                try:
                    audio_array, sr = sf.read(io.BytesIO(raw_bytes))
                    # soundfile yields (samples, sr). If samples are multi-channel, soundfile.write will handle it.
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        sf.write(tmp.name, audio_array, sr, format="WAV")
                        tmp_wav_path = tmp.name
                except Exception:
                    # Fallback: raw bytes might already be a WAV container — write as-is and hope for best
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(raw_bytes)
                        tmp_wav_path = tmp.name

                # Now transcribe the WAV file (synchronous)
                new_text = transcribe_audio(tmp_wav_path, model_size=whisper_size)

            finally:
                # Clean up the temp wav
                try:
                    if tmp_wav_path and os.path.exists(tmp_wav_path):
                        os.unlink(tmp_wav_path)
                except Exception:
                    pass

        # Re-use your existing encryption + concatenation logic
        fernet = get_session_fernet()

        # Decrypt previous transcript (if any), then append
        if st.session_state.get("encrypted_transcript"):
            try:
                prev_text = decrypt_text(st.session_state.encrypted_transcript, fernet)
            except Exception:
                prev_text = ""
        else:
            prev_text = ""

        combined_text = (prev_text + "\n\n---\n\n" + new_text).strip()
        st.session_state.encrypted_transcript = encrypt_text(combined_text, fernet)
        st.session_state.last_transcribed_source = source

        st.success(f"Transcription complete, encrypted, and appended from {source}.")
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
