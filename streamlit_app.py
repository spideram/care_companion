import os
import io
import time
import tempfile
import gc
import hashlib
from typing import Optional

import numpy as np
import soundfile as sf
import streamlit as st
from dotenv import load_dotenv
from cryptography.fernet import Fernet

# Audio recorder component (faster alternative to st.audio_input)
try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except Exception:
    audio_recorder = None
    AUDIO_RECORDER_AVAILABLE = False

# AssemblyAI import (preferred - HIPAA compliant)
try:
    import assemblyai as aai
    ASSEMBLYAI_AVAILABLE = True
except Exception:
    aai = None
    ASSEMBLYAI_AVAILABLE = False

# Faster-Whisper import (fallback)
try:
    from faster_whisper import WhisperModel
    FASTER_AVAILABLE = True
except Exception:
    WhisperModel = None
    FASTER_AVAILABLE = False

# Optional Gemini import
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    genai = None
    GEMINI_AVAILABLE = False

# Load env
load_dotenv()

# ==========================
# AssemblyAI setup
# ==========================
def init_assemblyai():
    """Initialize AssemblyAI with API key"""
    if not ASSEMBLYAI_AVAILABLE:
        return False
    
    api_key = os.getenv("ASSEMBLYAI_API_KEY") or st.secrets.get("ASSEMBLYAI_API_KEY", None)
    if not api_key:
        return False
    
    aai.settings.api_key = api_key
    # Speed up polling - check status every 0.5 seconds instead of 3 seconds
    aai.settings.polling_interval = 0.5
    return True

def transcribe_with_assemblyai(path: str) -> str:
    """
    Transcribe using AssemblyAI (HIPAA compliant with BAA)
    Includes automatic punctuation, capitalization, and medical terminology support
    """
    try:
        # Configure transcription with medical-optimized settings
        config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.best,  # Use the best/latest model (Universal-1)
            punctuate=True,
            format_text=True,
            language_detection=False,  # Set to English for medical transcription
        )
        
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(path)
        
        # Check for errors
        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")
        
        return transcript.text.strip()
    except Exception as e:
        raise RuntimeError(f"AssemblyAI transcription error: {e}")

# ==========================
# Faster-Whisper fallback
# ==========================
@st.cache_resource(show_spinner=False)
def load_fw_model():
    """Load Faster-Whisper model (fallback)"""
    if not FASTER_AVAILABLE:
        return None
    try:
        return WhisperModel("base", device="cpu", compute_type="int8")
    except Exception:
        try:
            return WhisperModel("base", device="cpu")
        except Exception:
            return None

def transcribe_with_faster_whisper(path: str, fw_model) -> str:
    """Transcribe using Faster-Whisper (fallback)"""
    if fw_model is None:
        raise RuntimeError("Faster-Whisper model not available.")
    
    segments, info = fw_model.transcribe(path)
    texts = [seg.text for seg in segments if getattr(seg, "text", None)]
    return " ".join(texts).strip()

# ==========================
# Gemini helpers
# ==========================
def init_gemini() -> Optional[object]:
    if not GEMINI_AVAILABLE:
        return None
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    try:
        return genai.GenerativeModel("gemini-2.5-flash")
    except Exception:
        try:
            return genai.GenerativeModel()
        except Exception:
            return None

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

# ==========================
# Encryption helpers (Fernet)
# ==========================
def get_session_fernet() -> Fernet:
    if "fernet_key" not in st.session_state:
        st.session_state.fernet_key = Fernet.generate_key()
    return Fernet(st.session_state.fernet_key)

def encrypt_text(plain_text: str, fernet: Fernet) -> bytes:
    return fernet.encrypt(plain_text.encode("utf-8"))

def decrypt_text(blob: bytes, fernet: Fernet) -> str:
    return fernet.decrypt(blob).decode("utf-8", errors="ignore")

# ==========================
# Helper to detect new audio
# ==========================
def get_audio_hash(audio_bytes: bytes) -> str:
    """Generate hash of audio bytes to detect changes"""
    return hashlib.md5(audio_bytes).hexdigest()

# ==========================
# Prompts & templates
# ==========================
PROMPT_FILE = "prompts/prompt_start.txt"
if os.path.exists(PROMPT_FILE):
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        DEFAULT_SYSTEM_PROMPT = f.read()
else:
    DEFAULT_SYSTEM_PROMPT = (
        "You are a clinical assistant that extracts practical, provider-facing insights from a patient encounter. "
        "Stay neutral, avoid diagnosis unless explicitly requested, and highlight uncertainties. "
        "Emphasize red flags, medication changes, follow-ups, and patient education points."
    )

# TEMPLATES = {
#     "Summary": "Summarize main concerns, pertinent positives/negatives, and proposed plan.",
#     "SOAP": "Produce a SOAP-style summary (Subjective, Objective, Assessment, Plan).",
#     "Follow-ups": "List 3–5 follow-up questions to clarify key uncertainties.",
#     "Patient education": "Draft a plain-language explanation and next steps for the patient.",
# }

# ==========================
# UI / Streamlit
# ==========================
st.set_page_config(page_title="Care Explained", page_icon="🎙️", layout="wide")

# Dark background style
st.markdown(
    """<style>
    body, .stApp { background-color: #000000; color: #FFFFFF; }
    div[data-testid="stMarkdownContainer"] p, label, span, h1, h2, h3, h4 {
        color: #FFFFFF !important;
    }
    </style>""",
    unsafe_allow_html=True,
)

# Collapsible Disclaimer
with st.expander("⚠️ Important: Disclaimer & Privacy Notice", expanded=True):
    st.markdown(
        """
        <div style="background-color:#111111; color:#ffffff; border:1px solid #444; padding:15px; border-radius:10px;">
            <strong>⚠️ Disclaimer:</strong><br>
            This application is currently under active development and the intended use is to help clarify medical terminology and care to users.
            <br><br>
            <ul style="margin-left:15px;">
                <li><b>Data privacy:</b> All audio and text processed during this session are handled temporarily in memory and protected using end-to-end encryption. No recordings or transcripts are stored, transmitted externally, or retained after the session ends.</li>
                <li><b>Operational status:</b> While we are implementing the required safeguards for protected health information, this application is not yet certified as HIPAA-compliant.</li>
            </ul>
            <em>By continuing, you acknowledge that you understand these limitations and agree to use this prototype solely for evaluation and non-production purposes.</em>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    agree = st.checkbox("✅ I understand and agree to the recording disclaimer above.")
    # hipaa_ack = st.checkbox("✅ I acknowledge HIPAA compliance requirements and will obtain a BAA if processing PHI.")

# Check agreement status (outside expander so it doesn't force expansion)
if "agree" not in locals() or not agree: #or not hipaa_ack
    st.warning("⚠️ Please expand the disclaimer above and agree to both terms before using this app.")
    st.stop()

# Tabs
tab_main = st.container()

# Session defaults
if "encrypted_transcript" not in st.session_state:
    st.session_state.encrypted_transcript = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None

# ========== MAIN TAB ==========
with tab_main:
    st.title("Care Explained")

    # Load Faster-Whisper model (for prototype)
    with st.spinner("Loading transcription model..."):
        model = load_fw_model()
        model_type = "faster_whisper"
        if model is None:
            st.error("Failed to load Faster-Whisper. Install: pip install faster-whisper")
            st.stop()

    st.markdown("#### Record Audio")
    st.caption("Press the microphone to start/stop recording")
    
    # Use audio-recorder-streamlit if available (much faster)
    if AUDIO_RECORDER_AVAILABLE:
        audio_bytes = audio_recorder(
            text="",  # No text, just icon
            recording_color="#e74c3c",
            neutral_color="#6c757d",
            icon_name="microphone",
            icon_size="3x",
            # Fix: Disable auto-stop on silence for medical recordings
            # Set energy_threshold to allow continuous recording
            energy_threshold=(-1.0, 1.0),  # Always consider as "speaking"
            pause_threshold=300.0,  # Allow up to 5 minutes of recording
        )
        
        # Convert to format compatible with rest of code
        audio_data = audio_bytes if audio_bytes else None
    else:
        audio_data_input = st.audio_input("Record up to 5 minutes")
        audio_data = audio_data_input.getvalue() if audio_data_input else None

    # ----------------------------
    # Auto-detect new recording and transcribe
    # ----------------------------
    if audio_data is not None:
        current_hash = get_audio_hash(audio_data)
        
        # Check if this is a new recording
        if current_hash != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = current_hash
            
            with st.spinner(f"Transcribing audio..."):
                tmp_file_path = None
                try:
                    # Convert to WAV for Faster-Whisper
                    try:
                        audio_array, sr = sf.read(io.BytesIO(audio_data))
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            sf.write(tmp.name, audio_array, sr, format="WAV")
                            tmp_file_path = tmp.name
                    except Exception:
                        # If soundfile fails, try writing raw bytes
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(audio_data)
                            tmp_file_path = tmp.name
                    
                    new_text = transcribe_with_faster_whisper(tmp_file_path, model)

                    if not new_text:
                        st.warning("Transcription returned empty. Check audio quality or format.")
                        new_text = "[No speech detected in audio]"

                except Exception as e:
                    st.error(f"Transcription error: {e}")
                    new_text = f"[Transcription failed: {e}]"
                finally:
                    if tmp_file_path and os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)

            # Encrypt and append transcript
            fernet = get_session_fernet()
            prev_text = ""
            if st.session_state.get("encrypted_transcript"):
                try:
                    prev_text = decrypt_text(st.session_state.encrypted_transcript, fernet)
                except Exception:
                    pass

            combined_text = (prev_text + "\n\n---\n\n" + new_text).strip()
            st.session_state.encrypted_transcript = encrypt_text(combined_text, fernet)

            st.success(f"✅ Transcription complete!")
            gc.collect()
            st.rerun()


    # Chat UI
    # template = st.selectbox("Prompt template", list(TEMPLATES.keys()), index=0)
    system_preamble = DEFAULT_SYSTEM_PROMPT
    system_goal = "Summary: Summarize main concerns, pertinent positives/negatives, and proposed plan."
    # system_goal = f"Template: {template}\n\n{TEMPLATES[template]}"
    
    # Initialize Gemini
    gemini_model = init_gemini()
    if gemini_model is None:
        st.warning("Gemini not configured. Set GEMINI_API_KEY in .env file to enable AI chat.")

    # show chat history
    for m in st.session_state.messages:
        with st.chat_message(m.get("role", "user")):
            st.markdown(m.get("content", ""))

    user_msg = st.chat_input("Ask about this encounter (e.g., 'Summarize main concerns')")

    if user_msg:
        st.session_state.messages.append({ "role": "user", "content": user_msg })
        with st.chat_message("user"):
            st.markdown(user_msg)

        if st.session_state.encrypted_transcript is None:
            reply = "Please record or upload audio before chatting."
        else:
            f = get_session_fernet()
            try:
                transcript = decrypt_text(st.session_state.encrypted_transcript, f)
            except Exception:
                transcript = "[Decryption failed]"
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    reply = run_gemini_chat(gemini_model, transcript, user_msg, f"{system_preamble}\n\n{system_goal}")
                    st.markdown(reply)

        st.session_state.messages.append({ "role": "assistant", "content": reply })
        gc.collect()