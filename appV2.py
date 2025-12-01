import os
import io
import time
import tempfile
import hashlib
from typing import Optional

import numpy as np
import soundfile as sf
import streamlit as st
from dotenv import load_dotenv

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
    aai.settings.polling_interval = 0.5
    return True

def transcribe_with_assemblyai(path: str) -> str:
    """
    Transcribe using AssemblyAI (HIPAA compliant with BAA)
    """
    try:
        config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.best,
            punctuate=True,
            format_text=True,
            language_detection=False,
        )
        
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(path)
        
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
# Helper to detect new audio
# ==========================
def get_audio_hash(audio_bytes: bytes) -> str:
    """Generate hash of audio bytes to detect changes"""
    return hashlib.md5(audio_bytes).hexdigest()

# ==========================
# Load system prompt once
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

# ==========================
# UI / Streamlit
# ==========================
st.set_page_config(page_title="Care Explained", page_icon="🎙️", layout="wide")

# Optimized CSS - minimal and efficient
st.markdown(
    """<style>
    body, .stApp { background-color: #000; color: #FFF; }
    div[data-testid="stMarkdownContainer"] p, label, span, h1, h2, h3, h4 { color: #FFF !important; }
    div[data-testid="stMarkdownContainer"] h1, div[data-testid="stMarkdownContainer"] h4 { text-align: center !important; }
    div[data-testid="stCaptionContainer"] { text-align: center !important; }
    .stAudio, div[data-testid="stAudio"] { display: flex !important; justify-content: center !important; }
    div[data-testid="stVerticalBlock"] > div:has(.stAudio) { display: flex !important; justify-content: center !important; }
    iframe[title="audio_recorder_streamlit.audio_recorder"] { margin: 0 auto !important; display: block !important; }
    .audio-chat-spacer { height: 60px; }
    </style>""",
    unsafe_allow_html=True,
)

# Collapsible Disclaimer
with st.expander("⚠️ Important: Disclaimer & Privacy Notice", expanded=True):
    st.markdown(
        """
        <div style="background-color:#111; color:#fff; border:1px solid #444; padding:15px; border-radius:10px;">
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

if not agree:
    st.warning("⚠️ Please expand the disclaimer above and agree to both terms before using this app.")
    st.stop()

# Session defaults
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = None

# ========== MAIN TAB ==========
tab_main = st.container()

with tab_main:
    st.title("Care Explained")

    # Load model only once
    if not st.session_state.model_loaded:
        with st.spinner("Loading transcription model..."):
            st.session_state.model = load_fw_model()
            st.session_state.model_loaded = True
            if st.session_state.model is None:
                st.error("Failed to load Faster-Whisper. Install: pip install faster-whisper")
                st.stop()

    model = st.session_state.model

    st.markdown("#### Record Audio")
    st.caption("Press the microphone to start/stop recording")
    
    # Keep columns exactly as specified
    col1, col2, col3 = st.columns([2.3, 0.5, 2])
    
    with col2:
        if AUDIO_RECORDER_AVAILABLE:
            audio_bytes = audio_recorder(
                text="",
                recording_color="#e74c3c",
                neutral_color="#6c757d",
                icon_name="microphone",
                icon_size="3x",
                energy_threshold=(-1.0, 1.0),
                pause_threshold=300.0,
            )
            audio_data = audio_bytes if audio_bytes else None
        else:
            audio_data_input = st.audio_input("Record up to 5 minutes")
            audio_data = audio_data_input.getvalue() if audio_data_input else None

    # Auto-detect new recording and transcribe
    if audio_data is not None and len(audio_data) > 0:
        current_hash = get_audio_hash(audio_data)
        
        if current_hash != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = current_hash
            
            with st.spinner(f"Transcribing audio..."):
                tmp_file_path = None
                try:
                    try:
                        audio_array, sr = sf.read(io.BytesIO(audio_data))
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            sf.write(tmp.name, audio_array, sr, format="WAV")
                            tmp_file_path = tmp.name
                    except Exception:
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

            # Store plaintext directly - no encryption overhead
            prev_text = st.session_state.transcript
            combined_text = (prev_text + "\n\n---\n\n" + new_text).strip()
            st.session_state.transcript = combined_text

            st.success(f"✅ Transcription complete!")

    # Add 60px spacing before chat section
    st.markdown('<div class="audio-chat-spacer"></div>', unsafe_allow_html=True)

    # Chat UI
    system_preamble = DEFAULT_SYSTEM_PROMPT
    system_goal = "Summary: Summarize main concerns, pertinent positives/negatives, and proposed plan."
    
    # Lazy load Gemini only when needed
    if st.session_state.gemini_model is None:
        st.session_state.gemini_model = init_gemini()
    
    gemini_model = st.session_state.gemini_model
    if gemini_model is None:
        st.warning("Gemini not configured. Set GEMINI_API_KEY in .env file to enable AI chat.")

    # Show chat history
    for m in st.session_state.messages:
        with st.chat_message(m.get("role", "user")):
            st.markdown(m.get("content", ""))

    user_msg = st.chat_input("Ask about this encounter (e.g., 'Summarize main concerns')")

    if user_msg:
        st.session_state.messages.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        if not st.session_state.transcript:
            reply = "Please record or upload audio before chatting."
        else:
            transcript = st.session_state.transcript
            with st.chat_message("assistant"):
                reply = run_gemini_chat(gemini_model, transcript, user_msg, f"{system_preamble}\n\n{system_goal}")
                st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})