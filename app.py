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

import io
import os
import subprocess
import numpy as np
import imageio_ffmpeg
import whisper.audio

# Get the full path to ffmpeg.exe (or the right binary for Linux/macOS)
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
print("Using ffmpeg at:", ffmpeg_path)

# Monkey-patch Whisper's audio loader so it always uses our ffmpeg binary
def load_audio_with_ffmpeg(path: str):
    cmd = [
        ffmpeg_path,  # use the full path to ffmpeg
        "-nostdin",
        "-threads", "0",
        "-i", path,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(whisper.audio.SAMPLE_RATE),
        "-"
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

# Override Whisper's default function
whisper.audio.load_audio = load_audio_with_ffmpeg

import tempfile
from typing import List, Dict, Optional
from dotenv import load_dotenv

import streamlit as st

# --- Optional: Encryption ---
try:
    from cryptography.fernet import Fernet
except Exception:
    Fernet = None  # Encryption toggle will be disabled if missing

# --- Optional: Gemini ---
GEMINI_AVAILABLE = False
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    pass

# --- Speech to text (Whisper) ---
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

@st.cache_resource(show_spinner=False)
def load_whisper(model_size: str = "base"):
    import whisper  # lazy import so app opens even if not installed yet
    return whisper.load_model(model_size)


def transcribe_audio(tmp_path: str, model_size: str = "base") -> str:
    model = load_whisper(model_size)
    result = model.transcribe(tmp_path)
    return result.get("text", "").strip()


# --- Gemini helper ---
# Load environment variables from .env
load_dotenv()
def init_gemini() -> Optional[object]:
    if not GEMINI_AVAILABLE:
        return None
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    try:
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception:
        return None


def run_gemini_chat(model, transcript: str, user_prompt: str, system_preamble: str) -> str:
    """Simple wrapper; if model is None, return a placeholder."""
    if model is None:
        return "[Gemini not configured] Provide GEMINI_API_KEY to enable AI responses.\n\nStub answer: Based on the transcript, consider summarizing key symptoms, medications, and next steps.]"

    prompt = f"""
    {system_preamble}

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


# --- Minimal prompt engineering templates ---

with open("prompts/prompt_start.txt", "r", encoding="utf-8") as f:
    DEFAULT_SYSTEM_PROMPT = f.read()


# DEFAULT_SYSTEM_PROMPT = (
#     "You are a clinical assistant that extracts practical, provider-facing insights from a patient encounter. "
#     "Stay neutral, avoid diagnosis unless explicitly requested, and highlight uncertainties. "
#     "Emphasize red flags, medication changes, follow-ups, and patient education points."
# )

TEMPLATES = {
    "Summary": "Summarize main concerns, pertinent positives/negatives, and proposed plan.",
    "SOAP": "Produce a SOAP-style summary (Subjective, Objective, Assessment, Plan).",
    "Follow-ups": "List 3–5 follow-up questions to clarify key uncertainties.",
    "Patient education": "Draft a plain-language explanation and next steps for the patient.",
}


# --- Encryption helpers ---
def get_fernet(key_b64: Optional[str]) -> Optional[Fernet]: # type: ignore
    if Fernet is None:
        return None
    try:
        if key_b64:
            return Fernet(key_b64)
        # Generate per-session key if missing
        if "fernet_key" not in st.session_state:
            st.session_state.fernet_key = Fernet.generate_key()
        return Fernet(st.session_state.fernet_key)
    except Exception:
        return None


def maybe_encrypt_text(text: str, enabled: bool, f: Optional[Fernet]) -> bytes: # type: ignore
    if enabled and f is not None:
        return f.encrypt(text.encode("utf-8"))
    return text.encode("utf-8")


def maybe_decrypt_text(blob: bytes, enabled: bool, f: Optional[Fernet]) -> str: # type: ignore
    if enabled and f is not None:
        try:
            return f.decrypt(blob).decode("utf-8", errors="ignore")
        except Exception:
            return "[Decryption failed]"
    return blob.decode("utf-8", errors="ignore")


# ===================== UI =====================
st.set_page_config(page_title="Provider Comms MVP", page_icon="🎙️", layout="wide")
st.title("🎙️ Provider Communication Assistant — MVP")
st.caption("Prototype: microphone → transcript → focused clinical chat")

with st.sidebar:
    st.header("Settings")

    st.subheader("Speech-to-Text")
    whisper_size = st.selectbox(
        "Whisper model",
        options=["tiny", "base", "small", "medium"],
        index=1,
        help="Smaller = faster, lower accuracy. 'base' is a good MVP default.",
    )

    st.subheader("AI Backend")
    use_gemini = st.toggle("Use Gemini for chat", value=True, help="Disable to use a stub reply.")

    st.subheader("Security")
    enc_enabled = st.toggle(
        "Demo encryption (Fernet)",
        value=bool(Fernet),
        help="Encrypt transcript before sending to AI. For demo only; not production E2EE.",
    )
    custom_key = st.text_input(
        "Optional Fernet key (base64)",
        value="",
        type="password",
        help="Leave blank to use a per-session random key.",
    )

    if enc_enabled and Fernet is None:
        st.info("Install 'cryptography' to enable encryption: pip install cryptography")

# Session state for transcript & chat
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "messages" not in st.session_state:
    st.session_state.messages = []  # type List[Dict[str, str]]

# Tabs for flow
rec_tab, chat_tab = st.tabs(["1) Capture & Transcribe", "2) Focused Clinical Chat"])

with rec_tab:
    st.subheader("Capture audio")
    st.write(
        "Record directly in the browser or upload a .wav/.mp3. For longer recordings, uploading a file is more reliable."
    )

    # Prefer native mic input if available
    audio_data = st.audio_input("Record up to ~60 seconds")
    uploaded = st.file_uploader("…or upload an audio file", type=["wav", "mp3", "m4a", "ogg", "webm"])  # ffmpeg recommended

    # Choose source priority: uploaded > mic (if both present)
    source_label = None
    raw_bytes: Optional[bytes] = None

    if uploaded is not None:
        raw_bytes = uploaded.read()
        source_label = f"Uploaded file: {uploaded.name}"
    elif audio_data is not None:
        # audio_input returns a BytesIO-like object
        raw_bytes = audio_data.getvalue() if hasattr(audio_data, "getvalue") else bytes(audio_data)
        source_label = "Browser recording"

    st.write(
        "**Selected source:** ", source_label if source_label else "None yet"
    )

    if st.button("Transcribe", type="primary", disabled=(raw_bytes is None)):
        if raw_bytes is None:
            st.warning("Please record or upload audio first.")
        else:
            with st.spinner("Transcribing with Whisper…"):
                # Save to a temp file; Whisper uses ffmpeg under the hood, so any common type is fine if ffmpeg is present
                suffix = ".wav" if (uploaded and uploaded.name.lower().endswith(".wav")) else ".mp3"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(raw_bytes)
                    tmp_path = tmp.name
                try:
                    text = transcribe_audio(tmp_path, model_size=whisper_size)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

            st.session_state.transcript = text
            st.success("Transcription complete.")

    st.text_area("Transcript (editable)", key="transcript", height=240)

    st.info(
        "Tip: You can edit the transcript before moving to the chat tab. This helps correct any ASR errors."
    )

with chat_tab:
    st.subheader("Focused clinical chat over the transcript")

    # Prompt template selection
    template = st.selectbox("Prompt template", list(TEMPLATES.keys()), index=0)
    system_preamble = st.text_area("System preamble (prompt engineering)", value=DEFAULT_SYSTEM_PROMPT, height=120)

    # Show current transcript for context
    with st.expander("Show transcript context"):
        st.write(st.session_state.transcript or "(No transcript yet)")

    # Initialize Gemini (once per session) if requested
    gemini_model = init_gemini() if use_gemini else None
    if use_gemini and gemini_model is None:
        st.warning("Gemini not configured or library missing. Set GEMINI_API_KEY and install google-generativeai.")

    # Chat history display
    for m in st.session_state.messages:
        with st.chat_message(m.get("role", "user")):
            st.markdown(m.get("content", ""))

    user_msg = st.chat_input("Ask something about this encounter (e.g., 'Summarize main concerns')")

    if user_msg:
        st.session_state.messages.append({"role": "user", "content": user_msg})

        # Encrypt transcript payload (demo)
        f = get_fernet(custom_key if custom_key else None) if enc_enabled else None
        encrypted_payload = maybe_encrypt_text(st.session_state.transcript, enc_enabled, f)

        # "Send" to AI and decrypt on the other side (demo only)
        decrypted_for_ai = maybe_decrypt_text(encrypted_payload, enc_enabled, f)

        system_goal = f"Template: {template}\n\n{TEMPLATES[template]}"
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer = run_gemini_chat(
                    gemini_model if use_gemini else None,
                    transcript=decrypted_for_ai,
                    user_prompt=user_msg,
                    system_preamble=f"{system_preamble}\n\n{system_goal}",
                )
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


# Footer
st.divider()
st.caption(
    "MVP prototype • Whisper for ASR • Optional Gemini for LLM • Demo encryption with Fernet.\n"
    "Next steps: Parakeet ASR backend, streaming transcripts, role-based access, audit logs, real E2EE, and BAA-backed hosting."
)
