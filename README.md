# Care Companion
Care Companion is a Streamlit-based prototype tool designed to enhance patient-provider
communication by transcribing conversations and providing context-aware, privacy-compliant
responses.
## Features
- Real-time audio recording and transcription (using Whisper)
- Secure, HIPAA-compliant handling of patient data
- Context-aware responses with AI (Gemini API)
- End-to-end encryption support
- Simple and intuitive Streamlit interface
## Installation
1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/care_companion.git
cd care_companion
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file with your API key:
```bash
GEMINI_API_KEY=your_api_key_here
```
4. Run the app:
```bash
streamlit run app.py
```
## Development Notes
- Uses `python-dotenv` to load environment variables
- `.gitignore` excludes sensitive files like `.env` and local cache
- You can edit `templates/system_prompt.txt` to adjust model behavior
## License
This project is licensed under the MIT License - see the LICENSE file for details.