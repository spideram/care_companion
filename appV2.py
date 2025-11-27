import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Upload Widget Comparison", page_icon="📁", layout="wide")

# Your app's dark theme
st.markdown(
    """<style>
    body, .stApp { background-color: #000000; color: #FFFFFF; }
    div[data-testid="stMarkdownContainer"] p, label, span, h1, h2, h3, h4 {
        color: #FFFFFF !important;
    }
    </style>""",
    unsafe_allow_html=True,
)

st.title("📁 File Upload Widget Options - Your App Style")
st.markdown("Compare different upload styles to match your microphone recorder")
st.markdown("---")

# Show the microphone for reference
st.markdown("### 🎙️ Your Current Recording Widget (For Reference)")
try:
    from audio_recorder_streamlit import audio_recorder
    
    st.markdown("""
        <style>
        .reference-recorder [data-testid="stVerticalBlock"] > div:has(audio-recorder) {
            border: 1px dashed rgba(250, 250, 250, 0.4);
            border-radius: 0.5rem;
            padding: 1.5rem;
            text-align: center;
            background-color: rgba(38, 39, 48, 0.4);
            min-height: 120px;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        audio-recorder {
            display: block;
            margin: 0 auto;
        }
        </style>
    """, unsafe_allow_html=True)
    
    ref_col1, ref_col2, ref_col3 = st.columns([1, 1, 1])
    with ref_col2:
        with st.container():
            audio_recorder(
                text="",
                recording_color="#e74c3c",
                neutral_color="#6c757d",
                icon_name="microphone",
                icon_size="3x",
                key="reference"
            )
            st.caption("This is what we want to match!")
except:
    st.info("Install audio-recorder-streamlit to see reference")

st.markdown("---")
st.markdown("### Now let's compare file upload options:")
st.markdown("---")

# ==================== OPTION A: Standard file_uploader ====================
st.markdown("## Option A: Standard st.file_uploader")
st.markdown("**The Traditional Drag & Drop Box**")

st.markdown("""
    <style>
    /* Style the standard file uploader */
    .optionA [data-testid="stFileUploader"] {
        border: 1px dashed rgba(250, 250, 250, 0.4);
        border-radius: 0.5rem;
        padding: 1.5rem;
        text-align: center;
        background-color: rgba(38, 39, 48, 0.4);
        min-height: 120px;
    }
    
    .optionA [data-testid="stFileUploader"]:hover {
        border-color: rgba(250, 250, 250, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

col_a1, col_a2, col_a3 = st.columns([1, 2, 1])
with col_a2:
    with st.container():
        st.markdown('<div class="optionA">', unsafe_allow_html=True)
        upload_a = st.file_uploader(
            "Drag and drop or browse files",
            type=["wav", "mp3", "m4a", "ogg", "webm", "flac"],
            key="upload_a",
            label_visibility="collapsed"
        )
        st.caption("Supported: WAV, MP3, M4A, OGG, WebM, FLAC")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("#### Pros & Cons:")
col_pros_a1, col_pros_a2 = st.columns(2)
with col_pros_a1:
    st.markdown("""
    ✅ **Native Streamlit** - Most reliable  
    ✅ **Drag & drop** - Better UX  
    ✅ **Shows file name** when selected  
    ✅ **Fully customizable** with CSS
    """)
with col_pros_a2:
    st.markdown("""
    ❌ **Doesn't match microphone** - Rectangle vs Icon  
    ❌ **Less symmetrical** - Different visual style  
    ❌ **More text-heavy** - Not as clean/minimal
    """)

st.markdown("---")

# ==================== OPTION B: Custom Icon Button ====================
st.markdown("## Option B: Custom File Icon Button ⭐")
st.markdown("**Icon-Based Design (Matches Microphone Perfectly!)**")

# Create custom file upload icon button
st.markdown("""
    <style>
    .custom-file-upload {
        border: 1px dashed rgba(250, 250, 250, 0.4);
        border-radius: 0.5rem;
        padding: 1.5rem;
        text-align: center;
        background-color: rgba(38, 39, 48, 0.4);
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        cursor: pointer;
        transition: all 0.3s;
        position: relative;
    }
    
    .custom-file-upload:hover {
        border-color: rgba(250, 250, 250, 0.6);
        background-color: rgba(38, 39, 48, 0.6);
    }
    
    .custom-file-upload input[type="file"] {
        position: absolute;
        width: 100%;
        height: 100%;
        opacity: 0;
        cursor: pointer;
        top: 0;
        left: 0;
    }
    
    .file-icon {
        font-size: 48px;
        color: #6c757d;
        margin-bottom: 10px;
    }
    
    .upload-text {
        font-size: 14px;
        color: rgba(250, 250, 250, 0.7);
    }
    
    /* Style for when file is selected */
    .file-selected {
        color: #28a745;
        margin-top: 10px;
        font-size: 14px;
    }
    </style>
    
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
""", unsafe_allow_html=True)

col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
with col_b2:
    # Custom HTML file upload button
    uploaded_file_b = components.html("""
        <div class="custom-file-upload" onclick="document.getElementById('fileInput').click()">
            <i class="fas fa-file-audio file-icon"></i>
            <input type="file" id="fileInput" accept="audio/*" onchange="handleFileSelect(this)">
            <div id="fileName" class="upload-text">Click to browse files</div>
        </div>
        
        <script>
        function handleFileSelect(input) {
            const fileName = document.getElementById('fileName');
            if (input.files.length > 0) {
                fileName.innerHTML = `<span class="file-selected">✓ ${input.files[0].name}</span>`;
                // Send file info back to Streamlit (if needed)
                window.parent.postMessage({type: 'streamlit:setComponentValue', value: input.files[0].name}, '*');
            }
        }
        </script>
    """, height=150)
    
    st.caption("Matches the microphone icon style!")

st.markdown("#### Pros & Cons:")
col_pros_b1, col_pros_b2 = st.columns(2)
with col_pros_b1:
    st.markdown("""
    ✅ **Perfect symmetry** - Matches microphone exactly  
    ✅ **Icon-based** - Same visual style  
    ✅ **Clean & minimal** - Professional look  
    ✅ **Same size/border** - Perfect alignment
    """)
with col_pros_b2:
    st.markdown("""
    ⚠️ **No drag & drop** - Click only  
    ⚠️ **Requires HTML/JS** - Slightly more complex  
    ⚠️ **File handling** - Need to integrate with Streamlit
    """)

st.markdown("---")

# ==================== OPTION C: Hybrid Approach ====================
st.markdown("## Option C: Hybrid - Icon + Small Upload Area")
st.markdown("**Best of Both Worlds?**")

st.markdown("""
    <style>
    .hybrid-upload {
        border: 1px dashed rgba(250, 250, 250, 0.4);
        border-radius: 0.5rem;
        padding: 1.5rem;
        text-align: center;
        background-color: rgba(38, 39, 48, 0.4);
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .hybrid-icon {
        font-size: 36px;
        color: #6c757d;
        margin-bottom: 15px;
    }
    </style>
    
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
""", unsafe_allow_html=True)

col_c1, col_c2, col_c3 = st.columns([1, 2, 1])
with col_c2:
    st.markdown('<div class="hybrid-upload">', unsafe_allow_html=True)
    st.markdown('<i class="fas fa-file-audio hybrid-icon"></i>', unsafe_allow_html=True)
    upload_c = st.file_uploader(
        "Drag & drop or click",
        type=["wav", "mp3", "m4a", "ogg", "webm", "flac"],
        key="upload_c",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("#### Pros & Cons:")
col_pros_c1, col_pros_c2 = st.columns(2)
with col_pros_c1:
    st.markdown("""
    ✅ **Icon present** - Visual consistency  
    ✅ **Drag & drop** - Maintains functionality  
    ✅ **Native Streamlit** - Reliable backend  
    ✅ **Good compromise** - Icon + upload area
    """)
with col_pros_c2:
    st.markdown("""
    ⚠️ **More cluttered** - Icon + text + upload UI  
    ⚠️ **Not as clean** - Less minimalist  
    ⚠️ **Taller height** - May not match perfectly
    """)

st.markdown("---")
st.markdown("---")

# ==================== SIDE-BY-SIDE COMPARISON ====================
st.markdown("## 🎨 Side-by-Side Comparison with Microphone")

comparison_col1, comparison_col2 = st.columns(2)

with comparison_col1:
    st.markdown("### 🎙️ Record Audio")
    st.markdown("""
        <div style="border: 1px dashed rgba(250, 250, 250, 0.4);
                    border-radius: 0.5rem;
                    padding: 1.5rem;
                    text-align: center;
                    background-color: rgba(38, 39, 48, 0.4);
                    min-height: 120px;
                    display: flex;
                    justify-content: center;
                    align-items: center;">
            <i class="fas fa-microphone" style="font-size: 48px; color: #6c757d;"></i>
        </div>
    """, unsafe_allow_html=True)
    st.caption("Click the microphone to start recording")

with comparison_col2:
    st.markdown("### 📁 Upload Audio File")
    
    option_choice = st.radio(
        "Preview which option?",
        ["Option A (Standard)", "Option B (Icon)", "Option C (Hybrid)"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if "Standard" in option_choice:
        st.markdown("""
            <div style="border: 1px dashed rgba(250, 250, 250, 0.4);
                        border-radius: 0.5rem;
                        padding: 1.5rem;
                        text-align: center;
                        background-color: rgba(38, 39, 48, 0.4);
                        min-height: 120px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;">
                <div style="font-size: 14px; color: rgba(250, 250, 250, 0.7);">
                    Drag and drop files here<br>
                    <small>Limit 200MB per file • WAV, MP3, M4A</small>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.caption("❌ Doesn't match - text-heavy, no icon")
        
    elif "Icon" in option_choice:
        st.markdown("""
            <div style="border: 1px dashed rgba(250, 250, 250, 0.4);
                        border-radius: 0.5rem;
                        padding: 1.5rem;
                        text-align: center;
                        background-color: rgba(38, 39, 48, 0.4);
                        min-height: 120px;
                        display: flex;
                        justify-content: center;
                        align-items: center;">
                <i class="fas fa-file-audio" style="font-size: 48px; color: #6c757d;"></i>
            </div>
        """, unsafe_allow_html=True)
        st.caption("✅ Perfect match - same style as microphone!")
        
    else:  # Hybrid
        st.markdown("""
            <div style="border: 1px dashed rgba(250, 250, 250, 0.4);
                        border-radius: 0.5rem;
                        padding: 1.5rem;
                        text-align: center;
                        background-color: rgba(38, 39, 48, 0.4);
                        min-height: 120px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                        align-items: center;">
                <i class="fas fa-file-audio" style="font-size: 36px; color: #6c757d; margin-bottom: 10px;"></i>
                <div style="font-size: 12px; color: rgba(250, 250, 250, 0.6);">
                    Drag & drop or click
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.caption("⚠️ Good compromise - icon + functionality")

st.markdown("---")

# ==================== FINAL RECOMMENDATION ====================
st.markdown("## 🏆 My Recommendation")

rec_col1, rec_col2, rec_col3 = st.columns([1, 3, 1])
with rec_col2:
    st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 30px; 
                    border-radius: 15px; 
                    text-align: center;">
            <h2 style="color: white; margin-top: 0;">Option B: Custom Icon Button ⭐</h2>
            <p style="color: white; font-size: 18px; line-height: 1.8;">
                <strong>Perfect visual symmetry with your microphone!</strong><br><br>
                
                Two identical boxes side-by-side:<br>
                🎙️ Microphone Icon = Record<br>
                📁 File Icon = Upload<br><br>
                
                Clean, modern, professional, and perfectly balanced.
            </p>
            <hr style="border-color: rgba(255,255,255,0.3); margin: 20px 0;">
            <p style="color: white; font-size: 16px;">
                <strong>Alternative:</strong> If you really need drag & drop, 
                go with Option C (Hybrid), but Option B gives you the cleanest, 
                most professional look that perfectly matches your recording widget.
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.success("**Ready to implement?** Let me know which option you want and I'll integrate it into your main app code!")