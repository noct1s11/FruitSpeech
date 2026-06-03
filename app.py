import streamlit as st
import os
import io
import time
import pickle
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Import functions from project
from utils.tts import generate_tts_audio, speak
from train_model import train, MODEL_PATH, DATASET_DIR
from predict import predict

# Available fruit classes
FRUITS = [
    "apel", "jeruk", "mangga", "pisang", "semangka", 
    "melon", "anggur", "pepaya", "nanas", "stroberi"
]

# Set page configuration
st.set_page_config(
    page_title="FruitSpeech - Enterprise AI Dashboard",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# In-memory WAV generator
def get_audio_bytes(y, sr):
    virtual_file = io.BytesIO()
    sf.write(virtual_file, y, sr, format='wav', subtype='PCM_16')
    virtual_file.seek(0)
    return virtual_file.read()

# Streamlit recording helper with visual waveform feedback
def record_audio_streamlit(filename, duration=2.0, sample_rate=16000):
    status_area = st.empty()
    waveform_area = st.empty()
    
    status_area.info("⏳ Bersiaplah... Perekaman akan dimulai dalam 1 detik.")
    time.sleep(1.0)
    
    # Custom HTML/CSS for audio recording waveform feedback (Dark Theme)
    recording_html = """
    <div style='text-align: center; padding: 24px; background: rgba(30, 36, 62, 0.6); border: 1.5px solid rgba(255, 107, 107, 0.2); border-radius: 20px; margin-bottom: 20px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2); backdrop-filter: blur(12px);'>
        <div style='display: flex; justify-content: center; align-items: center; gap: 8px; margin-bottom: 15px;'>
            <span style='width: 12px; height: 12px; background-color: #FF6B6B; border-radius: 50%; display: inline-block; animation: blink 1s infinite;'></span>
            <span style='font-weight: 700; color: #FFFFFF; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;'>MEREKAM... SILAKAN UCAPKAN KATA SEKARANG!</span>
        </div>
        <div style='display: flex; justify-content: center; align-items: flex-end; height: 40px; gap: 4px;'>
            <div style='width: 4px; background: #FF6B6B; border-radius: 2px; animation: wave 0.8s ease-in-out infinite alternate; height: 15px;'></div>
            <div style='width: 4px; background: #FF8E53; border-radius: 2px; animation: wave 1.2s ease-in-out infinite alternate; height: 35px;'></div>
            <div style='width: 4px; background: #FF6B6B; border-radius: 2px; animation: wave 0.7s ease-in-out infinite alternate; height: 20px;'></div>
            <div style='width: 4px; background: #FF8E53; border-radius: 2px; animation: wave 1.0s ease-in-out infinite alternate; height: 40px;'></div>
            <div style='width: 4px; background: #FF6B6B; border-radius: 2px; animation: wave 0.9s ease-in-out infinite alternate; height: 25px;'></div>
            <div style='width: 4px; background: #FF8E53; border-radius: 2px; animation: wave 1.1s ease-in-out infinite alternate; height: 30px;'></div>
            <div style='width: 4px; background: #FF6B6B; border-radius: 2px; animation: wave 0.8s ease-in-out infinite alternate; height: 15px;'></div>
        </div>
        <style>
            @keyframes blink {
                0% { opacity: 0.2; }
                50% { opacity: 1; }
                100% { opacity: 0.2; }
            }
            @keyframes wave {
                0% { height: 10px; }
                100% { height: 45px; }
            }
        </style>
    </div>
    """
    
    waveform_area.markdown(recording_html, unsafe_allow_html=True)
    
    # Record audio
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    
    # Simple simulated progress bar
    progress_bar = st.progress(0.0)
    for i in range(100):
        time.sleep(duration / 100.0)
        progress_bar.progress((i + 1) / 100.0)
        
    sd.wait()
    
    waveform_area.empty()
    progress_bar.empty()
    
    # Diagnostic check for silence (macOS permission block check)
    max_amp = float(np.max(np.abs(recording)))
    if max_amp == 0.0:
        status_area.error("⚠️ **Peringatan**: Mikrofon merekam keheningan total (amplitudo 0.0). Pastikan izin akses mikrofon telah diaktifkan untuk Terminal/IDE Anda di Pengaturan Privasi macOS (System Settings > Privacy & Security > Microphone).")
    else:
        status_area.success(f"✅ Perekaman selesai! (Amplitudo Maks: {max_amp:.4f})")
    
    # Ensure directory exists and write
    dir_name = os.path.dirname(filename)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    sf.write(filename, recording, sample_rate, subtype='PCM_16')
    return recording

# Helper to get dataset file count
def get_dataset_stats():
    stats = {}
    if not os.path.exists(DATASET_DIR):
        return {f: 0 for f in FRUITS}
    for fruit in FRUITS:
        path = os.path.join(DATASET_DIR, fruit)
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith(".wav")]
            stats[fruit] = len(files)
        else:
            stats[fruit] = 0
    return stats

# Helper to load model metadata
def load_model_metadata():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model_data = pickle.load(f)
            if isinstance(model_data, dict):
                return {
                    'model_type': model_data.get('model_type', 'SVM'),
                    'accuracy': model_data.get('accuracy', None),
                    'num_samples': model_data.get('num_samples', 0),
                    'trained_at': model_data.get('trained_at', 'N/A'),
                    'report': model_data.get('report', None),
                    'confusion_matrix': model_data.get('confusion_matrix', None),
                    'classes': model_data.get('classes', [])
                }
        except Exception as e:
            pass
    return {
        'model_type': 'N/A',
        'accuracy': None,
        'num_samples': 0,
        'trained_at': 'N/A',
        'report': None,
        'confusion_matrix': None,
        'classes': []
    }

# Inject background style dynamically (Cosmic Sunset Dark Theme - Presentation Friendly)
import base64
bg_image_path = "/Users/noct1s11/.gemini/antigravity-ide/brain/d2ff72b6-a40b-40aa-8a74-b95e6d9400e9/media__1780389981479.jpg"
if os.path.exists(bg_image_path):
    with open(bg_image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(180deg, rgba(10, 12, 22, 0.88) 0%, rgba(15, 18, 36, 0.92) 50%, rgba(6, 8, 15, 0.96) 100%), url(data:image/jpeg;base64,{encoded_image}) no-repeat center center fixed !important;
        background-size: cover !important;
        backdrop-filter: blur(8px) !important;
    }}
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp {
        background-color: #0A0C16 !important;
        background: linear-gradient(180deg, #0A0C16 0%, #0F1224 50%, #05060A 100%) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Inject Custom CSS for visual excellence (Enterprise Dark Mode AI Dashboard)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Font style and layout */
    html, body, [data-testid="stSidebar"], .stMarkdown {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Hide Streamlit default UI components for a clean SaaS app look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {
        background-color: transparent !important;
        border: none !important;
    }
    [data-testid="stHeader"] button {
        color: #FF6B6B !important;
    }
    
    /* Sidebar styling - Dark Slate */
    [data-testid="stSidebar"] {
        width: 280px !important;
        background: #0D0F1D !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    
    /* Custom vertical sidebar navigation cards */
    [data-testid="stSidebar"] div[role="radiogroup"] {
        background-color: transparent !important;
    }
    
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        display: flex !important;
        align-items: center !important;
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 14px 20px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        cursor: pointer !important;
        width: 100% !important;
    }
    
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        transform: translateX(4px) !important;
        border-color: rgba(255, 107, 107, 0.4) !important;
        box-shadow: 0 8px 16px rgba(255, 107, 107, 0.1) !important;
    }
    
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%) !important;
        border-color: transparent !important;
        box-shadow: 0 8px 20px rgba(255, 107, 107, 0.25) !important;
    }
    
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) * {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stSidebar"] div[role="radiogroup"] label input {
        display: none !important;
    }
    
    [data-testid="stSidebar"] div[role="radiogroup"] label [data-testid="stFiberManualRecord"] {
        display: none !important;
    }
    
    /* Layout main container styling */
    .block-container {
        max-width: 1400px !important;
        padding-top: 40px !important;
        padding-bottom: 40px !important;
        margin: auto !important;
    }
    
    /* Typography style overrides - SaaS dark background adjustments (Presentation Friendly) */
    h1 {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        letter-spacing: -1px !important;
        line-height: 1.2 !important;
    }
    h2 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
        line-height: 1.3 !important;
    }
    
    h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    p, span, label {
        color: #DFE6E9 !important;
    }

    /* Make markdown paragraphs outside cards white/light grey */
    [data-testid="stMarkdownContainer"] p {
        color: #DFE6E9 !important;
        line-height: 1.6 !important;
    }
    
    /* Premium Dashboard Glassmorphic Card (Dark Theme) */
    .dashboard-card {
        background: rgba(22, 28, 54, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 20px !important;
        padding: 24px !important;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3) !important;
        margin-bottom: 24px;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    .dashboard-card:hover {
        transform: translateY(-3px) !important;
        border-color: rgba(255, 107, 107, 0.3) !important;
        box-shadow: 0 15px 35px rgba(255, 107, 107, 0.08) !important;
    }
    
    /* Global Stat cards styling */
    .stat-card {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 20px !important;
        padding: 24px !important;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.25) !important;
        display: flex !important;
        align-items: center !important;
        gap: 20px !important;
        transition: all 0.3s ease !important;
        margin-bottom: 24px !important;
        width: 100% !important;
    }
    .stat-card:hover {
        transform: translateY(-3px) !important;
        border-color: rgba(255, 107, 107, 0.3) !important;
        box-shadow: 0 15px 35px rgba(255, 107, 107, 0.06) !important;
    }
    .stat-icon {
        font-size: 36px !important;
        background: rgba(255, 255, 255, 0.05) !important;
        width: 68px !important;
        height: 68px !important;
        border-radius: 16px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    .stat-info {
        display: flex !important;
        flex-direction: column !important;
    }
    .stat-label {
        font-size: 12px !important;
        color: #A0AEC0 !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
    }
    .stat-value {
        font-size: 26px !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        margin-top: 4px !important;
    }
    
    /* Result card styling (ASR Page) */
    .result-card {
        background: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 20px !important;
        padding: 28px !important;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.25) !important;
        display: flex !important;
        align-items: center !important;
        gap: 28px !important;
        margin-bottom: 24px !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .result-card:hover {
        transform: translateY(-3px) !important;
        border-color: rgba(255, 107, 107, 0.3) !important;
    }
    .result-emoji-container {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        width: 96px !important;
        height: 96px !important;
        border-radius: 24px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .result-emoji {
        font-size: 64px !important;
    }
    .result-details {
        display: flex !important;
        flex-direction: column !important;
    }
    .result-label {
        font-size: 13px !important;
        color: #A0AEC0 !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
    }
    .result-value {
        font-size: 44px !important;
        color: #FFFFFF !important;
        font-weight: 900 !important;
        margin-top: 4px !important;
        letter-spacing: -0.5px !important;
    }
    .result-badge {
        color: white !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        padding: 4px 12px !important;
        display: inline-block !important;
        width: fit-content !important;
        margin-top: 8px !important;
    }
    
    /* Dataset cards */
    .fruit-dataset-card {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 20px !important;
        padding: 20px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.3s ease !important;
        margin-bottom: 16px !important;
    }
    .fruit-dataset-card:hover {
        transform: translateY(-3px) !important;
        border-color: rgba(255, 107, 107, 0.3) !important;
    }
    .dataset-progress-bar-bg {
        background-color: rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        height: 8px !important;
        width: 100% !important;
        overflow: hidden !important;
        border: none !important;
    }
    .dataset-progress-bar-fg {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%) !important;
        height: 100% !important;
        border-radius: 10px !important;
    }
    
    /* Model Training page metrics card grid */
    .training-info-grid {
        display: grid !important;
        grid-template-columns: repeat(4, 1fr) !important;
        gap: 16px !important;
        margin-bottom: 24px !important;
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 20px !important;
        padding: 20px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2) !important;
    }
    .training-info-item {
        display: flex !important;
        flex-direction: column !important;
    }
    .training-info-label {
        font-size: 11px !important;
        color: #A0AEC0 !important;
        text-transform: uppercase !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
    }
    .training-info-value {
        font-size: 16px !important;
        color: #FFFFFF !important;
        font-weight: 750 !important;
        margin-top: 4px !important;
    }

    /* Customize Streamlit primary button (Large CTA) */
    div.stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%) !important;
        color: #ffffff !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 16px !important;
        height: 58px !important;
        width: 100% !important;
        box-shadow: 0 8px 24px rgba(255, 107, 107, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        letter-spacing: 0.5px !important;
    }
    
    div.stButton > button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #FF8E53 0%, #FF6B6B 100%) !important;
        box-shadow: 0 12px 30px rgba(255, 107, 107, 0.55) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Secondary buttons */
    div.stButton > button[data-testid="baseButton-secondary"] {
        background-color: rgba(255, 255, 255, 0.02) !important;
        color: #FF6B6B !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        border: 2px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        height: 54px !important;
        width: 100% !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div.stButton > button[data-testid="baseButton-secondary"]:hover {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-color: #FF6B6B !important;
        transform: translateY(-2px) !important;
    }

    /* Text Inputs and selectboxes styling (Dark Theme) */
    .stTextArea textarea {
        background-color: rgba(10, 12, 22, 0.6) !important;
        border: 1.5px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        color: #FFFFFF !important;
        font-size: 15px !important;
        padding: 18px !important;
    }
    .stTextArea textarea:focus {
        border-color: #FF6B6B !important;
        box-shadow: 0 0 0 4px rgba(255, 107, 107, 0.15) !important;
    }
    
    div[data-baseweb="select"] {
        background-color: rgba(10, 12, 22, 0.6) !important;
        border-radius: 16px !important;
        border: 1.5px solid rgba(255, 255, 255, 0.1) !important;
        color: #FFFFFF !important;
    }
    div[data-baseweb="select"]:hover {
        border-color: #FF6B6B !important;
    }
    div[data-baseweb="select"] * {
        color: #FFFFFF !important;
    }
    
    /* Dropdown list/menu styling (Presentation Friendly) */
    div[data-baseweb="popover"] ul,
    div[role="listbox"],
    ul[role="listbox"],
    [data-baseweb="menu"] {
        background-color: #161C36 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
    }
    
    div[data-baseweb="popover"] li,
    ul[role="listbox"] li,
    div[role="option"],
    [data-baseweb="menu"] [role="option"] {
        color: #DFE6E9 !important;
        background-color: transparent !important;
        transition: all 0.2s ease !important;
    }
    
    div[data-baseweb="popover"] li:hover,
    ul[role="listbox"] li:hover,
    div[role="option"]:hover,
    div[role="option"][aria-selected="true"],
    [data-baseweb="menu"] [role="option"]:hover {
        background-color: rgba(255, 107, 107, 0.15) !important;
        color: #FF6B6B !important;
    }
    
    .stTextInput input {
        background-color: rgba(10, 12, 22, 0.6) !important;
        border-radius: 16px !important;
        border: 1.5px solid rgba(255, 255, 255, 0.1) !important;
        color: #FFFFFF !important;
        padding: 12px 18px !important;
    }
    .stTextInput input:focus {
        border-color: #FF6B6B !important;
    }
    
    /* File uploader styling */
    div[data-testid="stFileUploader"] {
        border: 2px dashed rgba(255, 255, 255, 0.15) !important;
        background-color: rgba(255, 255, 255, 0.02) !important;
        border-radius: 20px !important;
        padding: 20px !important;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #FF6B6B !important;
    }
    div[data-testid="stFileUploader"] * {
        color: #DFE6E9 !important;
    }
    
    /* Alert overrides */
    div[data-testid="stAlert"] {
        border-left: 5px solid #FF6B6B !important;
        background-color: rgba(22, 28, 54, 0.8) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        color: #FFFFFF !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2) !important;
    }
    div[data-testid="stAlert"] * {
        color: #FFFFFF !important;
    }
    
    /* Tabs customization to match Inter SaaS style */
    div[data-baseweb="tab-list"] {
        gap: 24px !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        margin-bottom: 20px !important;
    }
    button[data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #A0AEC0 !important;
        background-color: transparent !important;
        padding: 12px 4px !important;
        border: none !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #FF6B6B !important;
        border-bottom: 2px solid #FF6B6B !important;
    }
    
    /* Sidebar text colors */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #A0AEC0 !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) * {
        color: #ffffff !important;
    }
    
    /* Restore dark text specifically for components that require it */
    .dashboard-card [data-testid="stMarkdownContainer"] p,
    .stat-card [data-testid="stMarkdownContainer"] p,
    .fruit-dataset-card [data-testid="stMarkdownContainer"] p,
    .result-card [data-testid="stMarkdownContainer"] p {
        color: #DFE6E9 !important;
    }
    
    .dashboard-card h1, .dashboard-card h2, .dashboard-card h3,
    .stat-card h1, .stat-card h2, .stat-card h3,
    .fruit-dataset-card h1, .fruit-dataset-card h2, .fruit-dataset-card h3,
    .result-card h1, .result-card h2, .result-card h3 {
        color: #FFFFFF !important;
    }
    
    .stat-card .stat-value {
        color: #FFFFFF !important;
    }
    .stat-card .stat-label {
        color: #A0AEC0 !important;
    }
    
    .fruit-dataset-card div {
        color: #FFFFFF !important;
    }
    .fruit-dataset-card span {
        color: #DFE6E9 !important;
    }
</style>
""", unsafe_allow_html=True)

# Session States for Page routing and parameters
if 'page' not in st.session_state:
    st.session_state.page = "ASR"
if 'tts_text' not in st.session_state:
    st.session_state.tts_text = ""
if 'tts_speed' not in st.session_state:
    st.session_state.tts_speed = "Normal"
if 'tts_gender' not in st.session_state:
    st.session_state.tts_gender = "Perempuan"
if 'last_pred' not in st.session_state:
    st.session_state.last_pred = "N/A"
if 'last_conf' not in st.session_state:
    st.session_state.last_conf = "N/A"

# Sidebar header and custom info
st.sidebar.markdown("""
<div style='margin-top: 10px; margin-bottom: 30px;'>
    <div style='display: flex; align-items: center; gap: 12px;'>
        <div style='background: linear-gradient(135deg, #FF6B6B, #FF8E53); width: 44px; height: 44px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 24px; color: white;'>🍊</div>
        <div>
            <div style='font-size: 20px; font-weight: 800; background: linear-gradient(135deg, #FF6B6B, #FF8E53); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.1;'>FruitSpeech</div>
            <div style='font-size: 10px; color: #7D635E; font-weight: 600;'>ASR & TTS AI Dashboard</div>
        </div>
    </div>
</div>
<hr style='border-color: rgba(255, 255, 255, 0.08); margin-top: 15px; margin-bottom: 25px;'/>
""", unsafe_allow_html=True)

# Sidebar menu navigation
page_choice = st.sidebar.radio(
    "Pilih Modul Aplikasi:",
    ["🎙️ Deteksi Suara (ASR)", "🗣️ Sintesis Suara (TTS)", "⚙️ Manajemen Dataset & Model"],
    index=0 if st.session_state.page == "ASR" else (1 if st.session_state.page == "TTS" else 2),
    label_visibility="collapsed"
)

# Sync sidebar choice to session state
if "ASR" in page_choice:
    st.session_state.page = "ASR"
elif "TTS" in page_choice:
    st.session_state.page = "TTS"
else:
    st.session_state.page = "Dataset"

# Sidebar footer
st.sidebar.markdown("""
<div style='margin-top: 150px;'>
    <hr style='border-color: rgba(255, 255, 255, 0.08); margin-bottom: 10px;'/>
    <div style='font-size: 11px; font-weight: 750; color: #FFFFFF;'>FruitSpeech AI</div>
    <div style='font-size: 9px; color: #A0AEC0;'>Version 1.0</div>
</div>
""", unsafe_allow_html=True)

# Load model metadata for global panel
meta = load_model_metadata()
accuracy_val = f"{meta['accuracy']*100:.2f}%" if meta['accuracy'] is not None else "N/A"
model_status_val = f"Siap ({meta['model_type']})" if meta['model_type'] != "N/A" else "Belum Latih"
model_status_color = "#2ECC71" if meta['model_type'] != "N/A" else "#F39C12"

# Calculate dataset stats
stats = get_dataset_stats()
total_samples = sum(stats.values())
total_fruits = len(FRUITS)

# Header inside main content (Presentation Friendly)
st.markdown("""
<div style='margin-top: -15px; margin-bottom: 30px;'>
    <h1 style='font-size: 42px; font-weight: 800; letter-spacing: -1px; margin-bottom: 6px; color: #FFFFFF;'>Dashboard Utama</h1>
    <p style='font-size: 18px; color: #FFE5DC; margin: 0; text-shadow: 0 1px 4px rgba(0,0,0,0.4); font-weight: 500; line-height: 1.6;'>Pusat kontrol terpadu untuk pengenalan suara (ASR) dan sintesis wicara (TTS) buah-buahan.</p>
</div>
""", unsafe_allow_html=True)

# Top Dashboard Stats Grid (4 columns)
col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

with col_stat1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon" style="color: #FF6B6B;">🎙️</div>
        <div class="stat-info">
            <div class="stat-label">Total Samples</div>
            <div class="stat-value">{total_samples}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_stat2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon" style="color: #3498DB;">🍇</div>
        <div class="stat-info">
            <div class="stat-label">Total Fruits</div>
            <div class="stat-value">{total_fruits} Kelas</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_stat3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon" style="color: {model_status_color};">🤖</div>
        <div class="stat-info">
            <div class="stat-label">Model Status</div>
            <div class="stat-value" style="font-size: 15px; margin-top: 8px; font-weight: 800; color: {model_status_color};">{model_status_val}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_stat4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon" style="color: #F1C40F;">🎯</div>
        <div class="stat-info">
            <div class="stat-label">Accuracy</div>
            <div class="stat-value">{accuracy_val}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Check if model trained
model_exists = os.path.exists(MODEL_PATH)

# ----------------- PAGE 1: DETEKSI SUARA (ASR) -----------------
if st.session_state.page == "ASR":
    st.markdown("<h2 style='font-size: 26px; font-weight: 700; margin-top: 20px; margin-bottom: 8px; color: #FFFFFF;'>🎙️ Deteksi Suara (ASR)</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 16px; color: #DFE6E9; line-height: 1.6;'>Gunakan modul ini untuk merekam atau mengunggah suara buah dalam Bahasa Indonesia dan mendeteksinya secara langsung.</p>", unsafe_allow_html=True)
    
    if not model_exists:
        st.warning("⚠️ **Model klasifikasi belum dilatih.** Silakan latih model terlebih dahulu di halaman **⚙️ Manajemen Dataset & Model** agar sistem dapat memprediksi suara.")
    
    # ASR Top Metrics Panel
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">⚙️</div>
            <div class="stat-info">
                <div class="stat-label">Model Status</div>
                <div class="stat-value">{meta['model_type'] if meta['model_type'] != 'N/A' else 'Belum Latih'}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">🍎</div>
            <div class="stat-info">
                <div class="stat-label">Prediksi Terakhir</div>
                <div class="stat-value">{st.session_state.last_pred}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">🎯</div>
            <div class="stat-info">
                <div class="stat-label">Confidence</div>
                <div class="stat-value">{st.session_state.last_conf}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    audio_source = None
    temp_predict_file = "temp_predict.wav"
    
    with col1:
        st.markdown("<div class='dashboard-card' style='height: 100%;'>", unsafe_allow_html=True)
        st.markdown("### 📥 Audio Input", unsafe_allow_html=True)
        st.write("")
        
        # Audio input selection
        input_type = st.radio("Pilih metode input audio:", ["Rekam Langsung (Mikrofon)", "Unggah File Audio (.wav)"])
        st.write("")
        
        if input_type == "Rekam Langsung (Mikrofon)":
            st.write("Tekan tombol di bawah untuk merekam suara Anda menyebutkan nama buah selama 5 detik.")
            st.write("")
            # Button to trigger recording (Primary CTA button layout)
            if st.button("🎙️ Mulai Merekam Suara (5 Detik)", use_container_width=True, type="primary"):
                try:
                    record_audio_streamlit(temp_predict_file, duration=5.0)
                    audio_source = temp_predict_file
                except Exception as e:
                    st.error(f"Gagal merekam suara: {e}. Pastikan mikrofon Anda aktif dan aplikasi memiliki izin akses.")
        else:
            uploaded_file = st.file_uploader("Pilih file audio WAV (Sample Rate disarankan 16000Hz):", type=["wav"])
            if uploaded_file is not None:
                with open(temp_predict_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                audio_source = temp_predict_file
                st.success("File berhasil diunggah!")
                
        # Audio Player if audio source exists
        if audio_source and os.path.exists(audio_source):
            st.write("---")
            st.write("🔊 Putar Audio Input:")
            st.audio(audio_source, format='audio/wav')
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='dashboard-card' style='height: 100%;'>", unsafe_allow_html=True)
        st.markdown("### 🔮 Hasil Analisis & Prediksi ASR", unsafe_allow_html=True)
        
        if audio_source and os.path.exists(audio_source):
            if not model_exists:
                st.info("Perekaman berhasil. Namun prediksi tidak dapat dijalankan karena model belum dilatih.")
            else:
                with st.spinner("Mengekstrak fitur dan memprediksi..."):
                    # Predict using predict function
                    res = predict(audio_source, return_dict=True)
                    
                    if res:
                        pred_fruit = res['prediction']
                        confidence = res['confidence']
                        probs = res['probabilities']
                        
                        # Set predictions state
                        st.session_state.last_pred = pred_fruit.upper()
                        st.session_state.last_conf = f"{confidence * 100:.2f}%"
                        
                        # Large premium result card (Presentation Friendly)
                        emoji_dict = {
                            "apel": "🍎", "jeruk": "🍊", "mangga": "🥭", "pisang": "🍌", "semangka": "🍉",
                            "melon": "🍈", "anggur": "🍇", "pepaya": "🥭", "nanas": "🍍", "stroberi": "🍓"
                        }
                        pred_emoji = emoji_dict.get(pred_fruit.lower(), "📁")
                        badge_color = "#2ECC71" if confidence > 0.70 else "#F39C12"
                        badge_text = "Keyakinan Tinggi" if confidence > 0.70 else "Keyakinan Rendah"
                        
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="result-emoji-container">
                                <span class="result-emoji">{pred_emoji}</span>
                            </div>
                            <div class="result-details">
                                <div class="result-label">Buah Terdeteksi</div>
                                <div class="result-value">{pred_fruit.upper()}</div>
                                <div class="result-badge" style="background-color: {badge_color};">{badge_text}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence Score Details
                        st.write("---")
                        st.markdown(f"**Tingkat Keyakinan (Confidence Score): {confidence * 100:.2f}%**")
                        st.progress(confidence)
                        
                        # Chart for probabilities
                        st.write("")
                        st.write("📊 Probabilitas untuk Setiap Buah:")
                        sorted_probs = dict(sorted(probs.items(), key=lambda item: item[1], reverse=True))
                        st.bar_chart(sorted_probs)
                        
                        # Integrasi ASR -> TTS
                        st.write("---")
                        st.markdown("🔄 **Integrasi ASR ➡️ TTS**")
                        
                        speak_text = f"Suara yang didengar adalah {pred_fruit}."
                        
                        if st.button("🗣️ Bacakan Hasil Prediksi (TTS)", use_container_width=True):
                            with st.spinner("Menyintesis suara hasil prediksi..."):
                                y_tts, sr_tts = generate_tts_audio(
                                    text=speak_text, 
                                    speed=st.session_state.tts_speed.lower(), 
                                    gender=st.session_state.tts_gender.lower()
                                )
                                tts_bytes = get_audio_bytes(y_tts, sr_tts)
                                st.audio(tts_bytes, format='audio/wav')
                                st.success(f"TTS: '{speak_text}'")
                        
                        if st.button("📋 Salin Teks Prediksi ke Modul TTS", use_container_width=True):
                            st.session_state.tts_text = f"Saya mendeteksi suara buah {pred_fruit}."
                            st.session_state.page = "TTS"
                            st.rerun()
                            
                    else:
                        st.error("Gagal mendeteksi suara. Pastikan input suara jelas.")
        else:
            st.info("Menunggu input audio. Rekam atau unggah suara terlebih dahulu di panel sebelah kiri.")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Extra Feature: MFCC Visualization
    if audio_source and os.path.exists(audio_source):
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("### 📈 Visualisasi Fitur MFCC (Mel-Frequency Cepstral Coefficients)")
        st.write("MFCC merupakan representasi 'sidik jari suara' (voiceprint) dari kata yang diucapkan. Grafik di bawah menunjukkan distribusi spektrum frekuensi audio dari waktu ke waktu.")
        
        try:
            # Load audio
            y_vis, sr_vis = librosa.load(audio_source, sr=16000)
            
            # Extract MFCC
            n_mfcc_vis = 13
            mfccs_vis = librosa.feature.mfcc(y=y_vis, sr=sr_vis, n_mfcc=n_mfcc_vis)
            
            fig, ax = plt.subplots(figsize=(10, 3.5))
            img = librosa.display.specshow(mfccs_vis, x_axis='time', sr=sr_vis, ax=ax, cmap='magma')
            cbar = fig.colorbar(img, ax=ax, label="Amplitudo Koefisien")
            
            # Dark Mode adjustments for Matplotlib plot
            cbar.ax.yaxis.label.set_color('#DFE6E9')
            cbar.ax.tick_params(colors='#DFE6E9')
            ax.set_title("Visualisasi Spektrogram MFCC (13 Koefisien)", fontsize=12, fontweight='bold', color='#FFFFFF')
            ax.set_xlabel("Waktu (detik)", fontsize=10, color='#DFE6E9')
            ax.set_ylabel("Indeks MFCC (1-13)", fontsize=10, color='#DFE6E9')
            ax.tick_params(colors='#DFE6E9')
            
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            
            st.pyplot(fig)
            
            # Display average MFCC values
            st.write("Rata-rata fitur MFCC (1D Vector yang diinputkan ke model klasifikasi):")
            mfcc_mean = np.mean(mfccs_vis.T, axis=0)
            st.line_chart(mfcc_mean)
            
        except Exception as e:
            st.error(f"Gagal memvisualisasikan MFCC: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- PAGE 2: SINTESIS SUARA (TTS) -----------------
elif st.session_state.page == "TTS":
    st.markdown("<h2 style='font-size: 26px; font-weight: 700; margin-top: 20px; margin-bottom: 8px; color: #FFFFFF;'>🗣️ Sintesis Suara (TTS)</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 16px; color: #DFE6E9; line-height: 1.6;'>Ketik teks dalam Bahasa Indonesia dan ubah menjadi ucapan suara yang alami dengan kontrol kecepatan dan gender suara.</p>", unsafe_allow_html=True)
    
    col_tts1, col_tts2 = st.columns(2)
    
    with col_tts1:
        st.markdown("<div class='dashboard-card' style='height: 100%;'>", unsafe_allow_html=True)
        st.markdown("### 📝 Text Input")
        st.write("Masukkan teks bahasa Indonesia untuk diubah menjadi wicara:")
        input_text = st.text_area(
            "Ketik teks bahasa Indonesia di sini:",
            value=st.session_state.tts_text if st.session_state.tts_text else "Halo! Selamat datang di modul sintesis suara FruitSpeech. Silakan ketik kalimat apa saja.",
            height=200,
            label_visibility="collapsed"
        )
        st.session_state.tts_text = input_text
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_tts2:
        st.markdown("<div class='dashboard-card' style='height: 100%;'>", unsafe_allow_html=True)
        st.markdown("### ⚙️ Voice Settings")
        st.write("Konfigurasi parameter modulator vokal:")
        st.write("")
        
        # Voice Gender selection
        gender_options = ["Perempuan (Original/High Pitch)", "Laki-laki (Deep Pitch)"]
        gender_idx = 0 if st.session_state.tts_gender == "Perempuan" else 1
        gender_sel = st.radio("Pilihan Gender Suara (Pitch-shifting):", gender_options, index=gender_idx)
        st.session_state.tts_gender = "Perempuan" if "Perempuan" in gender_sel else "Laki-laki"
        
        st.write("---")
        
        # Speed selection
        speed_options = ["Lambat", "Normal", "Cepat"]
        speed_idx = 1 # default normal
        if st.session_state.tts_speed == "Lambat":
            speed_idx = 0
        elif st.session_state.tts_speed == "Cepat":
            speed_idx = 2
            
        speed_sel = st.selectbox("Kecepatan Bicara (Time-stretching):", speed_options, index=speed_idx)
        st.session_state.tts_speed = speed_sel
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    
    # Process Speech
    if st.button("🗣️ Sintesis & Putar Suara", use_container_width=True, type="primary"):
        if not input_text.strip():
            st.warning("Silakan masukkan teks terlebih dahulu!")
        else:
            with st.spinner("Sedang memproses audio..."):
                try:
                    # Generate TTS
                    y_tts, sr_tts = generate_tts_audio(
                        text=input_text, 
                        speed=st.session_state.tts_speed.lower(), 
                        gender=st.session_state.tts_gender.lower()
                    )
                    
                    # Convert to bytes
                    audio_bytes = get_audio_bytes(y_tts, sr_tts)
                    st.session_state.tts_audio_bytes = audio_bytes
                    st.success("Sintesis suara berhasil!")
                except Exception as e:
                    st.error(f"Gagal memproses TTS: {e}")
                    
    # Display audio player card if audio exists in session state
    if 'tts_audio_bytes' in st.session_state:
        st.markdown("<div class='dashboard-card' style='margin-top: 24px;'>", unsafe_allow_html=True)
        st.markdown("### 🔊 Hasil Sintesis Suara")
        st.audio(st.session_state.tts_audio_bytes, format='audio/wav')
        
        # Download button
        st.download_button(
            label="💾 Unduh Hasil Suara (.wav)",
            data=st.session_state.tts_audio_bytes,
            file_name="sintesis_fruitspeech.wav",
            mime="audio/wav",
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- PAGE 3: MANAJEMEN DATASET & MODEL -----------------
else:
    st.markdown("<h2 style='font-size: 26px; font-weight: 700; margin-top: 20px; margin-bottom: 8px; color: #FFFFFF;'>⚙️ Manajemen Dataset & Model</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 16px; color: #DFE6E9; line-height: 1.6;'>Kelola data latih Anda, rekam sampel training baru, dan latih model klasifikasi suara.</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Statistik Jumlah Sampel Dataset")
    st.write("Setiap kelas buah memerlukan sampel suara untuk melatih model klasifikasi. Target ideal adalah minimal **10 sampel** per kelas buah.")
    st.write("")
    
    # Emojis for grid display
    emojis = {
        "apel": "🍎", "jeruk": "🍊", "mangga": "🥭", "pisang": "🍌", "semangka": "🍉",
        "melon": "🍈", "anggur": "🍇", "pepaya": "🥭", "nanas": "🍍", "stroberi": "🍓"
    }
    
    # Split into two rows of 5 for grid view
    row1 = FRUITS[:5]
    row2 = FRUITS[5:]
    
    col_g1 = st.columns(5)
    for idx, fruit in enumerate(row1):
        with col_g1[idx]:
            count = stats.get(fruit, 0)
            emoji = emojis.get(fruit, "📁")
            progress_pct = min(100, int((count / 10) * 100))
            st.markdown(f"""
            <div class="fruit-dataset-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 28px;">{emoji}</span>
                    <span style="font-size: 11px; font-weight: 750; color: #FF6B6B; background: rgba(255, 107, 107, 0.1); padding: 2px 10px; border-radius: 12px;">{count} Sampel</span>
                </div>
                <div style="font-size: 15px; font-weight: 700; color: #FFFFFF; margin-bottom: 12px;">{fruit.capitalize()}</div>
                <div class="dataset-progress-bar-bg">
                    <div class="dataset-progress-bar-fg" style="width: {progress_pct}%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px; font-size: 10px; color: #A0AEC0;">
                    <span>Progress Latih</span>
                    <span>{progress_pct}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)
    
    col_g2 = st.columns(5)
    for idx, fruit in enumerate(row2):
        with col_g2[idx]:
            count = stats.get(fruit, 0)
            emoji = emojis.get(fruit, "📁")
            progress_pct = min(100, int((count / 10) * 100))
            st.markdown(f"""
            <div class="fruit-dataset-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 28px;">{emoji}</span>
                    <span style="font-size: 11px; font-weight: 750; color: #FF6B6B; background: rgba(255, 107, 107, 0.1); padding: 2px 10px; border-radius: 12px;">{count} Sampel</span>
                </div>
                <div style="font-size: 15px; font-weight: 700; color: #FFFFFF; margin-bottom: 12px;">{fruit.capitalize()}</div>
                <div class="dataset-progress-bar-bg">
                    <div class="dataset-progress-bar-fg" style="width: {progress_pct}%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px; font-size: 10px; color: #A0AEC0;">
                    <span>Progress Latih</span>
                    <span>{progress_pct}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("</div>", unsafe_allow_html=True)

    # Dataset Analytics Panel
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Dataset Analytics")
    st.write("")
    
    # Calculate metadata stats
    if total_samples > 0:
        largest_class = max(stats.keys(), key=lambda k: stats[k])
        largest_count = stats[largest_class]
        smallest_class = min(stats.keys(), key=lambda k: stats[k])
        smallest_count = stats[smallest_class]
    else:
        largest_class = "N/A"
        largest_count = 0
        smallest_class = "N/A"
        smallest_count = 0
        
    col_an1, col_an2, col_an3 = st.columns(3)
    with col_an1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon" style="color: #2ECC71;">📦</div>
            <div class="stat-info">
                <div class="stat-label">Total Dataset Size</div>
                <div class="stat-value">{total_samples} WAV</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_an2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon" style="color: #3498DB;">📈</div>
            <div class="stat-info">
                <div class="stat-label">Kelas Terbanyak</div>
                <div class="stat-value" style="font-size: 16px; margin-top: 6px; font-weight: 750;">{largest_class.capitalize()} ({largest_count})</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_an3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon" style="color: #F39C12;">📉</div>
            <div class="stat-info">
                <div class="stat-label">Kelas Tersedikit</div>
                <div class="stat-value" style="font-size: 16px; margin-top: 6px; font-weight: 750;">{smallest_class.capitalize()} ({smallest_count})</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("📊 Grafik Distribusi Jumlah File Latih per Kelas Buah:")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sorted_stats = dict(sorted(stats.items(), key=lambda x: x[1], reverse=False)) # reverse=False to show largest at top of horizontal chart
    y_pos = np.arange(len(sorted_stats))
    counts = list(sorted_stats.values())
    fruits_labels = [f.capitalize() for f in sorted_stats.keys()]
    
    # Beautiful colors matching SaaS theme
    colors = ['#FF6B6B' if c > 0 else (1.0, 1.0, 1.0, 0.08) for c in counts]
    bars = ax.barh(y_pos, counts, color=colors, edgecolor=(1.0, 1.0, 1.0, 0.1), height=0.6)
    
    # Adjust axes and labels to be clearly readable (Presentation Friendly)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fruits_labels, fontsize=10, fontweight='bold', color='#FFFFFF')
    ax.set_xlabel("Jumlah Sampel Latih", fontsize=10, fontweight='bold', color='#DFE6E9')
    ax.tick_params(colors='#DFE6E9')
    
    # Hide spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color((1.0, 1.0, 1.0, 0.1))
    ax.spines['bottom'].set_color((1.0, 1.0, 1.0, 0.1))
    
    # Add data labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                va='center', ha='left', fontsize=10, fontweight='bold', color='#FF6B6B' if width > 0 else '#A0AEC0')
                
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Operations tabs (Record and Train)
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Operasi Dataset & Model")
    st.write("Silakan gunakan tab di bawah untuk merekam sampel baru atau melatih model klasifikasi.")
    
    tab1, tab2 = st.tabs(["🎙️ Rekam Sampel Latih Baru", "⚡ Pelatihan Model Klasifikasi"])
    
    with tab1:
        st.write("Rekam sampel suara baru melalui mikrofon MacBook Anda untuk ditambahkan ke dataset.")
        target_fruit = st.selectbox("Pilih buah untuk direkam:", FRUITS)
        st.write("")
        
        if st.button(f"🎙️ Mulai Rekam Sampel '{target_fruit.capitalize()}'", use_container_width=True, type="primary"):
            target_dir = os.path.join(DATASET_DIR, target_fruit)
            os.makedirs(target_dir, exist_ok=True)
            existing_files = [f for f in os.listdir(target_dir) if f.endswith(".wav")]
            next_idx = len(existing_files) + 1
            filename = os.path.join(target_dir, f"{target_fruit}_{next_idx:03d}.wav")
            
            try:
                # Play voice instruction
                speak(f"Ucapkan kata {target_fruit}", lang='id')
                
                # Record
                record_audio_streamlit(filename, duration=5.0)
                st.success(f"Sampel berhasil disimpan di: `{filename}`")
                
                # Playback
                st.audio(filename, format='audio/wav')
                
                # Rerun
                time.sleep(1.0)
                st.rerun()
            except Exception as e:
                st.error(f"Gagal merekam: {e}")
                
        st.write("---")
        st.write("⚙️ Kelola Berkas Data Latih:")
        col_del1, col_del2 = st.columns(2)
        with col_del1:
            if st.button(f"🗑️ Hapus Sampel Terakhir '{target_fruit.capitalize()}'", use_container_width=True):
                target_dir = os.path.join(DATASET_DIR, target_fruit)
                if os.path.exists(target_dir):
                    files = sorted([f for f in os.listdir(target_dir) if f.endswith(".wav")])
                    if files:
                        last_file = os.path.join(target_dir, files[-1])
                        try:
                            os.remove(last_file)
                            st.success(f"Berhasil menghapus sampel terakhir: `{files[-1]}`")
                            time.sleep(1.0)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Gagal menghapus file: {e}")
                    else:
                        st.info("Tidak ada sampel untuk dihapus.")
                else:
                    st.info("Tidak ada folder sampel.")
                    
        with col_del2:
            if st.button(f"🧹 Kosongkan Semua Sampel '{target_fruit.capitalize()}'", use_container_width=True):
                target_dir = os.path.join(DATASET_DIR, target_fruit)
                if os.path.exists(target_dir):
                    files = [f for f in os.listdir(target_dir) if f.endswith(".wav")]
                    if files:
                        for f in files:
                            try:
                                os.remove(os.path.join(target_dir, f))
                            except Exception:
                                pass
                        st.success(f"Berhasil mengosongkan semua sampel buah {target_fruit.capitalize()}")
                        time.sleep(1.0)
                        st.rerun()
                    else:
                        st.info("Folder sudah kosong.")
                else:
                    st.info("Tidak ada folder sampel.")
                
    with tab2:
        st.write("Latih ulang model klasifikasi suara menggunakan seluruh data latih yang ada di folder dataset.")
        st.write("")
        
        # Dedicated model status information grid
        st.markdown(f"""
        <div class="training-info-grid">
            <div class="training-info-item">
                <span class="training-info-label">Algoritma Latih</span>
                <span class="training-info-value">{meta['model_type']}</span>
            </div>
            <div class="training-info-item">
                <span class="training-info-label">Status Model</span>
                <span class="training-info-value" style="color: {model_status_color};">{"Siap Pakai" if meta['model_type'] != 'N/A' else "Belum Dilatih"}</span>
            </div>
            <div class="training-info-item">
                <span class="training-info-label">Akurasi Pengujian</span>
                <span class="training-info-value">{accuracy_val}</span>
            </div>
            <div class="training-info-item">
                <span class="training-info-label">Tanggal Latihan</span>
                <span class="training-info-value" style="font-size: 13px; font-weight: 750; margin-top: 6px;">{meta['trained_at']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        model_type = st.radio("Pilih Arsitektur Model:", ["SVM (Support Vector Machine)", "MLP (Multi-Layer Perceptron)"])
        model_key = "SVM" if "SVM" in model_type else "MLP"
        st.write("")
        
        # Train Button (Large Gradient CTA)
        if st.button("⚡ Latih Model Klasifikasi", use_container_width=True, type="primary"):
            if total_samples < len(FRUITS) or total_samples < 5:
                st.warning("⚠️ **Jumlah sampel audio terlalu sedikit.** Disarankan merekam minimal beberapa sampel untuk setiap buah terlebih dahulu sebelum melatih model agar klasifikasi akurat.")
            else:
                # Clean, informative step-by-step progress status container
                status_container = st.status("⚡ Memulai Pelatihan Model Klasifikasi...", expanded=True)
                progress_bar = st.progress(0)
                
                with status_container:
                    st.write("📂 Membaca dataset dan mengidentifikasi subfolder suara...")
                    time.sleep(0.4)
                    progress_bar.progress(25)
                    
                    st.write("🔊 Mengekstraksi 13 koefisien MFCC dari file audio WAV...")
                    time.sleep(0.4)
                    progress_bar.progress(60)
                    
                    st.write(f"🤖 Melatih classifier {model_key}...")
                    
                    # Train actual model
                    results = train(model_type=model_key)
                    progress_bar.progress(90)
                    time.sleep(0.3)
                    
                    if results['status'] == 'success':
                        st.write("📈 Mengevaluasi akurasi dan performa model pada data pengujian...")
                        progress_bar.progress(100)
                        time.sleep(0.3)
                        
                        status_container.update(label="🎉 Pelatihan model selesai dengan sukses!", state="complete", expanded=False)
                        st.success("🎉 **Pelatihan Model Selesai!**")
                        
                        # Show metric and report
                        st.metric(label="Akurasi Model Evaluasi (Test)", value=f"{results['accuracy'] * 100:.2f}%")
                        st.write(f"Total sampel latih yang digunakan: **{results['num_samples']}**")
                        
                        st.write("📋 **Laporan Klasifikasi per Kelas:**")
                        import pandas as pd
                        report_df = pd.DataFrame(results['report']).transpose()
                        st.dataframe(report_df.style.format(precision=3), use_container_width=True)
                        
                        # Plot Confusion Matrix
                        st.write("📊 **Matriks Kebingungan (Confusion Matrix):**")
                        cm = np.array(results['confusion_matrix'])
                        classes = results['classes']
                        
                        fig, ax = plt.subplots(figsize=(6, 5))
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
                        disp.plot(cmap='Oranges', ax=ax, xticks_rotation=45)
                        
                        # Apply high contrast labels to CM plot
                        ax.tick_params(colors='#FFFFFF')
                        ax.xaxis.label.set_color('#FFFFFF')
                        ax.yaxis.label.set_color('#FFFFFF')
                        ax.title.set_color('#FFFFFF')
                        for text in disp.text_.ravel():
                            text.set_color('#1A0C1A')
                            
                        fig.patch.set_alpha(0.0)
                        ax.patch.set_alpha(0.0)
                        
                        st.pyplot(fig)
                        
                        # Sleep and reload to update global metrics
                        time.sleep(1.0)
                        st.rerun()
                    else:
                        status_container.update(label="❌ Pelatihan model gagal!", state="error", expanded=True)
                        st.error(f"Gagal melatih model: {results['message']}")
                        
    st.markdown("</div>", unsafe_allow_html=True)
