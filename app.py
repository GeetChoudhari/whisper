import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import whisper
import vosk
import requests
import zipfile
import tempfile
import json
import pydub
import warnings
from vosk import Model, KaldiRecognizer
import wave
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize session state for Vosk model
if 'vosk_model' not in st.session_state:
    st.session_state.vosk_model = None

# Set title and page configuration
st.set_page_config(page_title="Speech Recognition App", layout="wide")
st.title("Speech Recognition using Whisper and Vosk")
st.write("Upload an audio file and select a speech recognition model")

def download_vosk_model():
    """Download and set up Vosk model if not present"""
    model_path = "vosk-model-small-en-us-0.15"
    if not os.path.exists(model_path):
        st.info("Downloading Vosk model... This may take a few minutes.")
        url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
        response = requests.get(url, stream=True)
        
        with open("model.zip", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        with zipfile.ZipFile("model.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        
        os.remove("model.zip")
        st.success("Model downloaded successfully!")
    return model_path

def initialize_vosk_model():
    """Initialize Vosk model with error handling"""
    try:
        model_path = download_vosk_model()
        if not os.path.exists(model_path):
            st.error(f"Model path {model_path} does not exist!")
            return None
        
        st.session_state.vosk_model = Model(model_path)
        return st.session_state.vosk_model
    except Exception as e:
        st.error(f"Error initializing Vosk model: {str(e)}")
        logger.error(f"Vosk model initialization error: {str(e)}")
        return None

def convert_to_wav(input_path):
    """Convert audio to WAV format suitable for Vosk"""
    try:
        audio = pydub.AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            audio.export(temp_wav.name, format="wav", parameters=["-acodec", "pcm_s16le"])
            return temp_wav.name
    except Exception as e:
        st.error(f"Error converting audio: {str(e)}")
        logger.error(f"Audio conversion error: {str(e)}")
        return None

def transcribe_with_vosk(audio_path):
    """Transcribe audio using Vosk with improved error handling"""
    try:
        # Ensure model is initialized
        if st.session_state.vosk_model is None:
            st.session_state.vosk_model = initialize_vosk_model()
            if st.session_state.vosk_model is None:
                return None

        # Convert audio to correct format
        wav_path = convert_to_wav(audio_path)
        if wav_path is None:
            return None

        # Create recognizer
        rec = KaldiRecognizer(st.session_state.vosk_model, 16000)
        
        # Process audio
        result = []
        with wave.open(wav_path, "rb") as wf:
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    part_result = json.loads(rec.Result())
                    result.append(part_result.get("text", ""))

        final_result = json.loads(rec.FinalResult())
        result.append(final_result.get("text", ""))
        
        # Cleanup
        os.unlink(wav_path)
        
        return " ".join(result).strip()
    except Exception as e:
        st.error(f"Error during Vosk transcription: {str(e)}")
        logger.error(f"Vosk transcription error: {str(e)}")
        return None

def transcribe_with_whisper(audio_path):
    """Transcribe audio using Whisper"""
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, fp16=False)
        return result["text"]
    except Exception as e:
        st.error(f"Error during Whisper transcription: {str(e)}")
        logger.error(f"Whisper transcription error: {str(e)}")
        return None

# Create model selection dropdown
model_choice = st.selectbox("Select Speech Recognition Model", 
                           ["OpenAI Whisper", "Vosk"])

# File upload section
uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "ogg", "m4a"])

# Process audio when file is uploaded
if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file)
    
    if st.button("Transcribe Audio"):
        with st.spinner("Processing audio..."):
            try:
                # Save uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # Verify file
                if not os.path.exists(tmp_file_path):
                    st.error("Error: Temporary file not created successfully")
                    st.stop()

                if os.path.getsize(tmp_file_path) == 0:
                    st.error("Error: Uploaded file is empty")
                    st.stop()

                # Process with selected model
                result = None
                if model_choice == "OpenAI Whisper":
                    result = transcribe_with_whisper(tmp_file_path)
                else:  # Vosk
                    result = transcribe_with_vosk(tmp_file_path)

                # Clean up
                try:
                    os.unlink(tmp_file_path)
                except Exception as e:
                    logger.warning(f"Could not delete temporary file: {str(e)}")

                # Display results
                if result:
                    st.success("Transcription Complete")
                    st.markdown(result)
                    st.download_button(
                        label='Download Text',
                        data=result,
                        file_name="transcription.txt"
                    )
                else:
                    st.error("Transcription failed. Please try again.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"General processing error: {str(e)}")
                if 'tmp_file_path' in locals():
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
