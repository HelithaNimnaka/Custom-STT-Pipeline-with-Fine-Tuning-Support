import streamlit as st
from stt_pipeline import transcribe

st.title("Whisper Speech-to-Text")

audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "ogg"])

if audio_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.read())
    transcription = transcribe("temp_audio.wav")
    st.text_area("Transcription", transcription, height=200)
