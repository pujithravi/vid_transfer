import os
import streamlit as st
import whisper
from transformers import BartTokenizer, BartForConditionalGeneration
from translate import Translator

st.set_page_config(page_title="Meeting Summarizer and Translator", layout="centered")

def convert_video_to_audio(video_file, audio_file):
    os.system(f"ffmpeg -y -i {video_file} -ab 160k -ac 2 -ar 44100 -vn {audio_file}")

@st.cache_resource
def load_whisper_model(model_type="tiny"):
    return whisper.load_model(model_type)

@st.cache_resource
def load_bart_model():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

def transcribe_audio_whisper(audio_file, model_type="tiny"):
    model = load_whisper_model(model_type)
    result = model.transcribe(audio_file)
    return result['text']

def summarize_text(text):
    tokenizer, model = load_bart_model()
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=300,
        min_length=100,
        length_penalty=1.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def translate_text(text, to_lang='te'):
    translator = Translator(to_lang=to_lang)
    try:
        return translator.translate(text)
    except Exception as e:
        st.error(f"Error during translation: {e}")
        return ""

def split_text(text, chunk_size=300):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def translate_large_text(large_text, max_tokens=300, to_lang='te'):
    chunks = split_text(large_text, max_tokens)
    translated_chunks = [translate_text(chunk, to_lang=to_lang) for chunk in chunks]
    return ''.join(translated_chunks)

st.title("MEETING SUMMARIZER AND TRANSLATOR")

uploaded_file = st.file_uploader("Drop your video file", type=["mp4", "mov", "mkv", "avi"])

if uploaded_file is not None:
    if st.button("Process File"):
        os.makedirs("uploads", exist_ok=True)
        video_path = os.path.join("uploads", uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        audio_path = "output_audio.wav"

        with st.spinner("Extracting audio..."):
            convert_video_to_audio(video_path, audio_path)

        with st.spinner("Transcribing with Whisper..."):
            transcription = transcribe_audio_whisper(audio_path)

        with st.spinner("Summarizing with BART..."):
            summary = summarize_text(transcription)

        with st.spinner("Translating to Telugu..."):
            translated_summary = translate_large_text(summary, to_lang='te')

        st.success("Done!")

        tab1, tab2, tab3 = st.tabs(["Transcription", "Summary", "Translated Summary"])
        with tab1:
            st.write(transcription)
        with tab2:
            st.write(summary)
        with tab3:
            st.write(translated_summary)
