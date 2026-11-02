import os
import re
import subprocess
import tempfile

import streamlit as st
import whisper
from transformers import BartTokenizer, BartForConditionalGeneration
from translate import Translator

st.set_page_config(page_title="Meeting Summarizer and Translator", layout="centered")

LANGUAGES = {
    "Telugu": "te",
    "Hindi": "hi",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
}

BART_MAX_TOKENS = 1024


def convert_video_to_audio(video_file, audio_file):
    # Pass arguments as a list (no shell) so filenames can't inject commands
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_file, "-ab", "160k", "-ac", "2",
         "-ar", "44100", "-vn", audio_file],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


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
    return result["text"]


def _bart_summarize(text, max_length, min_length):
    tokenizer, model = load_bart_model()
    inputs = tokenizer(text, return_tensors="pt", max_length=BART_MAX_TOKENS, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=1.0,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def chunk_by_tokens(text, max_tokens=900):
    """Split text on sentence boundaries into chunks that fit BART's input limit."""
    tokenizer, _ = load_bart_model()
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current, current_len = [], [], 0
    for sentence in sentences:
        n = len(tokenizer.tokenize(sentence))
        if current and current_len + n > max_tokens:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def summarize_text(text):
    """Map-reduce summarization so long meetings aren't cut off at 1024 tokens."""
    chunks = chunk_by_tokens(text)
    if len(chunks) <= 1:
        return _bart_summarize(text, max_length=300, min_length=100)
    partial = [_bart_summarize(c, max_length=150, min_length=40) for c in chunks]
    combined = " ".join(partial)
    if len(chunk_by_tokens(combined)) > 1:
        # Still too long for one pass: return the section summaries joined
        return combined
    return _bart_summarize(combined, max_length=300, min_length=100)


def split_text(text, chunk_size=300):
    """Split into chunks of up to chunk_size characters without cutting words."""
    words, chunks, current = text.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > chunk_size:
            chunks.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        chunks.append(current)
    return chunks


def translate_text(text, to_lang="te"):
    translator = Translator(to_lang=to_lang)
    try:
        return translator.translate(text)
    except Exception as e:
        st.error(f"Error during translation: {e}")
        return ""


def translate_large_text(large_text, max_chars=300, to_lang="te"):
    chunks = split_text(large_text, max_chars)
    return " ".join(translate_text(chunk, to_lang=to_lang) or "" for chunk in chunks)


st.title("MEETING SUMMARIZER AND TRANSLATOR")

uploaded_file = st.file_uploader("Drop your video file", type=["mp4", "mov", "mkv", "avi"])
language = st.selectbox("Translate summary to", list(LANGUAGES.keys()), index=0)

if uploaded_file is not None:
    if st.button("Process File"):
        # Unique temp files per run so simultaneous users don't overwrite each other
        with tempfile.TemporaryDirectory() as tmp_dir:
            ext = os.path.splitext(uploaded_file.name)[1].lower() or ".mp4"
            video_path = os.path.join(tmp_dir, f"input{ext}")
            audio_path = os.path.join(tmp_dir, "audio.wav")
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                with st.spinner("Extracting audio..."):
                    convert_video_to_audio(video_path, audio_path)

                with st.spinner("Transcribing with Whisper..."):
                    transcription = transcribe_audio_whisper(audio_path)
            except subprocess.CalledProcessError:
                st.error("Could not extract audio from this file. Is it a valid video?")
                st.stop()

        with st.spinner("Summarizing with BART..."):
            summary = summarize_text(transcription)

        with st.spinner(f"Translating to {language}..."):
            translated_summary = translate_large_text(summary, to_lang=LANGUAGES[language])

        st.success("Done!")

        tab1, tab2, tab3 = st.tabs(["Transcription", "Summary", f"{language} Summary"])
        with tab1:
            st.write(transcription)
        with tab2:
            st.write(summary)
        with tab3:
            st.write(translated_summary)
