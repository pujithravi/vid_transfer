import os
import re
import subprocess
import tempfile
from functools import lru_cache

from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import whisper
from transformers import BartTokenizer, BartForConditionalGeneration
from translate import Translator

app = Flask(__name__)

LANGUAGES = {
    "te": "Telugu",
    "hi": "Hindi",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
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


# Load models once and reuse them across requests
@lru_cache(maxsize=1)
def load_whisper_model(model_type="tiny"):
    return whisper.load_model(model_type)


@lru_cache(maxsize=1)
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


def translate_text(text, to_lang="te"):
    translator = Translator(to_lang=to_lang)
    try:
        return translator.translate(text)
    except Exception as e:
        print(f"Error during translation: {e}")
        return ""


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


def translate_large_text(large_text, max_chars=300, to_lang="te"):
    chunks = split_text(large_text, max_chars)
    return " ".join(translate_text(chunk, to_lang=to_lang) or "" for chunk in chunks)


@app.route("/")
def home():
    return render_template("index.html", languages=LANGUAGES)


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        print("No file part")
        return redirect("/")

    file = request.files["file"]
    if file.filename == "":
        print("No selected file")
        return redirect("/")

    to_lang = request.form.get("language", "te")
    if to_lang not in LANGUAGES:
        to_lang = "te"

    filename = secure_filename(file.filename) or "upload.mp4"
    print(f"Received file: {filename}")

    # Unique temp files per request so simultaneous users don't overwrite each other
    with tempfile.TemporaryDirectory() as tmp_dir:
        video_file = os.path.join(tmp_dir, filename)
        audio_file = os.path.join(tmp_dir, "audio.wav")
        file.save(video_file)

        try:
            convert_video_to_audio(video_file, audio_file)
        except subprocess.CalledProcessError:
            return "Could not extract audio from this file. Is it a valid video?", 400

        transcription = transcribe_audio_whisper(audio_file)

    summary = summarize_text(transcription)
    translated_summary = translate_large_text(summary, to_lang=to_lang)

    return render_template(
        "output.html",
        transcription=transcription,
        summary=summary,
        translated_summary=translated_summary,
        language=LANGUAGES[to_lang],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
