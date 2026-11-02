---
title: Vid Transfer
emoji: 🎥
colorFrom: yellow
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
---

# TalkTranslate: Meeting Summarizer and Translator

Upload a meeting video → ffmpeg extracts audio → Whisper (tiny) transcribes → BART (facebook/bart-large-cnn) summarizes → summary translated into your chosen language (Telugu, Hindi, Tamil, Kannada, Malayalam, French, Spanish, German).

- Long meetings are summarized chunk by chunk (map-reduce), so transcripts longer than BART's 1024-token limit aren't cut off.
- Models are loaded once and reused across requests.
- Each upload is processed in its own temporary folder, and ffmpeg is called without a shell.

## Local setup
1. Install ffmpeg: `brew install ffmpeg`
2. `python3 -m venv venv && source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python app.py`
5. Open http://127.0.0.1:5000
