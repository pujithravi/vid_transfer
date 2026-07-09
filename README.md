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

Upload a meeting video → ffmpeg extracts audio → Whisper (tiny) transcribes → BART (facebook/bart-large-cnn) summarizes → translated to Telugu.

## Local setup
1. Install ffmpeg: `brew install ffmpeg`
2. `python3 -m venv venv && source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python app.py`
5. Open http://127.0.0.1:5000
