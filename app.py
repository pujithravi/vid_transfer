import os
from flask import Flask, render_template, request, redirect
import whisper
from transformers import BartTokenizer, BartForConditionalGeneration
from translate import Translator

app = Flask(__name__)

def convert_video_to_audio(video_file, audio_file):
    os.system(f"ffmpeg -i {video_file} -ab 160k -ac 2 -ar 44100 -vn {audio_file}")

def transcribe_audio_whisper(audio_file, model_type="tiny"):
    model = whisper.load_model(model_type)
    result = model.transcribe(audio_file)
    return result['text']

def summarize_text(text):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=300,
        min_length=100,
        length_penalty=1.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def translate_text(text, to_lang='te'):
    translator = Translator(to_lang=to_lang)
    try:
        translation = translator.translate(text)
        return translation
    except Exception as e:
        print(f"Error during translation: {e}")
        return None

def split_text(text, chunk_size=300):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def translate_large_text(large_text, max_tokens=300, to_lang='te'):
    chunks = split_text(large_text, max_tokens)
    translated_chunks = [translate_text(chunk, to_lang=to_lang) for chunk in chunks]
    return ''.join(translated_chunks)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("No file part")
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return redirect(request.url)

    if file:
        print(f"Received file: {file.filename}")
        video_file = os.path.join('uploads', file.filename)
        file.save(video_file)

        audio_file = "output_audio.wav"
        convert_video_to_audio(video_file, audio_file)
        print(f"Converted {video_file} to {audio_file}")

        transcription = transcribe_audio_whisper(audio_file)
        print(f"Transcription: {transcription}")

        summary = summarize_text(transcription)
        print(f"Summary: {summary}")

        translated_summary = translate_large_text(summary, to_lang='te')
        print(f"Translated Summary: {translated_summary}")

        return render_template('output.html',
                                transcription=transcription,
                                summary=summary,
                                translated_summary=translated_summary)

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
