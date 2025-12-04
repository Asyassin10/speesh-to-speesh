import requests
from qdrant_client import QdrantClient
import wave
import json
from faster_whisper import WhisperModel
from TTS.api import TTS
import sounddevice as sd

OLLAMA_URL = "http://localhost:11434"
OPENROUTER_API_KEY = "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"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "faq"
RECORD_SECONDS = 5

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
tts = TTS(model_name="tts_models/fr/css10/vits", progress_bar=False)

def get_embedding(text):
    response = requests.post(f"{OLLAMA_URL}/api/embeddings",
                            json={"model": "embeddinggemma:latest", "prompt": text})
    return response.json()['embedding']

def search_chunks(question, top_k=3):
    query_vector = get_embedding(question)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )
    return [r.payload['text'] for r in results.points]

def get_answer(question):
    chunks = search_chunks(question, top_k=4)
    context = "\n\n".join(chunks)
    prompt = f"""Tu es assistant CIH Bank. Réponds en français de manière utile et claire.

La question peut contenir des erreurs. Comprends l'intention du client et utilise TOUTES les informations du contexte, même si elles sont indirectes.

Si vraiment aucune information pertinente n'existe dans le contexte, dis "Désolé, je n'ai pas cette information. Contactez votre agence CIH Bank."

Contexte:
{context}

Question: {question}

Réponse:"""

    print("🔊 Assistant: ", end="", flush=True)

    response = requests.post("https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}"
        },
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        },
        stream=True)

    full_text = ""
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                line = line[6:]
                if line.strip() == '[DONE]':
                    break
                try:
                    data = json.loads(line)
                    if 'choices' in data and len(data['choices']) > 0:
                        delta = data['choices'][0].get('delta', {})
                        if 'content' in delta:
                            chunk = delta['content']
                            print(chunk, end="", flush=True)
                            full_text += chunk
                except:
                    pass

    print("\n")
    return full_text

def record_audio():
    print(f"🎙️ Parlez maintenant ({RECORD_SECONDS}s)...")
    audio = sd.rec(int(RECORD_SECONDS * 16000), samplerate=16000, channels=1, dtype='int16')
    sd.wait()

    with wave.open("temp_audio.wav", 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio.tobytes())

    return "temp_audio.wav"

def transcribe(audio_file):
    print("🎤 Transcription...")
    segments, _ = whisper_model.transcribe(audio_file, language="fr", beam_size=1)
    text = " ".join([s.text.strip() for s in segments])
    print(f"Vous: {text}\n")
    return text

def speak(text):
    print("🎵 Lecture audio...", flush=True)
    wav = tts.tts(text=text)
    sd.play(wav, samplerate=23050)
    sd.wait()
    print()

if __name__ == "__main__":
    audio_file = record_audio()
    question = transcribe(audio_file)
    answer = get_answer(question)
    speak(answer)
