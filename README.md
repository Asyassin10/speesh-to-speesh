# 🎙️ Speech-to-Speech RAG System for CIH Bank FAQ

Voice-based question answering system using Retrieval-Augmented Generation (RAG) with French language support.

## 📊 System Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    SPEECH-TO-SPEECH RAG PIPELINE                │
└─────────────────────────────────────────────────────────────────┘

1. 🎤 AUDIO INPUT
   └─> User speaks question (5 seconds) // you can add more seconds 
       └─> sounddevice captures microphone input
           └─> Saves as WAV (16kHz, mono)

2. 🔤 SPEECH-TO-TEXT (STT)
   └─> faster-whisper processes audio
       ├─> Model: small (244M parameters)
       ├─> Language: French
       ├─> Device: CPU with int8 quantization
       └─> Output: "Comment ouvrir un compte?"

3. 🧠 EMBEDDING GENERATION
   └─> Ollama embeddinggemma:latest
       ├─> Input: Question text
       ├─> Output: 768-dimensional vector
       └─> Running on: localhost:11434

4. 🔍 VECTOR SEARCH
   └─> Qdrant semantic search
       ├─> Collection: "faq"
       ├─> Metric: Cosine similarity
       ├─> Returns: Top 4 relevant chunks
       └─> Running on: localhost:6333

5. 💬 LLM GENERATION
   └─> OpenAI GPT-4o-mini via OpenRouter
       ├─> Input: Context chunks + question
       ├─> Streaming: Yes (real-time text)
       └─> Output: French answer

6. 🔊 TEXT-TO-SPEECH (TTS)
   └─> Coqui TTS VITS model
       ├─> Model: tts_models/fr/css10/vits
       ├─> Language: French
       ├─> Sample rate: 22050 Hz
       └─> Output: WAV audio

7. 🎵 AUDIO PLAYBACK
   └─> sounddevice plays response
       └─> User hears the answer
```

## 🏗️ Project Structure

```
speesh-to-speesh/
├── rag.py                 # Main speech-to-speech pipeline
├── setup_vectors.py       # Vector database initialization
├── text_pdf.txt          # CIH Bank FAQ source document
├── requirements.txt       # Python dependencies
└── README.md             # This file

```

## 🔧 Technologies Used

### Speech-to-Text
- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** - Optimized OpenAI Whisper implementation
  - CTranslate2 backend for faster inference
  - Model: `small` (244M params)
  - Quantization: `int8` for CPU efficiency
  - Language: French (`fr`)
  - Beam size: 1 (fastest decoding)

### Embeddings
- **[Ollama](https://ollama.ai)** - Local LLM server
  - Model: `embeddinggemma:latest`
  - Dimensions: 768
  - Purpose: Convert text to semantic vectors

### Vector Database
- **[Qdrant](https://qdrant.tech)** - Vector similarity search
  - Distance metric: Cosine similarity
  - Collection: `faq` (CIH Bank FAQ chunks)
  - Chunk size: ~400 characters with overlap

### Language Model
- **OpenRouter API** - LLM gateway // or local llm
  - Model: `openai/gpt-4o-mini`
  - Streaming: Enabled
  - Purpose: Generate contextual answers

### Text-to-Speech
- **[Coqui TTS](https://github.com/coqui-ai/TTS)** - Neural TTS
  - Model: `tts_models/fr/css10/vits`
  - Quality: Clean French voice
  - Sample rate: 22050 Hz

### Audio I/O
- **sounddevice** - Cross-platform audio recording/playback

## 💻 CPU vs GPU Configuration

### Current Setup: CPU ✅
```python
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
```

**Pros:**
- ✅ Works on any machine (no GPU required)
- ✅ Lower memory usage (~1GB RAM)
- ✅ Good for development and testing
- ✅ Sufficient for single-user scenarios

**Cons:**
- ⚠️ Slower transcription (~2-3 seconds per request)
- ⚠️ Limited to smaller models (tiny, base, small)

**When to use CPU:**
- Development and testing
- Single-user or low-traffic deployments
- No NVIDIA GPU available
- Budget constraints

---

### GPU Configuration (Optional) 🚀
```python
whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
# Or for best quality:
whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
```

**Pros:**
- ✅ Much faster transcription (~0.2-0.5 seconds)
- ✅ Can use larger models (medium, large-v3)
- ✅ Better accuracy with large models
- ✅ Handles concurrent requests

**Cons:**
- ⚠️ Requires NVIDIA GPU with CUDA
- ⚠️ Higher memory usage (4-8GB VRAM)
- ⚠️ Additional setup (CUDA, cuDNN)

**When to use GPU:**
- Production deployments
- High-traffic scenarios
- Need for faster response times
- Have NVIDIA GPU with 4GB+ VRAM
- Want to use large/medium models

**GPU Requirements:**
- NVIDIA GPU with CUDA support
- Minimum 4GB VRAM (for small/medium)
- 8GB+ VRAM recommended (for large models)
- CUDA 11.8+ and cuDNN installed

## 🚀 Installation & Setup

### 1. Clone Repository

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start Qdrant
```bash
# Using Docker (recommended)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

```

### 4. Start Ollama & Pull Embedding Model
```bash
# Install Ollama from https://ollama.ai
# Then pull the embedding model:
ollama pull embeddinggemma:latest
```

### 5. Initialize Vector Database (First Time Only)
```bash
python setup_vectors.py
```

This will:
- Read `text_pdf.txt`
- Split into chunks (~400 chars each)
- Generate embeddings for each chunk
- Store in Qdrant collection

### 6. Run the Application
```bash
python rag.py
```

## 📖 Usage

1. Run `python rag.py`
2. Wait for "🎙️ Parlez maintenant (5s)..."
3. Speak your question in French
4. System will:
   - Transcribe your question
   - Search relevant FAQ chunks
   - Generate streaming answer (appears word-by-word)
   - Speak the answer back to you

**Example:**
```
🎙️ Parlez maintenant (5s)...
🎤 Transcription...
Vous: Comment ouvrir un compte?

🔊 Assistant: Pour ouvrir un compte à CIH Bank, vous devez vous
rendre dans une agence CIH Bank et souscrire...
🎵 Lecture audio...
✅ Done!
```

## ⚙️ Configuration

Edit `rag.py` to customize:

```python
# Recording duration
RECORD_SECONDS = 5  # Change to 3, 7, 10 seconds

# Whisper model size
whisper_model = WhisperModel("small", ...)  # tiny, base, medium, large

# Number of retrieved chunks
chunks = search_chunks(question, top_k=4)  # 2, 3, 5, etc.

# TTS speed (in speak function)
wav = tts.tts(text=text, speed=1.0)  # 0.8 (slower), 1.2 (faster)
```

## 🎯 Whisper Model Comparison

| Model | Parameters | Speed | Accuracy | VRAM (GPU) | Use Case |
|-------|-----------|-------|----------|------------|----------|
| `tiny` | 39M | ⚡⚡⚡ Very Fast | ⭐ Basic | ~1GB | Quick testing |
| `base` | 74M | ⚡⚡ Fast | ⭐⭐ Good | ~1GB | Development |
| **`small`** | **244M** | **⚡ Fast** | **⭐⭐⭐ Good** | **~2GB** | **Recommended (CPU)** |
| `medium` | 769M | 🐢 Medium | ⭐⭐⭐⭐ Very Good | ~5GB | GPU recommended |
| `large-v3` | 1.55B | 🐢🐢 Slow | ⭐⭐⭐⭐⭐ Excellent | ~10GB | GPU required |



## 📦 Dependencies

```
requests           # HTTP client for APIs
qdrant-client      # Vector database client
numpy              # Numerical operations
faster-whisper     # Optimized Whisper STT
TTS                # Coqui text-to-speech
sounddevice        # Audio I/O
```

## 🔄 Update Workflow

To update the FAQ knowledge base:

1. Edit `text_pdf.txt` with new content
2. Run `python setup_vectors.py` to rebuild vectors
3. Run `python rag.py` to test

