# Local AI Interview Assistant

A fully local Windows desktop application that records your voice, transcribes it using a local Speech-to-Text model, and generates interview-ready answers using a local Large Language Model — all without sending any data to the cloud.

Inspired by Parakeet AI. Built as an internship assignment project.

---

## Features

- **Record** microphone audio with Start/Stop controls and live duration display
- **Transcribe** speech to text locally using faster-whisper (no internet required)
- **Generate answers** using a local LLM via Ollama (`qwen2.5:3b`)
- **Progressive streaming** — watch the AI answer appear token-by-token in real time
- **Dark modern UI** built with PySide6
- **Fully offline** — no cloud APIs, no data leaves your machine
- **Status indicators** for microphone, Whisper model, and Ollama connection

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| GUI Framework | PySide6 (Qt for Python) |
| Speech-to-Text | faster-whisper (CTranslate2 backend) |
| LLM Inference | Ollama (local HTTP API) |
| Default LLM Model | qwen2.5:3b |
| Audio Capture | sounddevice + NumPy |
| Language | Python 3.11+ |
| Platform | Windows 10/11 |

---

## Architecture

```
┌─────────────────────────────────────────────┐
│                  MainWindow                 │
│              (PySide6 Dark UI)              │
├──────────┬──────────────┬───────────────────┤
│  Audio   │  Speech-to-  │    LLM Service    │
│ Recorder │    Text      │   (Ollama HTTP)   │
│(sounddev)│(faster-whisp)│                   │
├──────────┼──────────────┼───────────────────┤
│          │ Transcription│   Answer Worker   │
│          │   Worker     │   (QThread +      │
│          │  (QThread)   │    streaming)     │
└──────────┴──────────────┴───────────────────┘
```

### How it works (beginner-friendly)

1. **You click "Start Recording"** → The app opens your microphone using `sounddevice` and captures audio in a background thread so the window stays responsive.
2. **You click "Stop Recording"** → The captured audio (stored in memory as a NumPy array) is handed to a `TranscriptionWorker`.
3. **TranscriptionWorker** runs `faster-whisper` in a separate thread. It loads the "small" Whisper model (downloaded automatically on first run) and converts your speech into text.
4. **The text appears** in the Transcription box.
5. **You click "Suggest Answer"** → The transcription text is sent as a prompt to an `AnswerWorker`.
6. **AnswerWorker** makes an HTTP request to your local Ollama server (`http://localhost:11434`). It uses the **streaming** API, which means Ollama sends back one word/token at a time instead of waiting for the whole answer.
7. **Each token** arrives via a Qt signal and is appended to the Answer text box immediately — creating a smooth typing effect.
8. **Everything runs locally.** No audio or text is ever sent to OpenAI, Google, Anthropic, or any cloud service.

---

## Folder Structure

```
Local-AI-Interview-Assistant/
│
├── app/
│   ├── __init__.py
│   ├── main.py                      # Entry point
│   │
│   ├── ui/
│   │   ├── __init__.py
│   │   └── main_window.py            # PySide6 dark-themed UI
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── audio_recorder.py         # Microphone capture (sounddevice)
│   │   ├── speech_to_text.py         # faster-whisper wrapper
│   │   └── llm_service.py            # Ollama HTTP streaming client
│   │
│   ├── workers/
│   │   ├── __init__.py
│   │   ├── transcription_worker.py   # QThread for STT
│   │   └── answer_worker.py          # QThread for LLM streaming
│   │
│   └── utils/
│       ├── __init__.py
│       └── config.py                 # All configurable constants
│
├── models/
│   └── README.md                     # Model storage notes
│
├── assets/                           # Future icons/images
│
├── requirements.txt
├── .gitignore
├── README.md
├── run.bat                           # Windows launcher script
└── LICENSE
```

---

## Prerequisites

- **Python 3.11+** installed and on PATH
- **Ollama** installed and running
- A working **microphone**

---

## Installation & Setup

Open **PowerShell** and run these commands:

### 1. Navigate to the project folder

```powershell
cd C:\Users\singh\Desktop\Projects\Local-AI-Interview-Assistant
```

### 2. Create a virtual environment

```powershell
python -m venv venv
```

### 3. Activate the virtual environment

```powershell
venv\Scripts\activate
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

### 5. Install and set up Ollama

Download Ollama from [https://ollama.com/download/windows](https://ollama.com/download/windows) and install it.

Then pull the default model:

```powershell
ollama pull qwen2.5:3b
```

Make sure Ollama is running (it usually starts automatically after installation and sits in the system tray).

### 6. Run the application

```powershell
python -m app.main
```

Or simply double-click `run.bat` in the project folder.

---

## Configuration

All configurable values are in `app/utils/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `WHISPER_MODEL_SIZE` | `"small"` | Whisper model size (tiny/base/small/medium/large) |
| `WHISPER_DEVICE` | `"cpu"` | Inference device (cpu or cuda) |
| `OLLAMA_MODEL` | `"qwen2.5:3b"` | LLM model name |
| `OLLAMA_BASE_URL` | `"http://localhost:11434"` | Ollama API endpoint |
| `SAMPLE_RATE` | `16000` | Audio sample rate |

To use a GPU for Whisper (requires CUDA), change `WHISPER_DEVICE` to `"cuda"` and `WHISPER_COMPUTE_TYPE` to `"float16"`.

---

## Troubleshooting

### "Cannot connect to Ollama"
- Make sure Ollama is installed and running. Check the system tray for the Ollama icon.
- Run `ollama serve` manually in a terminal if needed.

### "Model 'qwen2.5:3b' not found"
- Run: `ollama pull qwen2.5:3b`

### "Microphone unavailable"
- Check Windows Settings → Privacy → Microphone → allow desktop apps.
- Ensure a microphone is connected and set as the default input device.

### "Whisper model download takes long"
- The `small` model (~460 MB) downloads automatically on first transcription. This only happens once.

### Application crashes on startup
- Ensure you're using Python 3.11+.
- Re-run `pip install -r requirements.txt` inside the activated virtual environment.

---

## How Local Speech-to-Text Works

The app uses **faster-whisper**, which is an optimized reimplementation of OpenAI's Whisper model using CTranslate2. It runs entirely on your CPU (or GPU if configured). When you stop recording, the audio buffer is passed to a background thread where the model processes it and returns text. No audio is uploaded anywhere.

## How Local LLM Works

**Ollama** is a local LLM runtime that serves models through an HTTP API on `localhost:11434`. The app sends the transcribed text as a prompt to Ollama's `/api/generate` endpoint with `stream: true`. Ollama runs `qwen2.5:3b` locally and streams back tokens one at a time.

## How Progressive Streaming Works

Instead of waiting for the full response, the app uses Ollama's streaming mode. Each JSON line from the HTTP response contains a single token. A `QThread` worker reads these tokens and emits them via Qt signals. The main window connects to these signals and appends each token to the text area immediately, creating a smooth real-time typing effect while keeping the GUI fully responsive.

---

## Testing Checklist

After setup, verify:

- [x] Application opens successfully with dark theme
- [x] Start button begins recording
- [x] Stop button stops recording and shows duration
- [x] Microphone audio is captured
- [x] Whisper transcribes the audio locally
- [x] Transcription appears in the UI
- [x] Suggest Answer button sends prompt to Ollama
- [x] Ollama generates an answer locally
- [x] Answer appears progressively (streaming)
- [x] GUI remains responsive during all operations
- [x] Error messages display correctly when services are unavailable

---

## Limitations

- First transcription requires downloading the Whisper model (~460 MB for "small").
- CPU-only Whisper may be slow on older machines for long recordings.
- Requires Ollama to be installed and running separately.
- The `qwen2.5:3b` model is lightweight; for higher quality answers, consider larger models like `qwen2.5:7b` or `llama3`.

---

## License

MIT License. See [LICENSE](LICENSE).
