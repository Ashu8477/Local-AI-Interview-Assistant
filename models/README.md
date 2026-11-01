# Models Directory

This directory is used for storing local model files if needed.

The `faster-whisper` library downloads its models automatically on first run
to its own cache directory. You do not need to manually place files here.

For Ollama LLM models, they are managed by the Ollama service itself.
Run `ollama pull qwen2.5:3b` to download the default model.
