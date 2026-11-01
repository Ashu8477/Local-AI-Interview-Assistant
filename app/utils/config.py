"""Application configuration."""

# Audio recording settings
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"

# Whisper settings
WHISPER_MODEL_SIZE = "small"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b"

# LLM system prompt
SYSTEM_PROMPT = (
    "You are a helpful AI assistant.\n\n"
    "Answer the user's question clearly, naturally, and accurately.\n"
    "Keep the answer concise but informative.\n"
    "For interview questions, provide structured and professional answers.\n"
    "Do not invent personal experiences or qualifications.\n"
    "Use simple, conversational language.\n"
    "Answer only the question asked."
)

# UI settings
APP_NAME = "Local AI Interview Assistant"
APP_SUBTITLE = "Your Private Local AI Assistant"
WINDOW_MIN_WIDTH = 700
WINDOW_MIN_HEIGHT = 800
