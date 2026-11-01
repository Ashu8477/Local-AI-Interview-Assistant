"""Local speech-to-text service using faster-whisper."""

import numpy as np
from faster_whisper import WhisperModel

from app.utils.config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE


class SpeechToTextService:
    """Wraps faster-whisper for local transcription."""

    def __init__(self):
        self._model: WhisperModel | None = None

    def load_model(self) -> str:
        """Load the whisper model. Returns an empty string on success or an error message."""
        try:
            self._model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
            return ""
        except Exception as e:
            return f"Failed to load Whisper model '{WHISPER_MODEL_SIZE}': {e}"

    def is_loaded(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> tuple[str, str]:
        """Transcribe audio array. Returns (text, error_message)."""
        if self._model is None:
            err = self.load_model()
            if err:
                return "", err

        if audio is None or len(audio) == 0:
            return "", "Empty audio — nothing to transcribe."

        try:
            segments, info = self._model.transcribe(
                audio,
                beam_size=5,
                vad_filter=True,
            )
            text_parts = [seg.text.strip() for seg in segments]
            text = " ".join(text_parts).strip()
            if not text:
                return "", "No speech detected in the audio."
            return text, ""
        except Exception as e:
            return "", f"Transcription failed: {e}"
