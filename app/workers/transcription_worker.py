"""QThread worker for running speech-to-text transcription off the GUI thread."""

import numpy as np
from PySide6.QtCore import QThread, Signal

from app.services.speech_to_text import SpeechToTextService


class TranscriptionWorker(QThread):
    """Runs faster-whisper transcription in a background thread."""

    finished = Signal(str)       # transcription text
    error = Signal(str)          # error message
    status = Signal(str)         # status updates

    def __init__(self, audio: np.ndarray, stt_service: SpeechToTextService, parent=None):
        super().__init__(parent)
        self._audio = audio
        self._stt = stt_service

    def run(self):
        self.status.emit("Loading Whisper model...")
        if not self._stt.is_loaded():
            err = self._stt.load_model()
            if err:
                self.error.emit(err)
                return

        self.status.emit("Transcribing audio...")
        text, err = self._stt.transcribe(self._audio)
        if err:
            self.error.emit(err)
        else:
            self.finished.emit(text)
