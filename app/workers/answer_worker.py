"""QThread worker for streaming LLM answer generation."""

from PySide6.QtCore import QThread, Signal

from app.services.llm_service import LLMService


class AnswerWorker(QThread):
    """Streams tokens from Ollama in a background thread."""

    token_received = Signal(str)   # each streamed token
    finished = Signal()            # generation complete
    error = Signal(str)            # error message
    status = Signal(str)           # status updates

    def __init__(self, prompt: str, llm_service: LLMService, parent=None):
        super().__init__(parent)
        self._prompt = prompt
        self._llm = llm_service
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        self.status.emit("Generating answer...")
        for token, err in self._llm.generate_stream(self._prompt):
            if self._cancelled:
                break
            if err:
                self.error.emit(err)
                return
            if token:
                self.token_received.emit(token)
        self.finished.emit()
