"""Main application window built with PySide6."""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QFrame, QSizePolicy,
)
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QFont

from app.utils.config import APP_NAME, APP_SUBTITLE, WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT
from app.services.audio_recorder import AudioRecorder
from app.services.speech_to_text import SpeechToTextService
from app.services.llm_service import LLMService
from app.workers.transcription_worker import TranscriptionWorker
from app.workers.answer_worker import AnswerWorker

import numpy as np


DARK_STYLESHEET = """
QMainWindow {
    background-color: #1a1a2e;
}
QWidget {
    color: #e0e0e0;
    font-family: "Segoe UI", Arial, sans-serif;
}
QLabel#title {
    font-size: 22px;
    font-weight: 700;
    color: #ffffff;
}
QLabel#subtitle {
    font-size: 13px;
    color: #8888aa;
}
QLabel#section_header {
    font-size: 14px;
    font-weight: 600;
    color: #c0c0d0;
    margin-top: 8px;
}
QPushButton#btn_start {
    background-color: #0f9b58;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 14px;
    font-weight: 600;
}
QPushButton#btn_start:hover { background-color: #0d8a4e; }
QPushButton#btn_start:disabled { background-color: #333355; color: #666688; }

QPushButton#btn_stop {
    background-color: #d93025;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 14px;
    font-weight: 600;
}
QPushButton#btn_stop:hover { background-color: #c12a20; }
QPushButton#btn_stop:disabled { background-color: #333355; color: #666688; }

QPushButton#btn_suggest {
    background-color: #4a6cf7;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 14px;
    font-weight: 600;
}
QPushButton#btn_suggest:hover { background-color: #3b5de7; }
QPushButton#btn_suggest:disabled { background-color: #333355; color: #666688; }

QTextEdit {
    background-color: #16213e;
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    padding: 8px;
    font-size: 13px;
    color: #d0d0e0;
}
QFrame#status_bar {
    background-color: #0f0f23;
    border-top: 1px solid #2a2a4a;
    padding: 6px;
}
QLabel#status_label {
    font-size: 11px;
    color: #8888aa;
}
QLabel#indicator_ok {
    font-size: 11px;
    color: #0f9b58;
    font-weight: 600;
}
QLabel#indicator_err {
    font-size: 11px;
    color: #d93025;
    font-weight: 600;
}
QLabel#recording_status {
    font-size: 13px;
    color: #ff6b6b;
    font-weight: 600;
}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.setStyleSheet(DARK_STYLESHEET)

        # Services
        self._recorder = AudioRecorder(self)
        self._stt = SpeechToTextService()
        self._llm = LLMService()

        # Workers (held as attributes so they aren't GC'd)
        self._transcription_worker: TranscriptionWorker | None = None
        self._answer_worker: AnswerWorker | None = None

        # Timer for duration display
        self._duration_timer = QTimer(self)
        self._duration_timer.setInterval(500)
        self._duration_timer.timeout.connect(self._tick_duration)
        self._recording_seconds = 0.0

        self._build_ui()
        self._connect_signals()
        self._check_services()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 16, 20, 8)
        layout.setSpacing(10)

        # --- Header ---
        title = QLabel(APP_NAME)
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel(APP_SUBTITLE)
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        self._local_indicator = QLabel("● Local AI")
        self._local_indicator.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._local_indicator)

        layout.addSpacing(6)

        # --- Recording section ---
        rec_header = QLabel("Recording")
        rec_header.setObjectName("section_header")
        layout.addWidget(rec_header)

        btn_row = QHBoxLayout()
        self._btn_start = QPushButton("Start Recording")
        self._btn_start.setObjectName("btn_start")
        self._btn_stop = QPushButton("Stop Recording")
        self._btn_stop.setObjectName("btn_stop")
        self._btn_stop.setEnabled(False)
        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_stop)
        btn_row.addStretch()

        self._lbl_rec_status = QLabel("")
        self._lbl_rec_status.setObjectName("recording_status")
        btn_row.addWidget(self._lbl_rec_status)

        self._lbl_duration = QLabel("00:00")
        self._lbl_duration.setStyleSheet("font-size:13px; color:#aaaacc;")
        btn_row.addWidget(self._lbl_duration)

        layout.addLayout(btn_row)

        # --- Transcription section ---
        trans_header = QLabel("Transcription")
        trans_header.setObjectName("section_header")
        layout.addWidget(trans_header)

        self._txt_transcription = QTextEdit()
        self._txt_transcription.setReadOnly(True)
        self._txt_transcription.setPlaceholderText("Speech transcription will appear here...")
        self._txt_transcription.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._txt_transcription)

        # --- AI Answer section ---
        ans_header = QLabel("AI Answer")
        ans_header.setObjectName("section_header")
        layout.addWidget(ans_header)

        suggest_row = QHBoxLayout()
        self._btn_suggest = QPushButton("Suggest Answer")
        self._btn_suggest.setObjectName("btn_suggest")
        self._btn_suggest.setEnabled(False)
        suggest_row.addWidget(self._btn_suggest)
        suggest_row.addStretch()
        self._lbl_gen_status = QLabel("")
        self._lbl_gen_status.setStyleSheet("font-size:12px; color:#4a6cf7;")
        suggest_row.addWidget(self._lbl_gen_status)
        layout.addLayout(suggest_row)

        self._txt_answer = QTextEdit()
        self._txt_answer.setReadOnly(True)
        self._txt_answer.setPlaceholderText("Generated answer will appear here...")
        self._txt_answer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._txt_answer)

        # --- Status bar ---
        status_frame = QFrame()
        status_frame.setObjectName("status_bar")
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(10, 4, 10, 4)

        self._lbl_mic = QLabel("Mic: checking...")
        self._lbl_mic.setObjectName("status_label")
        self._lbl_whisper = QLabel("Whisper: checking...")
        self._lbl_whisper.setObjectName("status_label")
        self._lbl_ollama = QLabel("Ollama: checking...")
        self._lbl_ollama.setObjectName("status_label")

        status_layout.addWidget(self._lbl_mic)
        status_layout.addStretch()
        status_layout.addWidget(self._lbl_whisper)
        status_layout.addStretch()
        status_layout.addWidget(self._lbl_ollama)

        layout.addWidget(status_frame)

    # -------------------------------------------------------------- Signals
    def _connect_signals(self):
        self._btn_start.clicked.connect(self._on_start_recording)
        self._btn_stop.clicked.connect(self._on_stop_recording)
        self._btn_suggest.clicked.connect(self._on_suggest_answer)

        self._recorder.recording_started.connect(self._on_recording_started)
        self._recorder.recording_stopped.connect(self._on_recording_stopped)
        self._recorder.error_occurred.connect(self._on_recording_error)
        self._recorder.duration_updated.connect(self._on_duration_update)

    # --------------------------------------------------------- Service checks
    def _check_services(self):
        # Microphone
        try:
            import sounddevice as sd
            devs = sd.query_devices()
            default_in = sd.default.device[0]
            if default_in is not None and default_in >= 0:
                self._lbl_mic.setText("Mic: ready")
                self._lbl_mic.setObjectName("indicator_ok")
            else:
                self._lbl_mic.setText("Mic: no input device")
                self._lbl_mic.setObjectName("indicator_err")
        except Exception:
            self._lbl_mic.setText("Mic: error")
            self._lbl_mic.setObjectName("indicator_err")
        self._lbl_mic.style().unpolish(self._lbl_mic)
        self._lbl_mic.style().polish(self._lbl_mic)

        # Whisper (deferred – loaded on first use)
        self._lbl_whisper.setText("Whisper: ready (lazy)")
        self._lbl_whisper.setObjectName("indicator_ok")
        self._lbl_whisper.style().unpolish(self._lbl_whisper)
        self._lbl_whisper.style().polish(self._lbl_whisper)

        # Ollama
        ok, msg = self._llm.check_connection()
        if ok:
            model_ok, model_msg = self._llm.check_model()
            if model_ok:
                self._lbl_ollama.setText("Ollama: connected")
                self._lbl_ollama.setObjectName("indicator_ok")
                self._local_indicator.setObjectName("indicator_ok")
            else:
                self._lbl_ollama.setText(f"Ollama: {model_msg}")
                self._lbl_ollama.setObjectName("indicator_err")
                self._local_indicator.setObjectName("indicator_err")
        else:
            self._lbl_ollama.setText("Ollama: not running")
            self._lbl_ollama.setObjectName("indicator_err")
            self._local_indicator.setObjectName("indicator_err")
        self._lbl_ollama.style().unpolish(self._lbl_ollama)
        self._lbl_ollama.style().polish(self._lbl_ollama)
        self._local_indicator.style().unpolish(self._local_indicator)
        self._local_indicator.style().polish(self._local_indicator)

    # -------------------------------------------------------- Recording slots
    @Slot()
    def _on_start_recording(self):
        self._recorder.start_recording()

    @Slot()
    def _on_stop_recording(self):
        self._recorder.stop_recording()

    @Slot()
    def _on_recording_started(self):
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._lbl_rec_status.setText("● Recording...")
        self._recording_seconds = 0.0
        self._lbl_duration.setText("00:00")
        self._duration_timer.start()

    @Slot(np.ndarray)
    def _on_recording_stopped(self, audio: np.ndarray):
        self._duration_timer.stop()
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._lbl_rec_status.setText("")

        if len(audio) == 0:
            self._txt_transcription.setPlainText("[Empty recording]")
            self._btn_suggest.setEnabled(False)
            return

        # Launch transcription worker
        self._transcription_worker = TranscriptionWorker(audio, self._stt)
        self._transcription_worker.finished.connect(self._on_transcription_done)
        self._transcription_worker.error.connect(self._on_transcription_error)
        self._transcription_worker.status.connect(self._on_transcription_status)
        self._transcription_worker.start()

    @Slot(str)
    def _on_recording_error(self, msg: str):
        self._duration_timer.stop()
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._lbl_rec_status.setText("")
        self._txt_transcription.setPlainText(f"Error: {msg}")

    @Slot(float)
    def _on_duration_update(self, seconds: float):
        self._recording_seconds = seconds

    def _tick_duration(self):
        mins = int(self._recording_seconds) // 60
        secs = int(self._recording_seconds) % 60
        self._lbl_duration.setText(f"{mins:02d}:{secs:02d}")

    # ----------------------------------------------- Transcription slots
    @Slot(str)
    def _on_transcription_done(self, text: str):
        self._txt_transcription.setPlainText(text)
        self._btn_suggest.setEnabled(bool(text.strip()))
        self._lbl_rec_status.setText("")

    @Slot(str)
    def _on_transcription_error(self, msg: str):
        self._txt_transcription.setPlainText(f"Transcription error: {msg}")
        self._btn_suggest.setEnabled(False)
        self._lbl_rec_status.setText("")

    @Slot(str)
    def _on_transcription_status(self, msg: str):
        self._lbl_rec_status.setText(msg)

    # --------------------------------------------------- Suggest answer
    @Slot()
    def _on_suggest_answer(self):
        prompt = self._txt_transcription.toPlainText().strip()
        if not prompt:
            return

        self._btn_suggest.setEnabled(False)
        self._txt_answer.clear()
        self._lbl_gen_status.setText("Generating...")

        self._answer_worker = AnswerWorker(prompt, self._llm)
        self._answer_worker.token_received.connect(self._on_token)
        self._answer_worker.finished.connect(self._on_generation_finished)
        self._answer_worker.error.connect(self._on_generation_error)
        self._answer_worker.start()

    @Slot(str)
    def _on_token(self, token: str):
        self._txt_answer.insertPlainText(token)
        # Auto-scroll
        sb = self._txt_answer.verticalScrollBar()
        sb.setValue(sb.maximum())

    @Slot()
    def _on_generation_finished(self):
        self._btn_suggest.setEnabled(True)
        self._lbl_gen_status.setText("")

    @Slot(str)
    def _on_generation_error(self, msg: str):
        self._txt_answer.setPlainText(f"Error: {msg}")
        self._btn_suggest.setEnabled(True)
        self._lbl_gen_status.setText("")
