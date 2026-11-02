from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QFrame, QSizePolicy, QGraphicsOpacityEffect, QSplitter, QStyle
)
from PySide6.QtCore import Qt, Slot, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont
from app.utils.config import APP_NAME, APP_SUBTITLE, WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT
from app.services.audio_recorder import AudioRecorder
from app.services.speech_to_text import SpeechToTextService
from app.services.llm_service import LLMService
from app.workers.transcription_worker import TranscriptionWorker
from app.workers.answer_worker import AnswerWorker
import numpy as np


DARK_STYLESHEET = """
QMainWindow, QWidget#root {
    background: #0B0F16;
    color: #E8EEF7;
    font-family: "Segoe UI", "Segoe UI Variable", Arial, sans-serif;
}
QWidget {
    color: #E8EEF7;
    font-family: "Segoe UI", "Segoe UI Variable", Arial, sans-serif;
}
QFrame#header {
    background: transparent;
    border: none;
}
QLabel#eyebrow {
    color: #7890AE;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1.6px;
}
QLabel#title {
    color: #F4F7FC;
    font-size: 22px;
    font-weight: 650;
}
QLabel#subtitle {
    color: #8291A8;
    font-size: 11px;
}
QFrame#connection_badge {
    background: #11251F;
    border: 1px solid #1C493B;
    border-radius: 13px;
}
QLabel#local_indicator {
    color: #55D6A2;
    font-size: 11px;
    font-weight: 600;
    padding: 5px 10px;
}
QFrame#record_card {
    background: #131B27;
    border: 1px solid #253246;
    border-radius: 14px;
}
QLabel#record_caption {
    color: #8293AA;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
}
QLabel#recording_status {
    color: #8FA0B8;
    font-size: 12px;
    font-weight: 600;
}
QLabel#duration_label {
    color: #F1F5FB;
    font-size: 22px;
    font-weight: 600;
    font-family: "Segoe UI", Arial, sans-serif;
}
QLabel#recording_dot {
    color: #F05E68;
    font-size: 17px;
    font-weight: 700;
}
QFrame#panel {
    background: #121A26;
    border: 1px solid #253246;
    border-radius: 14px;
}
QLabel#panel_index {
    color: #6E89AD;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
}
QLabel#panel_title {
    color: #F0F4FA;
    font-size: 14px;
    font-weight: 650;
}
QLabel#panel_hint {
    color: #7789A2;
    font-size: 10px;
}
QLabel#ai_indicator {
    color: #7FB2FF;
    font-size: 10px;
    font-weight: 600;
}
QTextEdit {
    background: #0E1520;
    border: 1px solid #202D40;
    border-radius: 9px;
    padding: 14px;
    color: #E3EAF4;
    font-size: 13px;
    selection-background-color: #315F9B;
}
QTextEdit:focus {
    border: 1px solid #477FC2;
    background: #101927;
}
QTextEdit::placeholder {
    color: #60718A;
}
QPushButton {
    border: none;
    border-radius: 8px;
    padding: 10px 17px;
    min-height: 20px;
    font-size: 11px;
    font-weight: 650;
}
QPushButton#btn_start {
    background: #27C58B;
    color: #071A14;
}
QPushButton#btn_start:hover { background: #35D69B; }
QPushButton#btn_start:pressed { background: #1EAA76; }
QPushButton#btn_start:disabled {
    background: #203B38;
    color: #718D85;
}
QPushButton#btn_stop {
    background: #38242B;
    color: #F18B96;
    border: 1px solid #5A303B;
}
QPushButton#btn_stop:hover { background: #4A2932; }
QPushButton#btn_stop:pressed { background: #5A303B; }
QPushButton#btn_stop:disabled {
    background: #202A38;
    color: #64748A;
    border: 1px solid #2A3749;
}
QPushButton#btn_suggest {
    background: #397FE0;
    color: #FFFFFF;
}
QPushButton#btn_suggest:hover { background: #4B91F0; }
QPushButton#btn_suggest:pressed { background: #2D6DC6; }
QPushButton#btn_suggest:disabled {
    background: #263A55;
    color: #748AA8;
}
QFrame#status_bar {
    background: #101722;
    border: 1px solid #202D3E;
    border-radius: 10px;
}
QLabel#status_label {
    color: #8292A8;
    font-size: 10px;
    font-weight: 500;
}
QLabel#indicator_ok {
    color: #53D5A0;
    font-size: 10px;
    font-weight: 600;
}
QLabel#indicator_err {
    color: #F07883;
    font-size: 10px;
    font-weight: 600;
}
QLabel#indicator_warn {
    color: #E8BB69;
    font-size: 10px;
    font-weight: 600;
}
QFrame#separator {
    background: #253246;
    max-height: 1px;
    border: none;
}
QSplitter::handle {
    background: transparent;
    width: 10px;
}
"""

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(max(WINDOW_MIN_WIDTH, 980), max(WINDOW_MIN_HEIGHT, 680))
        self.resize(1180, 790)
        self.setStyleSheet(DARK_STYLESHEET)

        self._recorder = AudioRecorder(self)
        self._stt = SpeechToTextService()
        self._llm = LLMService()
        self._transcription_worker: TranscriptionWorker | None = None
        self._answer_worker: AnswerWorker | None = None

        self._duration_timer = QTimer(self)
        self._duration_timer.setInterval(500)
        self._duration_timer.timeout.connect(self._tick_duration)
        self._recording_seconds = 0.0

        self._build_ui()
        self._connect_signals()
        self._check_services()

        self._recording_pulse_animation = QPropertyAnimation(self._recording_pulse_effect, b"opacity")
        self._recording_pulse_animation.setDuration(850)
        self._recording_pulse_animation.setStartValue(1.0)
        self._recording_pulse_animation.setEndValue(0.25)
        self._recording_pulse_animation.setLoopCount(-1)
        self._recording_pulse_animation.setEasingCurve(QEasingCurve.InOutSine)

        self._answer_fade_animation = QPropertyAnimation(self._answer_opacity_effect, b"opacity")
        self._answer_fade_animation.setDuration(260)
        self._answer_fade_animation.setStartValue(0.72)
        self._answer_fade_animation.setEndValue(1.0)
        self._answer_fade_animation.setEasingCurve(QEasingCurve.OutCubic)

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        page = QVBoxLayout(root)
        page.setContentsMargins(26, 22, 26, 18)
        page.setSpacing(18)

        header = QFrame()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(2, 0, 2, 4)
        header_layout.setSpacing(13)

        brand_mark = QLabel()
        brand_mark.setFixedSize(42, 42)
        brand_mark.setAlignment(Qt.AlignCenter)
        brand_mark.setPixmap(self.style().standardIcon(QStyle.SP_MediaVolume).pixmap(23, 23))
        brand_mark.setStyleSheet(
            "background:#17263A;border:1px solid #2A405D;border-radius:12px;padding:8px;"
        )
        header_layout.addWidget(brand_mark)

        brand_text = QVBoxLayout()
        brand_text.setSpacing(3)
        eyebrow = QLabel("PRIVATE VOICE WORKSPACE")
        eyebrow.setObjectName("eyebrow")
        brand_text.addWidget(eyebrow)

        title = QLabel(APP_NAME)
        title.setObjectName("title")
        brand_text.addWidget(title)

        subtitle = QLabel(APP_SUBTITLE)
        subtitle.setObjectName("subtitle")
        brand_text.addWidget(subtitle)
        header_layout.addLayout(brand_text)
        header_layout.addStretch()

        connection = QFrame()
        connection.setObjectName("connection_badge")
        connection_layout = QHBoxLayout(connection)
        connection_layout.setContentsMargins(4, 0, 4, 0)
        self._local_indicator = QLabel("●  LOCAL ENGINE")
        self._local_indicator.setObjectName("local_indicator")
        connection_layout.addWidget(self._local_indicator)
        header_layout.addWidget(connection, 0, Qt.AlignVCenter)
        page.addWidget(header)

        separator = QFrame()
        separator.setObjectName("separator")
        page.addWidget(separator)

        record_card = QFrame()
        record_card.setObjectName("record_card")
        record_layout = QHBoxLayout(record_card)
        record_layout.setContentsMargins(19, 15, 19, 15)
        record_layout.setSpacing(18)

        record_info = QVBoxLayout()
        record_info.setSpacing(5)
        record_caption = QLabel("RECORDING SESSION")
        record_caption.setObjectName("record_caption")
        record_info.addWidget(record_caption)

        status_line = QHBoxLayout()
        status_line.setSpacing(8)
        self._recording_pulse_dot = QLabel("●")
        self._recording_pulse_dot.setObjectName("recording_dot")
        self._recording_pulse_dot.setFixedWidth(15)
        self._recording_pulse_effect = QGraphicsOpacityEffect(self._recording_pulse_dot)
        self._recording_pulse_effect.setOpacity(1.0)
        self._recording_pulse_dot.setGraphicsEffect(self._recording_pulse_effect)
        status_line.addWidget(self._recording_pulse_dot)

        self._lbl_rec_status = QLabel("Ready to record")
        self._lbl_rec_status.setObjectName("recording_status")
        status_line.addWidget(self._lbl_rec_status)
        status_line.addStretch()
        record_info.addLayout(status_line)
        record_layout.addLayout(record_info)
        record_layout.addStretch()

        timer_box = QVBoxLayout()
        timer_box.setSpacing(2)
        timer_caption = QLabel("ELAPSED TIME")
        timer_caption.setObjectName("record_caption")
        timer_caption.setAlignment(Qt.AlignRight)
        timer_box.addWidget(timer_caption)
        self._lbl_duration = QLabel("00:00")
        self._lbl_duration.setObjectName("duration_label")
        self._lbl_duration.setAlignment(Qt.AlignRight)
        timer_box.addWidget(self._lbl_duration)
        record_layout.addLayout(timer_box)

        record_divider = QFrame()
        record_divider.setObjectName("separator")
        record_divider.setFixedWidth(1)
        record_divider.setMinimumHeight(42)
        record_layout.addWidget(record_divider)

        self._btn_start = QPushButton("Start recording")
        self._btn_start.setObjectName("btn_start")
        self._btn_start.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self._btn_start.setIconSize(self._btn_start.iconSize())
        self._btn_start.setMinimumWidth(154)
        self._btn_start.setFixedHeight(42)
        record_layout.addWidget(self._btn_start)

        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setObjectName("btn_stop")
        self._btn_stop.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self._btn_stop.setMinimumWidth(104)
        self._btn_stop.setFixedHeight(42)
        self._btn_stop.setEnabled(False)
        record_layout.addWidget(self._btn_stop)
        page.addWidget(record_card)

        workspace = QSplitter(Qt.Horizontal)
        workspace.setChildrenCollapsible(False)
        workspace.setHandleWidth(12)

        self._transcription_panel = self._create_workspace_panel(
            "01", "Transcription", "LIVE SPEECH-TO-TEXT", "Speech will appear here when you stop recording."
        )
        self._txt_transcription = self._transcription_panel.findChild(QTextEdit, "transcription_editor")
        workspace.addWidget(self._transcription_panel)

        self._answer_panel = self._create_workspace_panel(
            "02", "Suggested answer", "LOCAL MODEL RESPONSE", "Your generated answer will appear here."
        )
        self._txt_answer = self._answer_panel.findChild(QTextEdit, "answer_editor")
        workspace.addWidget(self._answer_panel)
        workspace.setStretchFactor(0, 1)
        workspace.setStretchFactor(1, 1)
        workspace.setSizes([560, 560])
        page.addWidget(workspace, 1)

        answer_controls = QHBoxLayout()
        answer_controls.setContentsMargins(0, 0, 0, 0)
        answer_controls.setSpacing(10)

        self._btn_suggest = QPushButton("Suggest answer")
        self._btn_suggest.setObjectName("btn_suggest")
        self._btn_suggest.setIcon(self.style().standardIcon(QStyle.SP_CommandLink))
        self._btn_suggest.setFixedHeight(40)
        self._btn_suggest.setMinimumWidth(155)
        self._btn_suggest.setEnabled(False)
        answer_controls.addWidget(self._btn_suggest)

        self._lbl_gen_status = QLabel("")
        self._lbl_gen_status.setObjectName("ai_indicator")
        answer_controls.addWidget(self._lbl_gen_status)
        answer_controls.addStretch()

        answer_host = QWidget()
        answer_host.setLayout(answer_controls)
        self._answer_panel.layout().insertWidget(2, answer_host)

        self._answer_opacity_effect = QGraphicsOpacityEffect(self._txt_answer)
        self._answer_opacity_effect.setOpacity(1.0)
        self._txt_answer.setGraphicsEffect(self._answer_opacity_effect)

        status_frame = QFrame()
        status_frame.setObjectName("status_bar")
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(15, 10, 15, 10)
        status_layout.setSpacing(10)

        self._lbl_mic = QLabel("MIC  Checking")
        self._lbl_mic.setObjectName("status_label")
        self._lbl_whisper = QLabel("WHISPER  Checking")
        self._lbl_whisper.setObjectName("status_label")
        self._lbl_ollama = QLabel("OLLAMA  Checking")
        self._lbl_ollama.setObjectName("status_label")

        status_layout.addWidget(self._lbl_mic)
        status_layout.addStretch()
        status_layout.addWidget(self._lbl_whisper)
        status_layout.addStretch()
        status_layout.addWidget(self._lbl_ollama)
        page.addWidget(status_frame)

    def _create_workspace_panel(self, index, title_text, hint_text, placeholder):
        panel = QFrame()
        panel.setObjectName("panel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(16, 16, 16, 16)
        panel_layout.setSpacing(12)

        heading = QHBoxLayout()
        heading.setSpacing(10)

        index_label = QLabel(index)
        index_label.setObjectName("panel_index")
        index_label.setFixedWidth(22)
        heading.addWidget(index_label)

        title = QLabel(title_text)
        title.setObjectName("panel_title")
        heading.addWidget(title)
        heading.addStretch()

        hint = QLabel(hint_text)
        hint.setObjectName("panel_hint")
        heading.addWidget(hint)
        panel_layout.addLayout(heading)

        line = QFrame()
        line.setObjectName("separator")
        panel_layout.addWidget(line)

        editor = QTextEdit()
        editor.setObjectName("transcription_editor" if index == "01" else "answer_editor")
        editor.setReadOnly(True)
        editor.setPlaceholderText(placeholder)
        editor.setMinimumHeight(220)
        editor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        panel_layout.addWidget(editor, 1)

        return panel


    def _connect_signals(self):
        self._btn_start.clicked.connect(self._on_start_recording)
        self._btn_stop.clicked.connect(self._on_stop_recording)
        self._btn_suggest.clicked.connect(self._on_suggest_answer)

        self._recorder.recording_started.connect(self._on_recording_started)
        self._recorder.recording_stopped.connect(self._on_recording_stopped)
        self._recorder.error_occurred.connect(self._on_recording_error)
        self._recorder.duration_updated.connect(self._on_duration_update)


    def _check_services(self):

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


        self._lbl_whisper.setText("Whisper: ready (lazy)")
        self._lbl_whisper.setObjectName("indicator_ok")
        self._lbl_whisper.style().unpolish(self._lbl_whisper)
        self._lbl_whisper.style().polish(self._lbl_whisper)


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


    @Slot()
    def _on_start_recording(self):
        self._recorder.start_recording()
        self._start_recording_animation()

    @Slot()
    def _on_stop_recording(self):
        self._recorder.stop_recording()
        self._stop_recording_animation()

    def _start_recording_animation(self):
        self._recording_pulse_animation.setStartValue(1.0)
        self._recording_pulse_animation.setEndValue(0.8)
        self._recording_pulse_animation.setDirection(QPropertyAnimation.Forward)
        self._recording_pulse_animation.start()

    def _stop_recording_animation(self):
        self._recording_pulse_animation.stop()
        self._recording_pulse_effect.setOpacity(1.0)



    @Slot()
    def _on_recording_started(self):
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._lbl_rec_status.setText("Recording in progress")
        self._recording_seconds = 0.0
        self._lbl_duration.setText("00:00")
        self._duration_timer.start()

    @Slot(np.ndarray)
    def _on_recording_stopped(self, audio: np.ndarray):
        self._duration_timer.stop()
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._lbl_rec_status.setText("Ready to record")

        if len(audio) == 0:
            self._txt_transcription.setPlainText("[Empty recording]")
            self._btn_suggest.setEnabled(False)
            return


        self._transcription_worker = TranscriptionWorker(audio, self._stt)
        self._transcription_worker.finished.connect(self._on_transcription_done)
        self._transcription_worker.error.connect(self._on_transcription_error)
        self._transcription_worker.status.connect(self._on_transcription_status)
        self._transcription_worker.start()

    @Slot(str)
    def _on_recording_error(self, msg: str):
        self._stop_recording_animation()
        self._duration_timer.stop()
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._lbl_rec_status.setText("Ready to record")
        self._txt_transcription.setPlainText(f"Error: {msg}")

    @Slot(float)
    def _on_duration_update(self, seconds: float):
        self._recording_seconds = seconds

    def _tick_duration(self):
        mins = int(self._recording_seconds) // 60
        secs = int(self._recording_seconds) % 60
        self._lbl_duration.setText(f"{mins:02d}:{secs:02d}")


    @Slot(str)
    def _on_transcription_done(self, text: str):
        self._txt_transcription.setPlainText(text)
        self._btn_suggest.setEnabled(bool(text.strip()))
        self._lbl_rec_status.setText("Ready to record")

    @Slot(str)
    def _on_transcription_error(self, msg: str):
        self._txt_transcription.setPlainText(f"Transcription error: {msg}")
        self._btn_suggest.setEnabled(False)
        self._lbl_rec_status.setText("Ready to record")

    @Slot(str)
    def _on_transcription_status(self, msg: str):
        self._lbl_rec_status.setText(msg)


    @Slot()
    def _on_suggest_answer(self):
        prompt = self._txt_transcription.toPlainText().strip()
        if not prompt:
            return

        self._btn_suggest.setEnabled(False)
        self._txt_answer.clear()
        self._lbl_gen_status.setText("Generating...")


        self._answer_fade_animation.start()

        self._answer_worker = AnswerWorker(prompt, self._llm)
        self._answer_worker.token_received.connect(self._on_token)
        self._answer_worker.finished.connect(self._on_generation_finished)
        self._answer_worker.error.connect(self._on_generation_error)
        self._answer_worker.start()

    @Slot(str)
    def _on_token(self, token: str):
        self._txt_answer.insertPlainText(token)

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
