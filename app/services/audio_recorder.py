"""Microphone audio recorder using sounddevice."""

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QObject, Signal, QThread, Slot

from app.utils.config import SAMPLE_RATE, CHANNELS, DTYPE


class _RecordingThread(QThread):
    """Background thread that reads audio chunks from the input stream."""

    chunk_ready = Signal(np.ndarray)
    error_occurred = Signal(str)

    def __init__(self, sample_rate: int, channels: int, dtype: str, parent=None):
        super().__init__(parent)
        self._sample_rate = sample_rate
        self._channels = channels
        self._dtype = dtype
        self._running = False
        self._stream = None

    def run(self):
        try:
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype=self._dtype,
            )
            self._stream.start()
            self._running = True
            while self._running:
                data, overflowed = self._stream.read(1024)
                if overflowed:
                    pass  # minor overflow is acceptable
                if data is not None and len(data) > 0:
                    self.chunk_ready.emit(data.copy())
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass

    def stop(self):
        self._running = False


class AudioRecorder(QObject):
    """Records microphone audio in a background thread."""

    recording_started = Signal()
    recording_stopped = Signal(np.ndarray)
    error_occurred = Signal(str)
    duration_updated = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_recording = False
        self._chunks: list[np.ndarray] = []
        self._thread: _RecordingThread | None = None
        self._total_samples = 0

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @Slot()
    def start_recording(self):
        if self._is_recording:
            return
        self._chunks.clear()
        self._total_samples = 0
        self._is_recording = True

        self._thread = _RecordingThread(SAMPLE_RATE, CHANNELS, DTYPE)
        self._thread.chunk_ready.connect(self._on_chunk)
        self._thread.error_occurred.connect(self._on_error)
        self._thread.start()
        self.recording_started.emit()

    @Slot()
    def stop_recording(self):
        if not self._is_recording:
            return
        self._is_recording = False
        if self._thread is not None:
            self._thread.stop()
            self._thread.wait(3000)
            self._thread = None

        if self._chunks:
            audio = np.concatenate(self._chunks, axis=0)
            # Flatten to 1D mono
            if audio.ndim > 1:
                audio = audio[:, 0]
        else:
            audio = np.array([], dtype=np.float32)

        self.recording_stopped.emit(audio)

    def _on_chunk(self, data: np.ndarray):
        self._chunks.append(data)
        self._total_samples += len(data)
        duration = self._total_samples / SAMPLE_RATE
        self.duration_updated.emit(duration)

    def _on_error(self, msg: str):
        self._is_recording = False
        self.error_occurred.emit(f"Recording error: {msg}")
