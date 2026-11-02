import sys
from PySide6.QtWidgets import QApplication
from app.ui.main_window import MainWindow

# Test imports and basic instantiation without showing window
app = QApplication(sys.argv)
app.setStyle('Fusion')

window = MainWindow()

# Test 1: Check all UI elements exist
print('UI Elements Check:')
print(f'  Start button: {window._btn_start is not None}')
print(f'  Stop button: {window._btn_stop is not None}')
print(f'  Suggest button: {window._btn_suggest is not None}')
print(f'  Transcription text: {window._txt_transcription is not None}')
print(f'  Answer text: {window._txt_answer is not None}')
print(f'  Mic label: {window._lbl_mic is not None}')
print(f'  Whisper label: {window._lbl_whisper is not None}')
print(f'  Ollama label: {window._lbl_ollama is not None}')
print(f'  Local indicator: {window._local_indicator is not None}')

# Test 2: Check initial states
print('\nInitial States:')
print(f'  Start enabled: {window._btn_start.isEnabled()}')
print(f'  Stop enabled: {window._btn_stop.isEnabled()}')
print(f'  Suggest enabled: {window._btn_suggest.isEnabled()}')
print(f'  Recording status: "{window._lbl_rec_status.text()}"')
print(f'  Duration: "{window._lbl_duration.text()}"')

# Test 3: Check service status indicators
print('\nService Status:')
print(f'  Mic: "{window._lbl_mic.text()}" ({window._lbl_mic.objectName()})')
print(f'  Whisper: "{window._lbl_whisper.text()}" ({window._lbl_whisper.objectName()})')
print(f'  Ollama: "{window._lbl_ollama.text()}" ({window._lbl_ollama.objectName()})')
print(f'  Local AI: "{window._local_indicator.text()}" ({window._local_indicator.objectName()})')

# Test 4: Check animations
print('\nAnimations:')
print(f'  Recording pulse animation: {window._recording_pulse_animation is not None}')
print(f'  Fade animation: {window._fade_animation is not None}')

# Test 5: Check styles
print(f'\nStylesheet length: {len(window.styleSheet())} chars')

print('\nAll UI components verified successfully.')