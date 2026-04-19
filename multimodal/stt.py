# multimodal/stt.py
"""Speech-to-Text module wrapping Whisper with real-time microphone support."""
import os
import tempfile

try:
    import whisper as _whisper_mod
    whisper = _whisper_mod
    HAS_WHISPER = True
except ImportError:
    whisper = None
    HAS_WHISPER = False

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None
    HAS_SOUNDDEVICE = False


class STTEngine:
    """Speech-to-Text engine backed by Whisper."""

    def __init__(self, model_size: str = "base", language: str = "zh"):
        self.model_size = model_size
        self.language = language
        self.model = None  # lazy-loaded

    def _load_model(self):
        if not HAS_WHISPER:
            raise ImportError(
                "openai-whisper not installed.\n"
                "Run: pip install openai-whisper"
            )
        if self.model is None:
            self.model = whisper.load_model(self.model_size)
        return self.model

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file to text."""
        model = self._load_model()
        result = model.transcribe(audio_path, language=self.language)
        return result["text"].strip()

    def record_and_transcribe(self, duration: float = 5.0) -> str:
        """Record from microphone for `duration` seconds and transcribe."""
        audio_path = record_once(duration=duration)
        try:
            return self.transcribe(audio_path)
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)


def transcribe_file(audio_path: str, model_size: str = "base",
                    language: str = "zh") -> str:
    """Transcribe an audio file. Convenience function."""
    engine = STTEngine(model_size=model_size, language=language)
    return engine.transcribe(audio_path)


def record_once(duration: float = 5.0, samplerate: int = 16000) -> str:
    """Record audio from microphone and save to a temp file.

    Returns:
        Path to temporary WAV file. Caller must delete it.
    """
    if not HAS_SOUNDDEVICE:
        raise ImportError(
            "sounddevice not installed.\n"
            "Run: pip install sounddevice soundfile"
        )
    try:
        import soundfile as sf
        import numpy as np
    except ImportError:
        raise ImportError("soundfile and numpy required. Run: pip install soundfile numpy")

    print(f"  🎤 录音中（{duration:.0f}秒）... 请说话")
    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("  ✓ 录音完成")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, samplerate)
    return tmp.name