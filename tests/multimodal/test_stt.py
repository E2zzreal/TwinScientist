# tests/multimodal/test_stt.py
import os
import pytest
from unittest.mock import MagicMock, patch
from multimodal.stt import transcribe_file, record_once, STTEngine


def test_stt_engine_init():
    """STTEngine should initialize with model name."""
    engine = STTEngine(model_size="base")
    assert engine.model_size == "base"
    assert engine.model is None  # lazy load


def test_transcribe_file_without_whisper():
    """transcribe_file should raise ImportError if whisper not installed."""
    with patch.dict("sys.modules", {"whisper": None}):
        with pytest.raises((ImportError, ModuleNotFoundError)):
            transcribe_file("/fake/audio.wav")


def test_transcribe_file_calls_whisper(tmp_path):
    """transcribe_file should call whisper model.transcribe."""
    # Create a fake audio file
    audio_path = str(tmp_path / "test.wav")
    with open(audio_path, "wb") as f:
        f.write(b"\x00" * 100)

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "text": "这是一段测试语音转写",
        "language": "zh",
    }

    with patch("multimodal.stt.whisper") as mock_whisper, \
         patch("multimodal.stt.HAS_WHISPER", True):
        mock_whisper.load_model.return_value = mock_model
        result = transcribe_file(audio_path, model_size="base")

    assert result == "这是一段测试语音转写"
    mock_model.transcribe.assert_called_once()


def test_record_once_without_sounddevice():
    """record_once should raise ImportError if sounddevice not installed."""
    with patch.dict("sys.modules", {"sounddevice": None}):
        with pytest.raises((ImportError, ModuleNotFoundError)):
            record_once(duration=1)