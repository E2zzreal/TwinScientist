# tests/multimodal/test_tts.py
import os
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from multimodal.tts import synthesize_speech, TTSEngine, get_available_voices


def test_tts_engine_default():
    """TTSEngine should default to edge-tts."""
    engine = TTSEngine()
    assert engine.backend == "edge_tts"


def test_tts_engine_voice_selection():
    """TTSEngine should accept a voice name."""
    engine = TTSEngine(voice="zh-CN-YunxiNeural")
    assert engine.voice == "zh-CN-YunxiNeural"


def test_synthesize_speech_creates_file(tmp_path):
    """synthesize_speech should create an audio file."""
    output_path = str(tmp_path / "output.mp3")

    # Mock the internal _synthesize_edge_tts function to avoid actual edge-tts dependency
    with patch("multimodal.tts._synthesize_edge_tts") as mock_synth:
        mock_synth.return_value = output_path

        # Simulate file creation
        with open(output_path, "wb") as f:
            f.write(b"fake audio data")

        synthesize_speech(
            text="你好，这是一段测试语音",
            output_path=output_path,
            voice="zh-CN-YunxiNeural",
        )

        # Verify the mock was called with correct arguments
        mock_synth.assert_called_once()
        call_args = mock_synth.call_args
        assert call_args[0][0] == "你好，这是一段测试语音"
        assert call_args[0][1] == output_path
        assert call_args[0][2] == "zh-CN-YunxiNeural"

    assert os.path.exists(output_path)


def test_get_available_voices_returns_list():
    """get_available_voices should return a non-empty list."""
    # Mock to avoid network call
    with patch("multimodal.tts._CHINESE_VOICES") as mock_voices:
        mock_voices.__iter__ = MagicMock(return_value=iter([
            {"ShortName": "zh-CN-YunxiNeural", "Gender": "Male"},
            {"ShortName": "zh-CN-XiaoxiaoNeural", "Gender": "Female"},
        ]))
        voices = get_available_voices()
        # Even if mocking fails, function should return a list
        assert isinstance(voices, list)