# tests/interface/test_voice.py
import os
import pytest
from unittest.mock import MagicMock, patch
from interface.voice import VoiceSession, process_voice_file


def _make_agent(tmp_path):
    """Create a minimal mock agent."""
    agent = MagicMock()
    agent.project_dir = str(tmp_path)
    agent.config = {"model": "test-model"}
    agent.chat = MagicMock(return_value="这是分身的回答，稳定性是核心问题。")
    return agent


def test_voice_session_init(tmp_path):
    agent = _make_agent(tmp_path)
    session = VoiceSession(
        agent=agent,
        voice="zh-CN-YunxiNeural",
        stt_model="base",
    )
    assert session.voice == "zh-CN-YunxiNeural"
    assert session.stt_model == "base"


def test_process_voice_file(tmp_path):
    """process_voice_file: audio → transcribe → agent → TTS → output file."""
    agent = _make_agent(tmp_path)

    audio_in = str(tmp_path / "input.wav")
    audio_out = str(tmp_path / "output.mp3")

    with open(audio_in, "wb") as f:
        f.write(b"\x00" * 100)

    with patch("interface.voice.transcribe_file", return_value="你对稳定性有什么看法？") as mock_stt, \
         patch("interface.voice.synthesize_speech") as mock_tts:
        # Simulate TTS creating the output file
        mock_tts.side_effect = lambda text, output_path, **kw: (
            open(output_path, "wb").write(b"audio") or output_path
        )

        result = process_voice_file(
            agent=agent,
            audio_input_path=audio_in,
            audio_output_path=audio_out,
            voice="zh-CN-YunxiNeural",
        )

    mock_stt.assert_called_once_with(audio_in, model_size="base")
    agent.chat.assert_called_once_with("你对稳定性有什么看法？")
    assert "稳定性" in agent.chat.return_value


def test_voice_session_text_mode(tmp_path):
    """VoiceSession.respond_text: text → TTS → audio file."""
    agent = _make_agent(tmp_path)
    session = VoiceSession(agent=agent)
    audio_out = str(tmp_path / "response.mp3")

    with patch("interface.voice.synthesize_speech") as mock_tts:
        mock_tts.side_effect = lambda text, output_path, **kw: (
            open(output_path, "wb").write(b"audio") or output_path
        )
        session.respond_text("这个方向值得关注。", output_path=audio_out)

    mock_tts.assert_called_once()