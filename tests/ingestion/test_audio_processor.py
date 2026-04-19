# tests/ingestion/test_audio_processor.py
import os
import yaml
import pytest
from unittest.mock import MagicMock, patch
from ingestion.audio_processor import transcribe_audio, process_audio


def test_transcribe_audio_without_whisper():
    """If whisper not installed, should raise ImportError with helpful message."""
    with patch.dict("sys.modules", {"whisper": None}):
        with pytest.raises((ImportError, ModuleNotFoundError)):
            transcribe_audio("/fake/audio.mp3")


def test_process_audio_from_existing_transcript(tmp_path):
    """process_audio should accept pre-existing transcript text."""
    persona_dir = str(tmp_path / "persona")
    os.makedirs(persona_dir, exist_ok=True)
    style_path = os.path.join(persona_dir, "style.yaml")
    with open(style_path, "w") as f:
        yaml.dump({"voice": {"summary": "", "exemplars": []}}, f,
                  allow_unicode=True)

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="""
exemplars:
  - context: "评价实验"
    good: "你看这个数据，100圈就衰减了15%"
    note: "锚定数据"
""")]
    mock_client.messages.create.return_value = mock_response

    result = process_audio(
        client=mock_client,
        model="claude-sonnet-4-20250514",
        persona_dir=str(persona_dir),
        speaker="张三",
        transcript_text="[张三] 你看这个数据，100圈就衰减了15%，离实用差太远了。",
    )

    assert result["exemplars_added"] >= 0
    with open(style_path, "r") as f:
        style = yaml.safe_load(f)
    assert "exemplars" in style["voice"]


def test_process_audio_saves_transcript(tmp_path):
    """process_audio should save transcript to file if provided."""
    persona_dir = str(tmp_path / "persona")
    os.makedirs(persona_dir, exist_ok=True)
    with open(os.path.join(persona_dir, "style.yaml"), "w") as f:
        yaml.dump({"voice": {"summary": "", "exemplars": []}}, f)

    transcript_save_path = str(tmp_path / "transcript.txt")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="exemplars: []")]
    mock_client.messages.create.return_value = mock_response

    process_audio(
        client=mock_client,
        model="claude-sonnet-4-20250514",
        persona_dir=str(persona_dir),
        speaker="张三",
        transcript_text="some transcript",
        save_transcript_to=transcript_save_path,
    )

    assert os.path.exists(transcript_save_path)