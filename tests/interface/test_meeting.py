# tests/interface/test_meeting.py
import os
import pytest
from unittest.mock import MagicMock, patch
from interface.meeting import MeetingBot, run_meeting_from_audio


def _make_bot(tmp_path):
    agent = MagicMock()
    agent.project_dir = str(tmp_path)
    agent.config = {"model": "test-model"}
    agent.chat = MagicMock(return_value="这个方向值得关注，稳定性是核心问题。")

    bot = MeetingBot(
        agent=agent,
        twin_name="张三",
        confident_domains=["氢能催化剂", "电解水"],
        voice="zh-CN-YunxiNeural",
    )
    return bot


def test_meeting_bot_init(tmp_path):
    bot = _make_bot(tmp_path)
    assert bot.twin_name == "张三"
    assert bot.meeting_context is not None
    assert bot.trigger_detector is not None


def test_process_utterance_no_trigger(tmp_path):
    """Utterance outside domain should not generate response."""
    bot = _make_bot(tmp_path)
    result = bot.process_utterance(
        speaker="李四",
        text="今天天气不错。",
    )
    assert result is None


def test_process_utterance_direct_mention(tmp_path):
    """Direct mention should generate response."""
    bot = _make_bot(tmp_path)
    with patch.object(bot, "_speak", return_value="audio.mp3") as mock_speak:
        result = bot.process_utterance(
            speaker="李四",
            text="张三你怎么看单原子催化剂的稳定性问题？",
        )
    assert result is not None
    mock_speak.assert_called_once()
    bot.agent.chat.assert_called_once()


def test_process_utterance_adds_to_context(tmp_path):
    """All utterances should be added to meeting context."""
    bot = _make_bot(tmp_path)
    bot.process_utterance("李四", "最近HER催化剂有什么新进展？")
    assert len(bot.meeting_context.recent_utterances) == 1


def test_run_meeting_from_audio(tmp_path):
    """run_meeting_from_audio should process a transcript file."""
    transcript_path = str(tmp_path / "transcript.txt")
    with open(transcript_path, "w") as f:
        f.write("李四：请问张三老师，氢能催化剂稳定性的问题怎么看？\n")
        f.write("王五：另外单原子催化的成本控制有什么思路？\n")

    agent = MagicMock()
    agent.project_dir = str(tmp_path)
    agent.config = {"model": "test-model"}
    agent.chat = MagicMock(return_value="稳定性是核心问题，1000圈是门槛。")

    responses = []
    with patch("interface.meeting.synthesize_speech") as mock_tts:
        mock_tts.return_value = "response.mp3"
        responses = run_meeting_from_audio(
            agent=agent,
            transcript_path=transcript_path,
            twin_name="张三",
            confident_domains=["氢能催化剂", "单原子催化"],
            auto_play=False,
        )

    assert len(responses) >= 1
    assert any("稳定性" in r["response"] for r in responses)