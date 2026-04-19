# tests/interface/test_meeting_context.py
import pytest
from unittest.mock import MagicMock
from interface.meeting_context import MeetingContext


def _make_context(window_minutes=1, max_tokens=500):
    return MeetingContext(
        window_minutes=window_minutes,
        max_tokens=max_tokens,
    )


def test_meeting_context_init():
    ctx = _make_context()
    assert ctx.topic == ""
    assert ctx.participants == []
    assert len(ctx.recent_utterances) == 0
    assert ctx.rolling_summary == ""


def test_add_utterance_stores_entry():
    ctx = _make_context()
    ctx.add_utterance(speaker="张三", text="氢能催化剂的稳定性还有很大提升空间。")
    assert len(ctx.recent_utterances) == 1
    assert ctx.recent_utterances[0]["speaker"] == "张三"
    assert "稳定性" in ctx.recent_utterances[0]["text"]


def test_get_snapshot_returns_dict():
    ctx = _make_context()
    ctx.topic = "氢能催化剂进展"
    ctx.participants = ["张三", "李四"]
    ctx.add_utterance("张三", "单原子催化是个好方向。")
    snapshot = ctx.get_snapshot()
    assert "topic" in snapshot
    assert "participants" in snapshot
    assert "recent_text" in snapshot
    assert "单原子" in snapshot["recent_text"]


def test_compress_old_utterances_on_overflow(monkeypatch):
    """When recent_utterances token count exceeds budget, should compress."""
    ctx = _make_context(max_tokens=100)

    compress_called = [False]
    original_compress = ctx._compress_old_utterances
    def mock_compress():
        compress_called[0] = True
        original_compress()
    ctx._compress_old_utterances = mock_compress

    # Add many utterances to trigger compression
    for i in range(20):
        ctx.add_utterance("张三", f"这是第{i}句话，内容比较长，包含很多信息。")

    # Either compress was called or tokens stayed within budget
    tokens = ctx._count_recent_tokens()
    assert tokens <= ctx.max_tokens * 2 or compress_called[0]


def test_format_for_agent_injection():
    """Format output should be ready for injection into system prompt."""
    ctx = _make_context()
    ctx.topic = "测试会议"
    ctx.participants = ["张三"]
    ctx.add_utterance("张三", "请问单原子催化的稳定性问题怎么解决？")
    formatted = ctx.format_for_agent()
    assert "会议" in formatted or "topic" in formatted.lower() or "测试" in formatted
    assert "张三" in formatted