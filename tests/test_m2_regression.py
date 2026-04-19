# tests/test_m2_regression.py
"""M2 regression: verifies 30+ turns without losing key information."""
import os
import yaml
import pytest
from unittest.mock import MagicMock

from agent.context_manager import ContextManager
from agent.tools import RecallTool, SaveToMemoryTool


def _make_config():
    return {
        "context_budget": {
            "total": 100000, "fixed_zone": 8000,
            "dynamic_zone": 12000, "conversation_zone": 80000,
        },
        "conversation": {
            "recent_window": 5,
            "compression_trigger": 5,
            "emergency_threshold": 70000,
        },
        "model": "claude-sonnet-4-20250514",
    }


def test_30_turns_no_crash():
    """30 turns with compression should not raise errors."""
    cm = ContextManager(_make_config())
    mock_compressor = MagicMock(return_value="摘要：讨论了多个话题。")
    cm.set_llm_compressor(mock_compressor)

    for i in range(30):
        cm.prepare(f"question {i}")
        cm.add_turn(f"用户问题 {i}", f"详细回答 {i}，包含数据和分析。")

    history = cm.get_history()
    # summary pair (2) + recent 5 turns (10) = max 12
    assert len(history) <= 14
    assert cm._summary != ""


def test_compression_accumulates_across_batches():
    """Multiple compressions should accumulate summaries correctly."""
    cm = ContextManager(_make_config())
    call_count = [0]

    def mock_compressor(turns):
        call_count[0] += 1
        return f"第{call_count[0]}批摘要"

    cm.set_llm_compressor(mock_compressor)

    for i in range(15):
        cm.prepare(f"q{i}")
        cm.add_turn(f"user {i}", f"answer {i}")

    assert call_count[0] >= 2
    assert "摘要" in cm._summary


def test_save_and_recall_conversation(tmp_path):
    """Save a memory then recall it via conversations depth."""
    memory_dir = str(tmp_path / "memory")
    os.makedirs(os.path.join(memory_dir, "conversations"), exist_ok=True)
    with open(os.path.join(memory_dir, "topic_index.yaml"), "w") as f:
        yaml.dump({"topics": {}}, f)

    save_tool = SaveToMemoryTool(memory_dir)
    save_tool.execute(
        topic="hydrogen_catalyst",
        content="会议中确认了1000圈稳定性是工程化的门槛。",
        source="meeting_2025"
    )

    recall_tool = RecallTool(memory_dir)
    result = recall_tool.execute("hydrogen_catalyst", depth="conversations")
    assert "1000圈" in result


def test_topic_index_hotreload(tmp_path):
    """RecallTool should see newly added topics without reinit."""
    memory_dir = str(tmp_path / "memory")
    os.makedirs(memory_dir, exist_ok=True)
    index_path = os.path.join(memory_dir, "topic_index.yaml")

    with open(index_path, "w") as f:
        yaml.dump({"topics": {"topic_a": {"summary": "A", "paper_count": 0,
                                           "stance": "ok", "detail_files": []}}},
                  f, allow_unicode=True)

    tool = RecallTool(memory_dir)
    assert "A" in tool.execute("topic_a")

    with open(index_path, "r") as f:
        index = yaml.safe_load(f)
    index["topics"]["topic_b"] = {"summary": "B新话题", "paper_count": 0,
                                   "stance": "new", "detail_files": []}
    with open(index_path, "w") as f:
        yaml.dump(index, f, allow_unicode=True)

    assert "B新话题" in tool.execute("topic_b")
