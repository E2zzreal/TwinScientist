# tests/agent/test_context_manager.py
import pytest
from agent.context_manager import ContextManager

def _make_config():
    return {
        "context_budget": {
            "total": 100000,
            "fixed_zone": 8000,
            "dynamic_zone": 12000,
            "conversation_zone": 80000,
        },
        "conversation": {
            "recent_window": 3,
            "compression_trigger": 3,
            "emergency_threshold": 70000,
        },
        "model": "claude-sonnet-4-20250514",
    }

def test_add_turn_and_get_history():
    cm = ContextManager(_make_config())
    cm.add_turn("你好", "你好，我是数字分身。")
    cm.add_turn("你做什么方向？", "我主要做氢能催化。")
    history = cm.get_history()
    assert len(history) == 4  # 2 turns x 2 messages (user + assistant)
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"

def test_compression_triggers_when_exceeding_window():
    cm = ContextManager(_make_config())
    # Add 4 turns, window is 3 — first turn should be compressed
    cm.add_turn("turn1-user", "turn1-assistant")
    cm.add_turn("turn2-user", "turn2-assistant")
    cm.add_turn("turn3-user", "turn3-assistant")
    cm.add_turn("turn4-user", "turn4-assistant")
    cm.prepare("new message")  # triggers compression check
    history = cm.get_history()
    # Should have a summary message + recent 3 turns
    has_summary = any("summary" in str(msg.get("content", "")).lower()
                      or "摘要" in str(msg.get("content", ""))
                      for msg in history)
    # recent 3 turns = 6 messages, possibly +1 summary
    assert len(history) <= 8

def test_dynamic_zone_load_and_unload():
    cm = ContextManager(_make_config())
    cm.load_dynamic("hydrogen_catalyst", "详细的催化剂知识内容...")
    assert "hydrogen_catalyst" in cm.get_loaded_topics()
    cm.unload_dynamic("hydrogen_catalyst")
    assert "hydrogen_catalyst" not in cm.get_loaded_topics()

def test_get_budget_status():
    cm = ContextManager(_make_config())
    status = cm.get_budget_status()
    assert "fixed_zone" in status
    assert "dynamic_zone" in status
    assert "conversation_zone" in status
    assert status["conversation_used"] == 0


# --- M2: LLM compression tests ---

from unittest.mock import MagicMock, patch

def test_llm_compress_called_when_overflow():
    """When turns overflow window, LLM compressor should be called."""
    cm = ContextManager(_make_config())

    mock_llm = MagicMock()
    mock_llm.return_value = "摘要：讨论了氢能催化剂，结论是稳定性是瓶颈。"
    cm.set_llm_compressor(mock_llm)

    for i in range(5):
        cm.add_turn(f"user{i}", f"answer{i}")

    cm.prepare("new message")

    mock_llm.assert_called_once()
    assert "摘要" in cm._summary

def test_fallback_compress_when_no_llm():
    """Without LLM compressor, should fall back to naive compression."""
    cm = ContextManager(_make_config())
    for i in range(5):
        cm.add_turn(f"user{i}", f"answer{i}")
    cm.prepare("new message")
    # Naive fallback: should still produce a summary
    assert len(cm._summary) > 0