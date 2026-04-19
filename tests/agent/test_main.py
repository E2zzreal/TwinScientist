# tests/agent/test_main.py
import pytest
from unittest.mock import MagicMock, patch
from agent.main import TwinScientist

def _make_project(tmp_path):
    """Create minimal project structure for testing."""
    # config
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
model: claude-sonnet-4-20250514
max_tokens: 4096
context_budget:
  total: 100000
  fixed_zone: 8000
  dynamic_zone: 12000
  conversation_zone: 80000
conversation:
  recent_window: 10
  compression_trigger: 10
  emergency_threshold: 70000
paths:
  persona_dir: persona
  memory_dir: memory
  evolution_dir: evolution
""")
    # persona
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "identity.yaml").write_text(
        "personality_sketch: |\n  说话直接\ncore_beliefs: []\n"
    )
    (persona_dir / "thinking_frameworks.yaml").write_text("frameworks: {}\n")
    (persona_dir / "boundaries.yaml").write_text(
        "confident_domains: []\nfamiliar_but_not_expert: []\noutside_expertise: []\n"
    )
    # memory
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "topic_index.yaml").write_text("topics: {}\n")
    (memory_dir / "topics").mkdir()
    (memory_dir / "papers").mkdir()
    (memory_dir / "conversations").mkdir()
    return str(tmp_path)

def test_twin_scientist_init(tmp_path):
    project_dir = _make_project(tmp_path)
    agent = TwinScientist(project_dir)
    assert agent is not None
    assert agent.config["model"] == "claude-sonnet-4-20250514"

@patch("agent.main.anthropic")
def test_twin_scientist_chat_simple(mock_anthropic, tmp_path):
    project_dir = _make_project(tmp_path)
    agent = TwinScientist(project_dir)

    # Mock the API response
    mock_response = MagicMock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [MagicMock(text="你好，我是数字分身。", type="text")]
    agent.client.messages.create = MagicMock(return_value=mock_response)

    answer = agent.chat("你好")
    assert "数字分身" in answer

@patch("agent.main.anthropic")
def test_twin_scientist_chat_with_tool_call(mock_anthropic, tmp_path):
    project_dir = _make_project(tmp_path)
    agent = TwinScientist(project_dir)

    # First response: tool call
    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.id = "tool_1"
    tool_use_block.name = "recall"
    tool_use_block.input = {"topic": "hydrogen_catalyst", "depth": "summary"}

    first_response = MagicMock()
    first_response.stop_reason = "tool_use"
    first_response.content = [tool_use_block]

    # Second response: final text
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "氢能催化是个好方向。"

    second_response = MagicMock()
    second_response.stop_reason = "end_turn"
    second_response.content = [text_block]

    agent.client.messages.create = MagicMock(
        side_effect=[first_response, second_response]
    )

    answer = agent.chat("你怎么看氢能？")
    assert agent.client.messages.create.call_count == 2