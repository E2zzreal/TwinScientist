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

import os
import yaml as _yaml
from unittest.mock import patch

def test_session_summary_saved_on_end(tmp_path):
    project_dir = _make_project(tmp_path)
    agent = TwinScientist(project_dir)
    agent.context.add_turn("氢能催化剂有什么进展？", "单原子Pt是个好方向，但稳定性差。")

    # Mock the LLM compressor to avoid real API calls
    with patch.object(agent, '_llm_compress', return_value="摘要：讨论了氢能催化剂稳定性问题。"):
        agent.end_session()

    session_dir = os.path.join(project_dir, "memory", "conversations")
    files = [f for f in os.listdir(session_dir) if f.startswith("session-")]
    assert len(files) == 1


def test_previous_session_loaded_on_init(tmp_path):
    project_dir = _make_project(tmp_path)

    # Write a fake previous session summary
    session_dir = os.path.join(project_dir, "memory", "conversations")
    os.makedirs(session_dir, exist_ok=True)
    summary_path = os.path.join(session_dir, "session-20250101-120000.yaml")
    with open(summary_path, "w") as f:
        _yaml.dump({
            "type": "session_summary",
            "summary": "上次讨论了单原子催化稳定性问题。",
            "timestamp": "20250101-120000",
        }, f, allow_unicode=True)

    agent = TwinScientist(project_dir)
    # Previous summary should be loaded into context
    assert "上次讨论" in agent.context._summary


def test_give_feedback_style_correction(tmp_path):
    """give_feedback with 'not like me' should trigger style correction."""
    project_dir = _make_project(tmp_path)
    agent = TwinScientist(project_dir)

    mock_client_response = MagicMock()
    mock_client_response.content = [MagicMock(text="""
context: "评价实验"
bad: "这个研究有价值"
good: "你看数据，100圈就衰减了"
note: "锚定数据"
""")]
    agent.client.messages.create = MagicMock(return_value=mock_client_response)

    result = agent._execute_tool("give_feedback", {
        "feedback_type": "style",
        "feedback": "不像我，我会直接说数据",
        "original_response": "这个研究有价值",
        "context": "讨论稳定性",
    })

    assert "已记录" in result or "风格" in result

    # Check changelog was written
    changelog_path = os.path.join(project_dir, "evolution", "changelog.yaml")
    with open(changelog_path) as f:
        data = _yaml.safe_load(f)
    assert len(data["changes"]) >= 1


def test_give_feedback_stance_update(tmp_path):
    """give_feedback with stance update should modify topic_index."""
    project_dir = _make_project(tmp_path)
    agent = TwinScientist(project_dir)

    result = agent._execute_tool("give_feedback", {
        "feedback_type": "stance",
        "topic": "hydrogen_catalyst",
        "new_stance": "对产业化更谨慎了",
        "reason": "成本太高",
    })

    assert "已更新" in result or "立场" in result