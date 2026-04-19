"""Integration test: verifies the full agent pipeline works end-to-end."""
import os
import yaml
import pytest
from unittest.mock import MagicMock

from agent.main import TwinScientist

@pytest.fixture
def project_dir(tmp_path):
    """Create a complete project structure for integration testing."""
    # config
    config = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "context_budget": {"total": 100000, "fixed_zone": 8000,
                           "dynamic_zone": 12000, "conversation_zone": 80000},
        "conversation": {"recent_window": 10, "compression_trigger": 10,
                         "emergency_threshold": 70000},
        "paths": {"persona_dir": "persona", "memory_dir": "memory",
                  "evolution_dir": "evolution"},
    }
    (tmp_path / "config.yaml").write_text(yaml.dump(config))

    # persona
    persona = tmp_path / "persona"
    persona.mkdir()
    (persona / "identity.yaml").write_text(
        "personality_sketch: |\n  说话直接。\ncore_beliefs: []\n"
        "research_focus: []\ntaste_profile: []\n"
    )
    (persona / "thinking_frameworks.yaml").write_text("frameworks: {}\n")
    (persona / "boundaries.yaml").write_text(
        "confident_domains:\n  - 氢能催化剂\n"
        "familiar_but_not_expert: []\noutside_expertise:\n  - 有机合成\n"
    )

    # memory
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "topics").mkdir()
    (memory / "papers").mkdir()
    (memory / "conversations").mkdir()

    topic_index = {
        "topics": {
            "hydrogen_catalyst": {
                "summary": "关注Pt替代和单原子催化",
                "paper_count": 1,
                "stance": "看好但担忧稳定性",
                "detail_files": ["papers/2024-chen.yaml"],
            }
        }
    }
    (memory / "topic_index.yaml").write_text(yaml.dump(topic_index, allow_unicode=True))

    paper = {
        "source": {"title": "SAC for HER", "year": 2024},
        "impression": {"one_sentence": "单原子Pt思路好但稳定性差", "attitude": "skeptical"},
    }
    (memory / "papers" / "2024-chen.yaml").write_text(yaml.dump(paper, allow_unicode=True))

    topic_detail = {"topic": "hydrogen_catalyst", "detailed_stance": "稳定性是瓶颈"}
    (memory / "topics" / "hydrogen_catalyst.yaml").write_text(
        yaml.dump(topic_detail, allow_unicode=True)
    )

    # evolution
    (tmp_path / "evolution").mkdir()

    return str(tmp_path)

def test_system_prompt_assembled(project_dir):
    agent = TwinScientist(project_dir)
    prompt = agent.build_system_prompt()
    assert "说话直接" in prompt
    assert "hydrogen_catalyst" in prompt
    assert "recall" in prompt

def test_recall_tool_works(project_dir):
    agent = TwinScientist(project_dir)
    result = agent._execute_tool("recall", {"topic": "hydrogen_catalyst", "depth": "summary"})
    assert "关注Pt替代" in result

def test_recall_unknown_topic(project_dir):
    agent = TwinScientist(project_dir)
    result = agent._execute_tool("recall", {"topic": "organic_chemistry"})
    assert "没有找到" in result

def test_save_to_memory(project_dir):
    agent = TwinScientist(project_dir)
    result = agent._execute_tool("save_to_memory", {
        "topic": "test_topic",
        "content": "测试内容",
        "source": "test",
    })
    assert "已保存" in result
    # Verify file was created
    conv_dir = os.path.join(project_dir, "memory", "conversations")
    files = os.listdir(conv_dir)
    assert len(files) == 1