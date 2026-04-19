# tests/agent/test_tools.py
import pytest
import yaml
from agent.tools import RecallTool

def _make_memory(tmp_path):
    """Create a test memory directory with sample data."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()

    # topic_index
    index = {
        "topics": {
            "hydrogen_catalyst": {
                "summary": "关注Pt替代和单原子催化",
                "paper_count": 2,
                "stance": "看好但担忧稳定性",
                "detail_files": [
                    "papers/2024-chen.yaml",
                    "papers/2023-wang.yaml",
                ],
            }
        }
    }
    (memory_dir / "topic_index.yaml").write_text(
        yaml.dump(index, allow_unicode=True)
    )

    # topic detail
    topics_dir = memory_dir / "topics"
    topics_dir.mkdir()
    topic_detail = {
        "topic": "hydrogen_catalyst",
        "detailed_stance": "单原子催化降低Pt用量是正确方向，但稳定性是核心瓶颈。",
        "key_debates": ["Pt用量 vs 稳定性权衡", "MOF衍生碳载体的可行性"],
    }
    (topics_dir / "hydrogen_catalyst.yaml").write_text(
        yaml.dump(topic_detail, allow_unicode=True)
    )

    # paper detail
    papers_dir = memory_dir / "papers"
    papers_dir.mkdir()
    paper = {
        "source": {"title": "Single-atom Pt on MOF-derived carbon", "year": 2024},
        "impression": {
            "one_sentence": "思路有意思但稳定性不够",
            "attitude": "skeptical_but_interested",
        },
    }
    (papers_dir / "2024-chen.yaml").write_text(
        yaml.dump(paper, allow_unicode=True)
    )
    return str(memory_dir)


def test_recall_summary(tmp_path):
    memory_dir = _make_memory(tmp_path)
    tool = RecallTool(memory_dir)
    result = tool.execute("hydrogen_catalyst", depth="summary")
    assert "关注Pt替代" in result
    assert "看好但担忧稳定性" in result


def test_recall_detail(tmp_path):
    memory_dir = _make_memory(tmp_path)
    tool = RecallTool(memory_dir)
    result = tool.execute("hydrogen_catalyst", depth="detail")
    assert "单原子催化降低Pt用量" in result


def test_recall_specific_paper(tmp_path):
    memory_dir = _make_memory(tmp_path)
    tool = RecallTool(memory_dir)
    result = tool.execute("hydrogen_catalyst", depth="specific_paper",
                          paper_id="2024-chen")
    assert "思路有意思但稳定性不够" in result


def test_recall_unknown_topic(tmp_path):
    memory_dir = _make_memory(tmp_path)
    tool = RecallTool(memory_dir)
    result = tool.execute("quantum_computing", depth="summary")
    assert "没有找到" in result or "不在" in result


# Task 2: New tests for conversation memory search and hot-reload

import os
import yaml as _yaml
from datetime import datetime


def _add_conversation_memory(memory_dir, topic, content):
    """Helper: create a conversation memory file."""
    conv_dir = os.path.join(memory_dir, "conversations")
    os.makedirs(conv_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(conv_dir, f"{ts}-{topic}.yaml")
    with open(path, "w") as f:
        _yaml.dump({"topic": topic, "content": content, "timestamp": ts}, f,
                   allow_unicode=True)


def test_recall_conversations(tmp_path):
    memory_dir = _make_memory(tmp_path)
    _add_conversation_memory(
        str(memory_dir),
        "hydrogen_catalyst",
        "会议中讨论了稳定性问题，认为1000圈以上才算实用。"
    )
    tool = RecallTool(str(memory_dir))
    result = tool.execute("hydrogen_catalyst", depth="conversations")
    assert "稳定性" in result or "1000圈" in result


def test_recall_topic_index_hotreload(tmp_path):
    """topic_index should be reloaded on each execute call."""
    memory_dir = _make_memory(tmp_path)
    tool = RecallTool(str(memory_dir))

    # Add a new topic to the index after init
    index_path = os.path.join(str(memory_dir), "topic_index.yaml")
    with open(index_path, "r") as f:
        index = _yaml.safe_load(f)
    index["topics"]["new_topic"] = {
        "summary": "新话题摘要",
        "paper_count": 0,
        "stance": "刚开始了解",
        "detail_files": [],
    }
    with open(index_path, "w") as f:
        _yaml.dump(index, f, allow_unicode=True)

    # Should find the new topic without reinitializing
    result = tool.execute("new_topic", depth="summary")
    assert "新话题摘要" in result