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