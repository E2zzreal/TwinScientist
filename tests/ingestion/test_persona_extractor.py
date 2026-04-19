# tests/ingestion/test_persona_extractor.py
import os
import yaml
import pytest
from unittest.mock import MagicMock
from ingestion.persona_extractor import (
    extract_style_exemplars,
    extract_verbal_habits,
    extract_reasoning_patterns,
    merge_into_persona,
)

SAMPLE_TRANSCRIPT = """
[张三发言]
这个方向我觉得有意思，但你看这个稳定性数据，100圈就衰减了15%，
离实用差太远了。你跟商业化的Pt/C比过没有？
[李四发言]
我们还没有做这个对比实验。
[张三发言]
那这个就说不清楚了。想法是好的，但control experiment要做干净。
你看Fig.3，没有跟bulk材料对比，怎么证明是single atom的贡献？
"""


def test_extract_style_exemplars_returns_list():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="""
exemplars:
  - context: "评价实验数据"
    good: "你看这个稳定性数据，100圈就衰减了15%，离实用差太远了。"
    note: "锚定具体数据，不说空话"
  - context: "质疑实验设计"
    good: "control experiment要做干净，没有跟bulk对比，怎么证明是single atom的贡献？"
    note: "精确指出实验缺陷"
""")]
    mock_client.messages.create.return_value = mock_response

    result = extract_style_exemplars(
        client=mock_client,
        model="claude-sonnet-4-20250514",
        transcript=SAMPLE_TRANSCRIPT,
        speaker="张三",
    )
    assert isinstance(result, list)
    assert len(result) > 0
    assert "context" in result[0]
    assert "good" in result[0]


def test_extract_verbal_habits_returns_dict():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="""
verbal_habits:
  high_frequency_phrases:
    - "你看这个数据"
    - "离实用差太远"
  sentence_style: "短句为主，直接给结论"
  language_mix: "术语用英文（single atom, control experiment），论述用中文"
""")]
    mock_client.messages.create.return_value = mock_response

    result = extract_verbal_habits(
        client=mock_client,
        model="claude-sonnet-4-20250514",
        transcript=SAMPLE_TRANSCRIPT,
        speaker="张三",
    )
    assert "verbal_habits" in result
    assert "high_frequency_phrases" in result["verbal_habits"]


def test_extract_reasoning_patterns_returns_dict():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="""
reasoning_patterns:
  evaluating_experiment:
    steps:
      - "先问有没有对照实验"
      - "再看具体数据是否支撑claim"
    bias: "对缺乏对照实验的工作天然不信任"
""")]
    mock_client.messages.create.return_value = mock_response

    result = extract_reasoning_patterns(
        client=mock_client,
        model="claude-sonnet-4-20250514",
        transcript=SAMPLE_TRANSCRIPT,
        speaker="张三",
    )
    assert "reasoning_patterns" in result


def test_merge_into_persona(tmp_path):
    """merge_into_persona should update style.yaml with new exemplars."""
    persona_dir = str(tmp_path / "persona")
    os.makedirs(persona_dir, exist_ok=True)

    # Existing style.yaml with one exemplar
    existing = {"voice": {"summary": "直接", "exemplars": [
        {"context": "old", "good": "old good", "note": "old note"}
    ]}}
    style_path = os.path.join(persona_dir, "style.yaml")
    with open(style_path, "w") as f:
        yaml.dump(existing, f, allow_unicode=True)

    new_exemplars = [
        {"context": "评价数据", "good": "100圈就衰减了15%，离实用差太远", "note": "锚定数据"}
    ]
    merge_into_persona(persona_dir, new_exemplars=new_exemplars)

    with open(style_path, "r") as f:
        updated = yaml.safe_load(f)

    assert len(updated["voice"]["exemplars"]) == 2
    assert any("100圈" in e["good"] for e in updated["voice"]["exemplars"])