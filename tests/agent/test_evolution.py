# tests/agent/test_evolution.py
import os
import yaml
import pytest
from agent.evolution import record_change, load_changelog, rollback_last_n


def _make_changelog(tmp_path) -> str:
    path = str(tmp_path / "changelog.yaml")
    with open(path, "w") as f:
        yaml.dump({"changes": []}, f)
    return path


def test_record_change_appends_entry(tmp_path):
    path = _make_changelog(tmp_path)
    record_change(path, change_type="stance_update", details={
        "topic": "hydrogen_catalyst",
        "before": "看好单原子催化",
        "after": "对产业化更谨慎",
        "reason": "会议讨论了成本问题",
    })
    with open(path) as f:
        data = yaml.safe_load(f)
    assert len(data["changes"]) == 1
    assert data["changes"][0]["type"] == "stance_update"
    assert "timestamp" in data["changes"][0]


def test_record_change_multiple_entries(tmp_path):
    path = _make_changelog(tmp_path)
    record_change(path, "style_drift", {"detail": "反问句增多"})
    record_change(path, "knowledge_expansion", {"topic": "perovskite"})
    with open(path) as f:
        data = yaml.safe_load(f)
    assert len(data["changes"]) == 2


def test_load_changelog_returns_list(tmp_path):
    path = _make_changelog(tmp_path)
    record_change(path, "stance_update", {"topic": "test"})
    changes = load_changelog(path)
    assert isinstance(changes, list)
    assert len(changes) == 1


def test_rollback_last_n_removes_entries(tmp_path):
    path = _make_changelog(tmp_path)
    record_change(path, "stance_update", {"topic": "a", "snapshot": {"key": "v1"}})
    record_change(path, "stance_update", {"topic": "b", "snapshot": {"key": "v2"}})
    record_change(path, "style_drift", {"detail": "test", "snapshot": {}})

    rolled_back = rollback_last_n(path, n=2)
    assert rolled_back == 2

    with open(path) as f:
        data = yaml.safe_load(f)
    assert len(data["changes"]) == 1


from unittest.mock import MagicMock
from agent.evolution import apply_style_correction


def test_apply_style_correction_adds_exemplar(tmp_path):
    """apply_style_correction should add a corrected exemplar to style.yaml."""
    persona_dir = str(tmp_path / "persona")
    os.makedirs(persona_dir, exist_ok=True)
    style_path = os.path.join(persona_dir, "style.yaml")
    with open(style_path, "w") as f:
        yaml.dump({"voice": {"summary": "直接", "exemplars": []}}, f,
                  allow_unicode=True)

    changelog_path = _make_changelog(tmp_path)

    mock_client = MagicMock()
    mock_client.simple_chat.return_value = """context: "被问到催化剂稳定性"
bad: "这个方向有一定研究价值。"
good: "你看Table 2，只测了100圈，这离实用差太远。"
note: "应当锚定具体数据，不说空话"
"""

    apply_style_correction(
        client=mock_client,
        model="claude-sonnet-4-20250514",
        persona_dir=persona_dir,
        changelog_path=changelog_path,
        original_response="这个方向有一定研究价值。",
        feedback="不像我，我会直接说数据，而不是说'有研究价值'",
        context="用户问催化剂稳定性进展",
    )

    with open(style_path) as f:
        style = yaml.safe_load(f)
    assert len(style["voice"]["exemplars"]) == 1
    assert "100圈" in style["voice"]["exemplars"][0]["good"]

    changes = load_changelog(changelog_path)
    assert len(changes) == 1
    assert changes[0]["type"] == "style_correction"


def test_apply_style_correction_records_before_snapshot(tmp_path):
    persona_dir = str(tmp_path / "persona")
    os.makedirs(persona_dir, exist_ok=True)
    style_path = os.path.join(persona_dir, "style.yaml")
    existing = {"voice": {"summary": "直接", "exemplars": [
        {"context": "old", "good": "old text", "note": "old"}
    ]}}
    with open(style_path, "w") as f:
        yaml.dump(existing, f, allow_unicode=True)

    changelog_path = _make_changelog(tmp_path)

    mock_client = MagicMock()
    mock_client.simple_chat.return_value = """context: "测试"
bad: "不好的回答"
good: "好的回答"
note: "说明"
"""

    apply_style_correction(
        client=mock_client,
        model="claude-sonnet-4-20250514",
        persona_dir=persona_dir,
        changelog_path=changelog_path,
        original_response="不好的回答",
        feedback="语气太正式了",
        context="讨论论文",
    )

    changes = load_changelog(changelog_path)
    # Snapshot should contain previous exemplar count
    assert "before_exemplar_count" in changes[0]


from agent.evolution import apply_stance_update


def test_apply_stance_update_modifies_topic_index(tmp_path):
    memory_dir = str(tmp_path / "memory")
    os.makedirs(memory_dir, exist_ok=True)
    index_path = os.path.join(memory_dir, "topic_index.yaml")
    index = {"topics": {
        "hydrogen_catalyst": {
            "summary": "关注Pt替代",
            "paper_count": 5,
            "stance": "看好单原子催化方向",
            "detail_files": [],
        }
    }}
    with open(index_path, "w") as f:
        yaml.dump(index, f, allow_unicode=True)

    changelog_path = _make_changelog(tmp_path)

    apply_stance_update(
        memory_dir=memory_dir,
        changelog_path=changelog_path,
        topic="hydrogen_catalyst",
        new_stance="对单原子催化产业化前景更谨慎了",
        reason="会议讨论了成本和规模化问题",
    )

    with open(index_path) as f:
        updated = yaml.safe_load(f)
    assert "谨慎" in updated["topics"]["hydrogen_catalyst"]["stance"]

    changes = load_changelog(changelog_path)
    assert changes[0]["type"] == "stance_update"
    assert changes[0]["before"] == "看好单原子催化方向"
    assert "谨慎" in changes[0]["after"]


def test_apply_stance_update_unknown_topic(tmp_path):
    memory_dir = str(tmp_path / "memory")
    os.makedirs(memory_dir, exist_ok=True)
    index_path = os.path.join(memory_dir, "topic_index.yaml")
    with open(index_path, "w") as f:
        yaml.dump({"topics": {}}, f)

    changelog_path = _make_changelog(tmp_path)

    # Should not raise, just do nothing / create topic
    apply_stance_update(
        memory_dir=memory_dir,
        changelog_path=changelog_path,
        topic="nonexistent_topic",
        new_stance="开始了解这个方向",
        reason="刚看了几篇paper",
    )

    with open(index_path) as f:
        data = yaml.safe_load(f)
    # Topic should be created
    assert "nonexistent_topic" in data["topics"]