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