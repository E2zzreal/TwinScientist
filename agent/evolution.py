# agent/evolution.py
"""Evolution engine: feedback correction, stance updates, changelog management."""
import os
import yaml
from datetime import datetime


# ─── Changelog management ────────────────────────────────────────────────────

def _load_raw(changelog_path: str) -> dict:
    if not os.path.exists(changelog_path):
        return {"changes": []}
    with open(changelog_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"changes": []}


def _save_raw(changelog_path: str, data: dict):
    with open(changelog_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def record_change(changelog_path: str, change_type: str, details: dict):
    """Append a change record to the changelog."""
    data = _load_raw(changelog_path)
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": change_type,
        **details,
    }
    data["changes"].append(entry)
    _save_raw(changelog_path, data)


def load_changelog(changelog_path: str) -> list:
    """Return all change records as a list."""
    return _load_raw(changelog_path).get("changes", [])


def rollback_last_n(changelog_path: str, n: int) -> int:
    """Remove the last N changelog entries. Returns number actually removed."""
    data = _load_raw(changelog_path)
    changes = data.get("changes", [])
    actual = min(n, len(changes))
    data["changes"] = changes[:-actual] if actual else changes
    _save_raw(changelog_path, data)
    return actual