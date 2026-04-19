# agent/session.py
"""Session persistence: save and load conversation summaries across sessions."""
import os
import yaml
from datetime import datetime


def save_session_summary(memory_dir: str, summary: str) -> str:
    """Persist current session summary to disk."""
    if not summary.strip():
        return ""
    conv_dir = os.path.join(memory_dir, "conversations")
    os.makedirs(conv_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"session-{timestamp}.yaml"
    filepath = os.path.join(conv_dir, filename)

    entry = {
        "type": "session_summary",
        "summary": summary,
        "timestamp": timestamp,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(entry, f, allow_unicode=True)

    return filepath


def load_latest_session_summary(memory_dir: str) -> str:
    """Load the most recent session summary, if any."""
    conv_dir = os.path.join(memory_dir, "conversations")
    if not os.path.exists(conv_dir):
        return ""

    session_files = sorted([
        f for f in os.listdir(conv_dir)
        if f.startswith("session-") and f.endswith(".yaml")
    ], reverse=True)  # most recent first

    if not session_files:
        return ""

    latest_path = os.path.join(conv_dir, session_files[0])
    with open(latest_path, "r", encoding="utf-8") as f:
        entry = yaml.safe_load(f) or {}

    return entry.get("summary", "")