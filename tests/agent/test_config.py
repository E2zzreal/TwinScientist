# tests/agent/test_config.py
import os
import pytest
from agent.config import load_config

def test_load_config_returns_dict(tmp_path):
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
    config = load_config(str(config_file))
    assert config["model"] == "claude-sonnet-4-20250514"
    assert config["context_budget"]["total"] == 100000
    assert config["conversation"]["recent_window"] == 10

def test_load_config_with_env_override(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("model: claude-sonnet-4-20250514\nmax_tokens: 4096\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    config = load_config(str(config_file))
    assert config["api_key"] == "test-key"

def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")