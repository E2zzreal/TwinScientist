# tests/agent/test_prompts.py
import pytest
from agent.prompts import build_system_prompt

def test_build_system_prompt_contains_identity(tmp_path):
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "identity.yaml").write_text(
        "personality_sketch: |\n  I speak directly.\ncore_beliefs: []\n"
    )
    (persona_dir / "thinking_frameworks.yaml").write_text("frameworks: {}\n")
    (persona_dir / "boundaries.yaml").write_text(
        "confident_domains: []\nfamiliar_but_not_expert: []\noutside_expertise: []\n"
    )
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "topic_index.yaml").write_text("topics: {}\n")

    prompt = build_system_prompt(str(persona_dir), str(memory_dir))
    assert "I speak directly" in prompt
    assert "知识领域" in prompt
    assert "思维方式" in prompt

def test_build_system_prompt_includes_tool_instructions(tmp_path):
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "identity.yaml").write_text("personality_sketch: test\n")
    (persona_dir / "thinking_frameworks.yaml").write_text("frameworks: {}\n")
    (persona_dir / "boundaries.yaml").write_text(
        "confident_domains: []\nfamiliar_but_not_expert: []\noutside_expertise: []\n"
    )
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "topic_index.yaml").write_text("topics: {}\n")

    prompt = build_system_prompt(str(persona_dir), str(memory_dir))
    assert "recall" in prompt