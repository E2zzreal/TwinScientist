# tests/ingestion/test_interactive_init.py
import os
import yaml
import pytest
from unittest.mock import MagicMock, patch
from ingestion.interactive_init import (
    build_calibration_questions,
    run_gap_filling,
    apply_corrections,
)


def test_build_calibration_questions_from_persona(tmp_path):
    """Should generate questions based on empty or sparse persona fields."""
    persona_dir = str(tmp_path / "persona")
    os.makedirs(persona_dir, exist_ok=True)
    with open(os.path.join(persona_dir, "identity.yaml"), "w") as f:
        yaml.dump({"personality_sketch": "", "core_beliefs": [],
                   "research_focus": [], "taste_profile": []}, f)
    with open(os.path.join(persona_dir, "boundaries.yaml"), "w") as f:
        yaml.dump({"confident_domains": [], "familiar_but_not_expert": [],
                   "outside_expertise": []}, f)

    questions = build_calibration_questions(persona_dir)
    assert isinstance(questions, list)
    assert len(questions) > 0
    assert all("question" in q and "field" in q for q in questions)


def test_apply_corrections_updates_identity(tmp_path):
    """apply_corrections should update identity.yaml with user answers."""
    persona_dir = str(tmp_path / "persona")
    os.makedirs(persona_dir, exist_ok=True)
    identity_path = os.path.join(persona_dir, "identity.yaml")
    with open(identity_path, "w") as f:
        yaml.dump({"personality_sketch": "", "core_beliefs": [],
                   "research_focus": [], "taste_profile": []}, f)

    corrections = [
        {"field": "research_focus", "value": ["氢能催化剂", "电解水"], "target_file": "identity.yaml"},
        {"field": "personality_sketch",
         "value": "说话直接，喜欢用数据说话。", "target_file": "identity.yaml"},
    ]
    apply_corrections(persona_dir, corrections)

    with open(identity_path, "r") as f:
        updated = yaml.safe_load(f)
    assert "氢能催化剂" in updated["research_focus"]
    assert "说话直接" in updated["personality_sketch"]


def test_run_gap_filling_returns_corrections(tmp_path):
    """run_gap_filling should collect user answers and return corrections."""
    persona_dir = str(tmp_path / "persona")
    os.makedirs(persona_dir, exist_ok=True)
    with open(os.path.join(persona_dir, "identity.yaml"), "w") as f:
        yaml.dump({"personality_sketch": "", "core_beliefs": [],
                   "research_focus": [], "taste_profile": []}, f)
    with open(os.path.join(persona_dir, "boundaries.yaml"), "w") as f:
        yaml.dump({"confident_domains": [], "familiar_but_not_expert": [],
                   "outside_expertise": []}, f)

    # Simulate user inputs
    with patch("builtins.input", side_effect=["氢能催化剂，电解水", "实验为主", ""]):
        corrections = run_gap_filling(persona_dir, max_questions=2)

    assert isinstance(corrections, list)