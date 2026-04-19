# ingestion/persona_extractor.py
"""Extract persona characteristics from transcripts and papers."""
import os
import re
import yaml

EXEMPLARS_PROMPT = """以下是一段会议或讨论的转写文本，请找出发言人「{speaker}」
最能体现其个人风格的 3-5 段发言。

转写文本：
{transcript}

请按以下YAML格式输出（只输出YAML）：

exemplars:
  - context: "场景描述（评价论文/回应提问/解释概念/表达异议等）"
    good: "原文发言摘录（保持原话）"
    note: "这段话体现了什么风格特征"
"""

VERBAL_HABITS_PROMPT = """分析以下转写文本中发言人「{speaker}」的语言习惯。

转写文本：
{transcript}

请按以下YAML格式输出（只输出YAML）：

verbal_habits:
  high_frequency_phrases:
    - "口头禅或高频表达1"
    - "口头禅或高频表达2"
  sentence_style: "句子结构特点描述"
  language_mix: "中英文混用模式描述"
"""

REASONING_PROMPT = """分析以下转写文本中发言人「{speaker}」的思维模式。

转写文本：
{transcript}

请按以下YAML格式输出（只输出YAML）：

reasoning_patterns:
  evaluating_experiment:
    steps:
      - "步骤1"
    bias: "偏好或偏见描述"
"""


def _call_llm(client, model: str, prompt: str) -> dict:
    """Call LLM and parse YAML response."""
    raw = client.simple_chat(prompt, max_tokens=1000)
    raw = re.sub(r"^```(?:yaml)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    try:
        return yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        return {}


def extract_style_exemplars(client, model: str, transcript: str,
                             speaker: str) -> list:
    prompt = EXEMPLARS_PROMPT.format(speaker=speaker,
                                      transcript=transcript[:4000])
    result = _call_llm(client, model, prompt)
    return result.get("exemplars", [])


def extract_verbal_habits(client, model: str, transcript: str,
                           speaker: str) -> dict:
    prompt = VERBAL_HABITS_PROMPT.format(speaker=speaker,
                                          transcript=transcript[:4000])
    return _call_llm(client, model, prompt)


def extract_reasoning_patterns(client, model: str, transcript: str,
                                speaker: str) -> dict:
    prompt = REASONING_PROMPT.format(speaker=speaker,
                                      transcript=transcript[:4000])
    return _call_llm(client, model, prompt)


def merge_into_persona(persona_dir: str,
                       new_exemplars: list | None = None,
                       verbal_habits: dict | None = None,
                       reasoning_patterns: dict | None = None):
    """Merge extracted features into persona YAML files."""
    style_path = os.path.join(persona_dir, "style.yaml")

    if os.path.exists(style_path):
        with open(style_path, "r", encoding="utf-8") as f:
            style = yaml.safe_load(f) or {}
    else:
        style = {"voice": {"summary": "", "exemplars": []}}

    if "voice" not in style:
        style["voice"] = {"summary": "", "exemplars": []}
    if "exemplars" not in style["voice"]:
        style["voice"]["exemplars"] = []

    if new_exemplars:
        style["voice"]["exemplars"].extend(new_exemplars)

    if verbal_habits and "verbal_habits" in verbal_habits:
        style["voice"]["verbal_habits"] = verbal_habits["verbal_habits"]

    with open(style_path, "w", encoding="utf-8") as f:
        yaml.dump(style, f, allow_unicode=True, default_flow_style=False)

    if reasoning_patterns and "reasoning_patterns" in reasoning_patterns:
        frameworks_path = os.path.join(persona_dir, "thinking_frameworks.yaml")
        if os.path.exists(frameworks_path):
            with open(frameworks_path, "r", encoding="utf-8") as f:
                frameworks = yaml.safe_load(f) or {}
        else:
            frameworks = {"frameworks": {}}

        if "frameworks" not in frameworks:
            frameworks["frameworks"] = {}
        frameworks["frameworks"].update(
            reasoning_patterns["reasoning_patterns"]
        )
        with open(frameworks_path, "w", encoding="utf-8") as f:
            yaml.dump(frameworks, f, allow_unicode=True, default_flow_style=False)