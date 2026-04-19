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


STYLE_CORRECTION_PROMPT = """用户对数字分身的一段回答给出了反馈，说它"不像本人"。

原始回答：
{original_response}

用户反馈：
{feedback}

对话上下文：
{context}

请生成一个"示范对"（exemplar pair），说明本人应该怎么回答。
只输出YAML，不要其他文字：

context: "场景描述（一句话）"
bad: "不像本人的回答（即上面的原始回答或其核心问题）"
good: "本人会怎么说（根据反馈推断，要具体、有风格）"
note: "这个示范对体现了什么风格原则"
"""


def apply_style_correction(client, model: str, persona_dir: str,
                            changelog_path: str, original_response: str,
                            feedback: str, context: str = ""):
    """Apply user feedback to generate a new style exemplar."""
    import re

    # Snapshot current state
    style_path = os.path.join(persona_dir, "style.yaml")
    if os.path.exists(style_path):
        with open(style_path, "r", encoding="utf-8") as f:
            style = yaml.safe_load(f) or {}
    else:
        style = {"voice": {"summary": "", "exemplars": []}}

    before_count = len(style.get("voice", {}).get("exemplars", []))

    # Generate corrected exemplar via LLM
    prompt = STYLE_CORRECTION_PROMPT.format(
        original_response=original_response,
        feedback=feedback,
        context=context or "（未提供上下文）",
    )
    response = client.messages.create(
        model=model,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    raw = re.sub(r"^```(?:yaml)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        exemplar = yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        exemplar = {"context": context, "bad": original_response,
                    "good": feedback, "note": "用户直接提供的修正"}

    # Append to style.yaml
    if "voice" not in style:
        style["voice"] = {"summary": "", "exemplars": []}
    if "exemplars" not in style["voice"]:
        style["voice"]["exemplars"] = []
    style["voice"]["exemplars"].append(exemplar)

    with open(style_path, "w", encoding="utf-8") as f:
        yaml.dump(style, f, allow_unicode=True, default_flow_style=False)

    # Record change
    record_change(changelog_path, "style_correction", {
        "feedback": feedback,
        "context": context,
        "before_exemplar_count": before_count,
        "after_exemplar_count": before_count + 1,
        "snapshot": {"exemplar_added": exemplar},
    })


def apply_stance_update(memory_dir: str, changelog_path: str,
                        topic: str, new_stance: str, reason: str = ""):
    """Update the stance on a topic in topic_index.yaml."""
    index_path = os.path.join(memory_dir, "topic_index.yaml")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = yaml.safe_load(f) or {"topics": {}}
    else:
        index = {"topics": {}}

    topics = index.setdefault("topics", {})
    old_stance = topics.get(topic, {}).get("stance", "（无记录）")

    # Update or create topic entry
    if topic not in topics:
        topics[topic] = {
            "summary": f"新话题，{new_stance}",
            "paper_count": 0,
            "stance": new_stance,
            "detail_files": [],
        }
    else:
        topics[topic]["stance"] = new_stance

    with open(index_path, "w", encoding="utf-8") as f:
        yaml.dump(index, f, allow_unicode=True, default_flow_style=False)

    # Record change with before/after snapshot
    record_change(changelog_path, "stance_update", {
        "topic": topic,
        "before": old_stance,
        "after": new_stance,
        "reason": reason,
        "snapshot": {"topic_entry": topics[topic]},
    })