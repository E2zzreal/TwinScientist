# M4: 能进化 — 反馈回路与变更日志实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现三条进化路径——风格反馈校正、立场更新、变更日志审计回滚，让数字分身能响应"不像我"或"我现在不这么看了"的指令持续改进。

**Architecture:** 核心模块 `agent/evolution.py` 提供 changelog 管理、风格校正、立场更新、回滚四个能力。Agent 新增 `give_feedback` 工具接入进化引擎。所有变更写入 `evolution/changelog.yaml`，每条记录含 before/after 快照，支持按步骤回滚。

**Tech Stack:** 复用 M1-M3 栈，无新依赖。

**M4 任务依赖：**
- Task 1（changelog 管理器）、Task 2（风格校正）、Task 3（立场更新）独立并行
- Task 4（接入 Agent 工具）依赖 Task 1+2+3
- Task 5（全量回归）依赖 Task 4

---

### Task 1: Changelog 管理器

**Files:**
- Create: `agent/evolution.py`（changelog 部分）
- Create: `tests/agent/test_evolution.py`（Task 1 的测试）

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/agent/test_evolution.py -v`
Expected: FAIL

**Step 3: Write implementation (changelog part only)**

```python
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
```

**Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest tests/agent/test_evolution.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add agent/evolution.py tests/agent/test_evolution.py
git commit -m "feat(agent): add evolution changelog manager with rollback support"
```

---

### Task 2: 风格校正 — "这个回答不像我"

**Files:**
- Modify: `agent/evolution.py`（新增 apply_style_correction）
- Modify: `tests/agent/test_evolution.py`（追加测试）

**Step 1: 追加测试到 tests/agent/test_evolution.py**

```python
# 追加到 tests/agent/test_evolution.py
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
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="""
context: "被问到催化剂稳定性"
bad: "这个方向有一定研究价值。"
good: "你看Table 2，只测了100圈，这离实用差太远。"
note: "应当锚定具体数据，不说空话"
""")]
    mock_client.messages.create.return_value = mock_response

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
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="""
context: "测试"
bad: "不好的回答"
good: "好的回答"
note: "说明"
""")]
    mock_client.messages.create.return_value = mock_response

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
```

**Step 2: Run tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/agent/test_evolution.py::test_apply_style_correction_adds_exemplar tests/agent/test_evolution.py::test_apply_style_correction_records_before_snapshot -v`
Expected: FAIL

**Step 3: Append apply_style_correction to agent/evolution.py**

```python
# 追加到 agent/evolution.py

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
```

**Step 4: Run all evolution tests**

Run: `./venv/bin/python -m pytest tests/agent/test_evolution.py -v`
Expected: 6 passed

**Step 5: Commit**

```bash
git add agent/evolution.py tests/agent/test_evolution.py
git commit -m "feat(agent): add style correction from user feedback with exemplar generation"
```

---

### Task 3: 立场更新 — "我现在不这么看了"

**Files:**
- Modify: `agent/evolution.py`（新增 apply_stance_update）
- Modify: `tests/agent/test_evolution.py`（追加测试）

**Step 1: 追加测试**

```python
# 追加到 tests/agent/test_evolution.py
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
```

**Step 2: Run tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/agent/test_evolution.py::test_apply_stance_update_modifies_topic_index tests/agent/test_evolution.py::test_apply_stance_update_unknown_topic -v`
Expected: FAIL

**Step 3: Append apply_stance_update to agent/evolution.py**

```python
# 追加到 agent/evolution.py

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
```

**Step 4: Run all evolution tests**

Run: `./venv/bin/python -m pytest tests/agent/test_evolution.py -v`
Expected: 8 passed

**Step 5: Commit**

```bash
git add agent/evolution.py tests/agent/test_evolution.py
git commit -m "feat(agent): add stance update with changelog snapshot"
```

---

### Task 4: 接入 Agent 工具 — give_feedback

**Files:**
- Modify: `agent/main.py`（新增 give_feedback 工具定义和执行）
- Modify: `tests/agent/test_main.py`（追加测试）

**Step 1: 追加测试到 tests/agent/test_main.py**

```python
# 追加到 tests/agent/test_main.py
import yaml as _yaml

def test_give_feedback_style_correction(tmp_path):
    """give_feedback with 'not like me' should trigger style correction."""
    project_dir = _make_project(tmp_path)
    agent = TwinScientist(project_dir)

    mock_client_response = MagicMock()
    mock_client_response.content = [MagicMock(text="""
context: "评价实验"
bad: "这个研究有价值"
good: "你看数据，100圈就衰减了"
note: "锚定数据"
""")]
    agent.client.messages.create = MagicMock(return_value=mock_client_response)

    result = agent._execute_tool("give_feedback", {
        "feedback_type": "style",
        "feedback": "不像我，我会直接说数据",
        "original_response": "这个研究有价值",
        "context": "讨论稳定性",
    })

    assert "已记录" in result or "风格" in result

    # Check changelog was written
    changelog_path = os.path.join(project_dir, "evolution", "changelog.yaml")
    with open(changelog_path) as f:
        data = _yaml.safe_load(f)
    assert len(data["changes"]) >= 1


def test_give_feedback_stance_update(tmp_path):
    """give_feedback with stance update should modify topic_index."""
    project_dir = _make_project(tmp_path)
    agent = TwinScientist(project_dir)

    result = agent._execute_tool("give_feedback", {
        "feedback_type": "stance",
        "topic": "hydrogen_catalyst",
        "new_stance": "对产业化更谨慎了",
        "reason": "成本太高",
    })

    assert "已更新" in result or "立场" in result
```

**Step 2: Run tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/agent/test_main.py::test_give_feedback_style_correction tests/agent/test_main.py::test_give_feedback_stance_update -v`
Expected: FAIL

**Step 3: Modify agent/main.py**

在 `TOOL_DEFINITIONS` 列表末尾追加：

```python
    {
        "name": "give_feedback",
        "description": "记录用户对回答的反馈，触发风格校正或立场更新。当用户说'不像我'时用feedback_type=style；当用户说'我现在不这么看了'时用feedback_type=stance。",
        "input_schema": {
            "type": "object",
            "properties": {
                "feedback_type": {
                    "type": "string",
                    "enum": ["style", "stance"],
                    "description": "style=风格校正，stance=立场更新",
                },
                "feedback": {
                    "type": "string",
                    "description": "用户的反馈内容（style类型必填）",
                },
                "original_response": {
                    "type": "string",
                    "description": "被反馈的原始回答（style类型必填）",
                },
                "context": {
                    "type": "string",
                    "description": "对话上下文简述",
                },
                "topic": {
                    "type": "string",
                    "description": "相关话题ID（stance类型必填）",
                },
                "new_stance": {
                    "type": "string",
                    "description": "新的立场（stance类型必填）",
                },
                "reason": {
                    "type": "string",
                    "description": "立场变化的原因",
                },
            },
            "required": ["feedback_type"],
        },
    },
```

在 `_execute_tool` 方法中，在最后的 `return f"Unknown tool: {tool_name}"` 之前追加：

```python
        elif tool_name == "give_feedback":
            return self._handle_feedback(tool_input)
```

在 `TwinScientist` 类末尾追加 `_handle_feedback` 方法：

```python
    def _handle_feedback(self, tool_input: dict) -> str:
        """Route feedback to the appropriate evolution handler."""
        from agent.evolution import apply_style_correction, apply_stance_update
        import os

        changelog_path = os.path.join(
            self.project_dir,
            self.config.get("paths", {}).get("evolution_dir", "evolution"),
            "changelog.yaml",
        )
        # Ensure changelog file exists
        if not os.path.exists(changelog_path):
            import yaml
            os.makedirs(os.path.dirname(changelog_path), exist_ok=True)
            with open(changelog_path, "w") as f:
                yaml.dump({"changes": []}, f)

        feedback_type = tool_input.get("feedback_type", "style")

        if feedback_type == "style":
            apply_style_correction(
                client=self.client,
                model=self.config["model"],
                persona_dir=self.persona_dir,
                changelog_path=changelog_path,
                original_response=tool_input.get("original_response", ""),
                feedback=tool_input.get("feedback", ""),
                context=tool_input.get("context", ""),
            )
            return "已记录风格反馈，下次对话将体现改进。"

        elif feedback_type == "stance":
            apply_stance_update(
                memory_dir=self.memory_dir,
                changelog_path=changelog_path,
                topic=tool_input.get("topic", ""),
                new_stance=tool_input.get("new_stance", ""),
                reason=tool_input.get("reason", ""),
            )
            # Reload recall tool's index
            self.recall_tool = self.recall_tool.__class__(self.memory_dir)
            return f"已更新「{tool_input.get('topic', '')}」的立场。"

        return "未知的反馈类型。"
```

**Step 4: Run all main tests**

Run: `./venv/bin/python -m pytest tests/agent/test_main.py -v`
Expected: 7 passed

**Step 5: Commit**

```bash
git add agent/main.py tests/agent/test_main.py
git commit -m "feat(agent): add give_feedback tool for style correction and stance update"
```

---

### Task 5: 全量回归测试 + push

**Step 1: Run all tests**

Run: `./venv/bin/python -m pytest tests/ -v`
Expected: All passed (60+ tests)

**Step 2: Push**

```bash
git push origin main
```
