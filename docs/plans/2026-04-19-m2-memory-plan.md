# M2: 能记忆 — 智能压缩 + 跨会话记忆实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现 30 轮+对话不丢失关键信息，新观点跨会话可持久化，对话记忆可被 recall 工具检索。

**Architecture:** 三个升级点——①用 LLM 替换朴素截断压缩；②RecallTool 扩展对话记忆搜索 + 热重载 topic_index；③TwinScientist 启动时加载历史会话摘要。三者互相独立，可并行开发。

**Tech Stack:** 复用 M1 栈（anthropic SDK, PyYAML, tiktoken）。无新依赖。

**M1 现有问题：**
- `context_manager.py:59-75`：`_compress()` 朴素截断到 100 字，丢失关键信息
- `tools.py:10`：`_topic_index` 初始化后不刷新，新保存的对话记忆无法被 recall
- `tools.py:20-58`：`RecallTool.execute()` 不搜索 `memory/conversations/` 目录
- `main.py`：启动时不加载前次会话摘要

---

### Task 1: LLM 驱动的对话压缩

**Files:**
- Modify: `agent/context_manager.py`
- Modify: `agent/main.py`（向 ContextManager 注入 LLM 客户端）
- Modify: `tests/agent/test_context_manager.py`

**核心改动：** `_compress()` 改为调用 Claude API 生成真正的摘要，而非截断拼接。

**Step 1: 在现有测试文件末尾追加新测试**

```python
# 追加到 tests/agent/test_context_manager.py

from unittest.mock import MagicMock, patch

def test_llm_compress_called_when_overflow():
    """When turns overflow window, LLM compressor should be called."""
    cm = ContextManager(_make_config())

    mock_llm = MagicMock()
    mock_llm.return_value = "摘要：讨论了氢能催化剂，结论是稳定性是瓶颈。"
    cm.set_llm_compressor(mock_llm)

    for i in range(5):
        cm.add_turn(f"user{i}", f"answer{i}")

    cm.prepare("new message")

    mock_llm.assert_called_once()
    assert "摘要" in cm._summary

def test_fallback_compress_when_no_llm():
    """Without LLM compressor, should fall back to naive compression."""
    cm = ContextManager(_make_config())
    for i in range(5):
        cm.add_turn(f"user{i}", f"answer{i}")
    cm.prepare("new message")
    # Naive fallback: should still produce a summary
    assert len(cm._summary) > 0
```

**Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/agent/test_context_manager.py::test_llm_compress_called_when_overflow tests/agent/test_context_manager.py::test_fallback_compress_when_no_llm -v`

Expected: FAIL — `AttributeError: 'ContextManager' object has no attribute 'set_llm_compressor'`

**Step 3: Modify ContextManager**

在 `agent/context_manager.py` 中，在 `__init__` 末尾增加：

```python
# 在 __init__ 末尾加
self._llm_compressor = None  # optional: fn(turns: list[tuple]) -> str
```

在类中增加以下两个方法，并替换 `_compress()`：

```python
def set_llm_compressor(self, compressor_fn):
    """Inject LLM-based compressor. fn(turns: list[tuple[str,str]]) -> str"""
    self._llm_compressor = compressor_fn

def _compress(self):
    """Compress older turns into a summary using LLM if available."""
    overflow_count = len(self._turns) - self.recent_window
    if overflow_count <= 0:
        return

    old_turns = self._turns[:overflow_count]

    if self._llm_compressor:
        # LLM-powered compression
        new_part = self._llm_compressor(old_turns)
    else:
        # Naive fallback: keep first 200 chars of each message
        lines = []
        for user_msg, assistant_msg in old_turns:
            lines.append(f"- 用户: {user_msg[:200]}")
            lines.append(f"  回答: {assistant_msg[:200]}")
        new_part = "\n".join(lines)

    if self._summary:
        self._summary = self._summary + "\n" + new_part
    else:
        self._summary = new_part

    self._turns = self._turns[overflow_count:]
```

**Step 4: 在 agent/main.py 中注入 LLM 压缩器**

在 `TwinScientist.__init__` 末尾加：

```python
# 注入 LLM 压缩器到 ContextManager
self.context.set_llm_compressor(self._llm_compress)
```

在 `TwinScientist` 类中增加方法：

```python
def _llm_compress(self, turns: list[tuple[str, str]]) -> str:
    """Use Claude to compress conversation turns into a concise summary."""
    turns_text = "\n".join(
        f"用户: {u}\n回答: {a}" for u, a in turns
    )
    prompt = f"""以下是一段对话记录，请压缩成简洁摘要（不超过300字）。
保留：话题、关键结论、重要观点、提到的具体数据或论文。
舍弃：重复内容、寒暄、过渡语句。

对话记录：
{turns_text}

摘要："""

    response = self.client.messages.create(
        model=self.config["model"],
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()
```

**Step 5: Run all context manager tests**

Run: `./venv/bin/python -m pytest tests/agent/test_context_manager.py -v`

Expected: 6 passed

**Step 6: Commit**

```bash
git add agent/context_manager.py agent/main.py tests/agent/test_context_manager.py
git commit -m "feat(agent): replace naive compression with LLM-powered summarization"
```

---

### Task 2: RecallTool 扩展 — 对话记忆检索 + 热重载

**Files:**
- Modify: `agent/tools.py`
- Modify: `tests/agent/test_tools.py`

**核心改动：**
1. `RecallTool` 新增 `recall_conversations(topic)` 方法，搜索 `memory/conversations/` 目录
2. `execute()` 新增 `depth="conversations"` 分支
3. `_load_topic_index()` 改为每次 execute 前热重载，避免 stale 数据

**Step 1: 在现有测试文件末尾追加新测试**

```python
# 追加到 tests/agent/test_tools.py

import yaml as _yaml
from datetime import datetime

def _add_conversation_memory(memory_dir, topic, content):
    """Helper: create a conversation memory file."""
    conv_dir = os.path.join(memory_dir, "conversations")
    os.makedirs(conv_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(conv_dir, f"{ts}-{topic}.yaml")
    with open(path, "w") as f:
        _yaml.dump({"topic": topic, "content": content, "timestamp": ts}, f,
                   allow_unicode=True)


def test_recall_conversations(tmp_path):
    memory_dir = _make_memory(tmp_path)
    _add_conversation_memory(
        str(memory_dir),
        "hydrogen_catalyst",
        "会议中讨论了稳定性问题，认为1000圈以上才算实用。"
    )
    tool = RecallTool(str(memory_dir))
    result = tool.execute("hydrogen_catalyst", depth="conversations")
    assert "稳定性" in result or "1000圈" in result


def test_recall_topic_index_hotreload(tmp_path):
    """topic_index should be reloaded on each execute call."""
    memory_dir = _make_memory(tmp_path)
    tool = RecallTool(str(memory_dir))

    # Add a new topic to the index after init
    index_path = os.path.join(str(memory_dir), "topic_index.yaml")
    with open(index_path, "r") as f:
        index = _yaml.safe_load(f)
    index["topics"]["new_topic"] = {
        "summary": "新话题摘要",
        "paper_count": 0,
        "stance": "刚开始了解",
        "detail_files": [],
    }
    with open(index_path, "w") as f:
        _yaml.dump(index, f, allow_unicode=True)

    # Should find the new topic without reinitializing
    result = tool.execute("new_topic", depth="summary")
    assert "新话题摘要" in result
```

**Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/agent/test_tools.py::test_recall_conversations tests/agent/test_tools.py::test_recall_topic_index_hotreload -v`

Expected: FAIL

**Step 3: Modify agent/tools.py**

将 `RecallTool` 替换为以下完整实现：

```python
# agent/tools.py
import os
import yaml


class RecallTool:
    """Progressive disclosure memory recall — reads structured YAML files."""

    def __init__(self, memory_dir: str):
        self.memory_dir = memory_dir

    def _load_topic_index(self) -> dict:
        """Hot-reload topic index on every call."""
        path = os.path.join(self.memory_dir, "topic_index.yaml")
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("topics", {})

    def execute(self, topic: str, depth: str = "summary",
                paper_id: str | None = None) -> str:
        topic_index = self._load_topic_index()

        if depth == "conversations":
            return self._recall_conversations(topic)

        if topic not in topic_index:
            return f"关于「{topic}」没有找到相关记忆。"

        if depth == "summary":
            return self._recall_summary(topic, topic_index)
        elif depth == "detail":
            return self._recall_detail(topic, topic_index)
        elif depth == "specific_paper":
            return self._recall_paper(topic, paper_id, topic_index)
        else:
            return self._recall_summary(topic, topic_index)

    def _recall_summary(self, topic: str, topic_index: dict) -> str:
        info = topic_index[topic]
        return yaml.dump(info, allow_unicode=True, default_flow_style=False)

    def _recall_detail(self, topic: str, topic_index: dict) -> str:
        detail_path = os.path.join(
            self.memory_dir, "topics", f"{topic}.yaml"
        )
        if not os.path.exists(detail_path):
            return self._recall_summary(topic, topic_index)
        with open(detail_path, "r", encoding="utf-8") as f:
            return f.read()

    def _recall_paper(self, topic: str, paper_id: str | None,
                      topic_index: dict) -> str:
        if not paper_id:
            return self._recall_detail(topic, topic_index)
        info = topic_index.get(topic, {})
        for detail_file in info.get("detail_files", []):
            if paper_id in detail_file:
                paper_path = os.path.join(self.memory_dir, detail_file)
                if os.path.exists(paper_path):
                    with open(paper_path, "r", encoding="utf-8") as f:
                        return f.read()
        return f"关于「{paper_id}」没有找到具体论文记忆。"

    def _recall_conversations(self, topic: str) -> str:
        """Search conversation memories for a given topic."""
        conv_dir = os.path.join(self.memory_dir, "conversations")
        if not os.path.exists(conv_dir):
            return f"没有找到关于「{topic}」的对话记忆。"

        results = []
        for filename in sorted(os.listdir(conv_dir)):
            if not filename.endswith(".yaml"):
                continue
            filepath = os.path.join(conv_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                entry = yaml.safe_load(f) or {}
            if entry.get("topic") == topic:
                results.append(
                    f"[{entry.get('timestamp', '')}] {entry.get('content', '')}"
                )

        if not results:
            return f"没有找到关于「{topic}」的对话记忆。"
        return "\n---\n".join(results)


class SaveToMemoryTool:
    """Persist new insights from conversation to memory files."""

    def __init__(self, memory_dir: str):
        self.memory_dir = memory_dir

    def execute(self, topic: str, content: str, source: str = "conversation") -> str:
        conv_dir = os.path.join(self.memory_dir, "conversations")
        os.makedirs(conv_dir, exist_ok=True)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}-{topic}.yaml"
        filepath = os.path.join(conv_dir, filename)

        entry = {
            "topic": topic,
            "content": content,
            "source": source,
            "timestamp": timestamp,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(entry, f, allow_unicode=True)

        return f"已保存到 {filename}"
```

**Step 4: Run all tools tests**

Run: `./venv/bin/python -m pytest tests/agent/test_tools.py -v`

Expected: 6 passed

**Step 5: Commit**

```bash
git add agent/tools.py tests/agent/test_tools.py
git commit -m "feat(agent): extend recall with conversation memory search and hot-reload"
```

---

### Task 3: 跨会话记忆 — 启动时加载历史摘要

**Files:**
- Modify: `agent/main.py`
- Create: `agent/session.py`
- Modify: `tests/agent/test_main.py`

**核心改动：** 每次会话结束时将对话摘要持久化；下次启动时自动加载最近一次的摘要注入 ContextManager。

**Step 1: 追加测试到 tests/agent/test_main.py**

```python
# 追加到 tests/agent/test_main.py
import yaml as _yaml

def test_session_summary_saved_on_end(tmp_path):
    project_dir = _make_project(tmp_path)
    agent = TwinScientist(project_dir)
    agent.context.add_turn("氢能催化剂有什么进展？", "单原子Pt是个好方向，但稳定性差。")

    agent.end_session()

    session_dir = os.path.join(project_dir, "memory", "conversations")
    files = [f for f in os.listdir(session_dir) if f.startswith("session-")]
    assert len(files) == 1


def test_previous_session_loaded_on_init(tmp_path):
    project_dir = _make_project(tmp_path)

    # Write a fake previous session summary
    session_dir = os.path.join(project_dir, "memory", "conversations")
    os.makedirs(session_dir, exist_ok=True)
    summary_path = os.path.join(session_dir, "session-20250101-120000.yaml")
    with open(summary_path, "w") as f:
        _yaml.dump({
            "type": "session_summary",
            "summary": "上次讨论了单原子催化稳定性问题。",
            "timestamp": "20250101-120000",
        }, f, allow_unicode=True)

    agent = TwinScientist(project_dir)
    # Previous summary should be loaded into context
    assert "上次讨论" in agent.context._summary
```

**Step 2: Run tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/agent/test_main.py::test_session_summary_saved_on_end tests/agent/test_main.py::test_previous_session_loaded_on_init -v`

Expected: FAIL

**Step 3: Create agent/session.py**

```python
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
```

**Step 4: Modify agent/main.py**

在 `TwinScientist.__init__` 末尾（注入压缩器之后）加：

```python
# Load previous session summary
from agent.session import load_latest_session_summary
prev_summary = load_latest_session_summary(self.memory_dir)
if prev_summary:
    self.context._summary = f"[上次会话摘要]\n{prev_summary}"
```

在 `TwinScientist` 类末尾增加 `end_session` 方法：

```python
def end_session(self):
    """Persist current session summary to disk for next session."""
    from agent.session import save_session_summary
    # Build a summary of this entire session
    all_turns = list(self.context._turns)
    if not all_turns and not self.context._summary:
        return

    summary_parts = []
    if self.context._summary:
        summary_parts.append(self.context._summary)

    # Use LLM to compress remaining turns if any
    if all_turns:
        compressed = self._llm_compress(all_turns)
        summary_parts.append(compressed)

    final_summary = "\n".join(summary_parts)
    save_session_summary(self.memory_dir, final_summary)
```

**Step 5: Run all main tests**

Run: `./venv/bin/python -m pytest tests/agent/test_main.py -v`

Expected: 5 passed

**Step 6: Wire up end_session in CLI**

在 `interface/cli.py` 的 `run()` 函数中，在退出逻辑前加：

```python
# 在 "再见" 输出之前
try:
    agent.end_session()
except Exception:
    pass  # session save failure should not block exit
```

具体位置：找到两处退出点（`/quit` 和 `KeyboardInterrupt`），各加一行 `agent.end_session()`。

**Step 7: Commit**

```bash
git add agent/session.py agent/main.py interface/cli.py tests/agent/test_main.py
git commit -m "feat(agent): add cross-session memory persistence and startup loading"
```

---

### Task 4: 全量回归测试 + 30轮压力测试

**Files:**
- Create: `tests/test_m2_regression.py`

**Step 1: Write regression tests**

```python
# tests/test_m2_regression.py
"""M2 regression: verifies 30+ turns without losing key information."""
import os
import yaml
import pytest
from unittest.mock import MagicMock, patch

from agent.context_manager import ContextManager
from agent.tools import RecallTool, SaveToMemoryTool


def _make_config():
    return {
        "context_budget": {
            "total": 100000, "fixed_zone": 8000,
            "dynamic_zone": 12000, "conversation_zone": 80000,
        },
        "conversation": {
            "recent_window": 5,
            "compression_trigger": 5,
            "emergency_threshold": 70000,
        },
        "model": "claude-sonnet-4-20250514",
    }


def test_30_turns_no_crash():
    """30 turns with compression should not raise errors."""
    cm = ContextManager(_make_config())

    mock_compressor = MagicMock(return_value="摘要：讨论了多个话题。")
    cm.set_llm_compressor(mock_compressor)

    for i in range(30):
        cm.prepare(f"question {i}")
        cm.add_turn(f"用户问题 {i}", f"详细回答 {i}，包含数据和分析。")

    history = cm.get_history()
    # Should have summary + recent 5 turns = 1 summary pair + 10 messages
    assert len(history) <= 14
    assert cm._summary != ""


def test_compression_preserves_summary():
    """Multiple compressions should accumulate summaries correctly."""
    cm = ContextManager(_make_config())

    call_count = [0]
    def mock_compressor(turns):
        call_count[0] += 1
        return f"第{call_count[0]}批摘要"

    cm.set_llm_compressor(mock_compressor)

    for i in range(15):
        cm.prepare(f"q{i}")
        cm.add_turn(f"user {i}", f"answer {i}")

    assert call_count[0] >= 2
    assert "摘要" in cm._summary


def test_save_and_recall_conversation(tmp_path):
    """Save a memory then recall it via conversations depth."""
    memory_dir = str(tmp_path / "memory")
    os.makedirs(os.path.join(memory_dir, "conversations"), exist_ok=True)

    index_path = os.path.join(memory_dir, "topic_index.yaml")
    with open(index_path, "w") as f:
        yaml.dump({"topics": {}}, f)

    save_tool = SaveToMemoryTool(memory_dir)
    save_tool.execute(
        topic="hydrogen_catalyst",
        content="会议中确认了1000圈稳定性是工程化的门槛。",
        source="meeting_2025"
    )

    recall_tool = RecallTool(memory_dir)
    result = recall_tool.execute("hydrogen_catalyst", depth="conversations")
    assert "1000圈" in result


def test_topic_index_hotreload(tmp_path):
    """RecallTool should see newly added topics without reinit."""
    memory_dir = str(tmp_path / "memory")
    os.makedirs(memory_dir, exist_ok=True)

    index_path = os.path.join(memory_dir, "topic_index.yaml")
    with open(index_path, "w") as f:
        yaml.dump({"topics": {"topic_a": {"summary": "A", "paper_count": 0,
                                           "stance": "ok", "detail_files": []}}},
                  f, allow_unicode=True)

    tool = RecallTool(memory_dir)
    assert "A" in tool.execute("topic_a")

    # Add new topic to index
    with open(index_path, "r") as f:
        index = yaml.safe_load(f)
    index["topics"]["topic_b"] = {"summary": "B", "paper_count": 0,
                                   "stance": "new", "detail_files": []}
    with open(index_path, "w") as f:
        yaml.dump(index, f, allow_unicode=True)

    # Should find topic_b immediately (hot-reload)
    assert "B" in tool.execute("topic_b")
```

**Step 2: Run regression tests**

Run: `./venv/bin/python -m pytest tests/test_m2_regression.py -v`

Expected: 4 passed

**Step 3: Run full test suite**

Run: `./venv/bin/python -m pytest tests/ -v`

Expected: All passed (30+ tests)

**Step 4: Final commit and push**

```bash
git add tests/test_m2_regression.py
git commit -m "test: add M2 regression tests for compression and memory recall"
git push origin main
```
