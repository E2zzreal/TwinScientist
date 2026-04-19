# M7: 能参会 — 实时会议参与实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让数字分身能实时参与学术会议——持续监听音频流、维护会议上下文、在合适时机发言，并能理解屏幕上共享的PPT内容。

**Architecture:** 四个组件——`MeetingContext`（滑动窗口+滚动摘要，维持~16K tokens 会议状态）、`TriggerDetector`（判断何时发言：被点名/涉及领域/主动发言）、`multimodal/meeting_vision.py`（定时截屏+PPT理解）、`interface/meeting.py`（串联全部组件的会议主循环）。真实会议SDK接入作为可选层，核心逻辑与平台无关。

**Tech Stack:** 复用 M5/M6 已有的 STT/TTS/vision 模块；`mss`（屏幕截图，轻量）；核心逻辑无额外依赖。

**会议上下文预算（始终 ≤16K tokens）：**
```
固定区: 身份+索引      ~4K
会议快照区:             ~4K  (主题/参会者/已讨论话题摘要)
近期原文区:             ~4K  (最近5分钟转写)
推理空间:              ~4K
```

---

### Task 1: MeetingContext — 滑动窗口 + 滚动摘要

**Files:**
- Create: `interface/meeting_context.py`
- Create: `tests/interface/test_meeting_context.py`

**Step 1: Write the failing test**

```python
# tests/interface/test_meeting_context.py
import pytest
from unittest.mock import MagicMock
from interface.meeting_context import MeetingContext


def _make_context(window_minutes=1, max_tokens=500):
    return MeetingContext(
        window_minutes=window_minutes,
        max_tokens=max_tokens,
    )


def test_meeting_context_init():
    ctx = _make_context()
    assert ctx.topic == ""
    assert ctx.participants == []
    assert len(ctx.recent_utterances) == 0
    assert ctx.rolling_summary == ""


def test_add_utterance_stores_entry():
    ctx = _make_context()
    ctx.add_utterance(speaker="张三", text="氢能催化剂的稳定性还有很大提升空间。")
    assert len(ctx.recent_utterances) == 1
    assert ctx.recent_utterances[0]["speaker"] == "张三"
    assert "稳定性" in ctx.recent_utterances[0]["text"]


def test_get_snapshot_returns_dict():
    ctx = _make_context()
    ctx.topic = "氢能催化剂进展"
    ctx.participants = ["张三", "李四"]
    ctx.add_utterance("张三", "单原子催化是个好方向。")
    snapshot = ctx.get_snapshot()
    assert "topic" in snapshot
    assert "participants" in snapshot
    assert "recent_text" in snapshot
    assert "单原子" in snapshot["recent_text"]


def test_compress_old_utterances_on_overflow(monkeypatch):
    """When recent_utterances token count exceeds budget, should compress."""
    ctx = _make_context(max_tokens=100)

    compress_called = [False]
    original_compress = ctx._compress_old_utterances
    def mock_compress():
        compress_called[0] = True
        original_compress()
    ctx._compress_old_utterances = mock_compress

    # Add many utterances to trigger compression
    for i in range(20):
        ctx.add_utterance("张三", f"这是第{i}句话，内容比较长，包含很多信息。")

    # Either compress was called or tokens stayed within budget
    tokens = ctx._count_recent_tokens()
    assert tokens <= ctx.max_tokens * 2 or compress_called[0]


def test_format_for_agent_injection():
    """Format output should be ready for injection into system prompt."""
    ctx = _make_context()
    ctx.topic = "测试会议"
    ctx.participants = ["张三"]
    ctx.add_utterance("张三", "请问单原子催化的稳定性问题怎么解决？")
    formatted = ctx.format_for_agent()
    assert "会议" in formatted or "topic" in formatted.lower() or "测试" in formatted
    assert "张三" in formatted
```

**Step 2: Run test to verify it fails**

```bash
./venv/bin/python -m pytest tests/interface/test_meeting_context.py -v
```
Expected: FAIL

**Step 3: Write implementation**

```python
# interface/meeting_context.py
"""Meeting context manager: maintains sliding window + rolling summary."""
import time
from agent.tokens import count_tokens


class MeetingContext:
    """Maintains meeting state within a fixed token budget.

    Keeps:
    - Meeting metadata (topic, participants, discussed topics)
    - Recent utterances as raw text (sliding window)
    - Rolling summary of older utterances
    """

    def __init__(self, window_minutes: float = 5.0,
                 max_tokens: int = 4000,
                 compressor_fn=None):
        """
        Args:
            window_minutes: How long to keep raw utterances before compressing.
            max_tokens: Max tokens for recent_utterances before triggering compression.
            compressor_fn: Optional fn(utterances) -> str for LLM compression.
        """
        self.window_minutes = window_minutes
        self.max_tokens = max_tokens
        self.compressor_fn = compressor_fn

        # Meeting metadata
        self.topic: str = ""
        self.participants: list[str] = []
        self.discussed_topics: list[str] = []

        # Content
        self.recent_utterances: list[dict] = []  # {speaker, text, timestamp}
        self.rolling_summary: str = ""

    def add_utterance(self, speaker: str, text: str):
        """Record a new utterance and compress if needed."""
        self.recent_utterances.append({
            "speaker": speaker,
            "text": text,
            "timestamp": time.time(),
        })
        # Add speaker to participants if new
        if speaker and speaker not in self.participants:
            self.participants.append(speaker)

        # Compress if over budget
        if self._count_recent_tokens() > self.max_tokens:
            self._compress_old_utterances()

    def get_snapshot(self) -> dict:
        """Return current meeting state as a dict."""
        recent_text = "\n".join(
            f"{u['speaker']}: {u['text']}"
            for u in self.recent_utterances[-20:]  # last 20 utterances max
        )
        return {
            "topic": self.topic,
            "participants": self.participants,
            "discussed_topics": self.discussed_topics,
            "rolling_summary": self.rolling_summary,
            "recent_text": recent_text,
        }

    def format_for_agent(self) -> str:
        """Format meeting context for injection into agent system prompt."""
        snap = self.get_snapshot()
        parts = ["## 当前会议上下文\n"]

        if snap["topic"]:
            parts.append(f"**会议主题：** {snap['topic']}\n")
        if snap["participants"]:
            parts.append(f"**参会者：** {', '.join(snap['participants'])}\n")
        if snap["discussed_topics"]:
            parts.append(
                f"**已讨论话题：** {', '.join(snap['discussed_topics'])}\n"
            )
        if snap["rolling_summary"]:
            parts.append(f"**早期讨论摘要：**\n{snap['rolling_summary']}\n")
        if snap["recent_text"]:
            parts.append(f"**最近发言：**\n{snap['recent_text']}\n")

        return "\n".join(parts)

    def _count_recent_tokens(self) -> int:
        text = " ".join(u["text"] for u in self.recent_utterances)
        return count_tokens(text)

    def _compress_old_utterances(self):
        """Compress the oldest half of utterances into rolling_summary."""
        if len(self.recent_utterances) < 4:
            return

        cutoff = len(self.recent_utterances) // 2
        to_compress = self.recent_utterances[:cutoff]
        self.recent_utterances = self.recent_utterances[cutoff:]

        compressed_text = "\n".join(
            f"{u['speaker']}: {u['text']}" for u in to_compress
        )

        if self.compressor_fn:
            new_summary = self.compressor_fn(to_compress)
        else:
            # Naive fallback: keep first 200 chars of each
            lines = [
                f"{u['speaker']}: {u['text'][:100]}"
                for u in to_compress
            ]
            new_summary = "\n".join(lines)

        if self.rolling_summary:
            self.rolling_summary = self.rolling_summary + "\n" + new_summary
        else:
            self.rolling_summary = new_summary
```

**Step 4: Run tests**

```bash
./venv/bin/python -m pytest tests/interface/test_meeting_context.py -v
```
Expected: 5 passed

**Step 5: Commit**

```bash
git add interface/meeting_context.py tests/interface/test_meeting_context.py
git commit -m "feat(interface): add MeetingContext with sliding window and rolling summary"
```

---

### Task 2: TriggerDetector — 发言时机判断

**Files:**
- Create: `interface/trigger_detector.py`
- Create: `tests/interface/test_trigger_detector.py`

**Step 1: Write the failing test**

```python
# tests/interface/test_trigger_detector.py
import pytest
from unittest.mock import MagicMock
from interface.trigger_detector import TriggerDetector, TriggerType


def _make_detector(twin_name="张三", domains=None):
    return TriggerDetector(
        twin_name=twin_name,
        confident_domains=domains or ["氢能催化剂", "电解水", "材料表征"],
    )


def test_detect_direct_mention():
    """Should trigger when twin is directly named."""
    detector = _make_detector(twin_name="张三")
    result = detector.check("李四：张三老师你怎么看这个数据？")
    assert result.should_speak is True
    assert result.trigger_type == TriggerType.DIRECT_MENTION
    assert result.urgency == "high"


def test_detect_domain_relevance():
    """Should suggest speaking when own domain is discussed."""
    detector = _make_detector(domains=["氢能催化剂", "HER"])
    result = detector.check("李四：最近HER催化剂的研究进展怎么样？")
    assert result.should_speak is True
    assert result.trigger_type == TriggerType.DOMAIN_RELEVANT


def test_no_trigger_for_irrelevant_content():
    """Should not trigger for off-domain content."""
    detector = _make_detector(domains=["氢能催化剂"])
    result = detector.check("李四：大家今天的午饭吃什么？")
    assert result.should_speak is False


def test_detect_question_in_domain():
    """Direct question about twin's domain should trigger."""
    detector = _make_detector(domains=["催化剂稳定性", "单原子催化"])
    result = detector.check("有没有人研究过单原子催化在长期稳定性方面的进展？")
    assert result.should_speak is True


def test_trigger_urgency_levels():
    """Direct mention should be higher urgency than domain match."""
    detector = _make_detector(twin_name="张三", domains=["氢能"])
    direct = detector.check("张三你来说说？")
    domain = detector.check("氢能方面有什么新进展？")
    assert direct.urgency == "high"
    assert domain.urgency in ("medium", "low")
```

**Step 2: Run test to verify it fails**

```bash
./venv/bin/python -m pytest tests/interface/test_trigger_detector.py -v
```
Expected: FAIL

**Step 3: Write implementation**

```python
# interface/trigger_detector.py
"""Determines when the digital twin should speak in a meeting."""
import re
from dataclasses import dataclass
from enum import Enum


class TriggerType(Enum):
    DIRECT_MENTION = "direct_mention"   # 被点名
    DOMAIN_RELEVANT = "domain_relevant" # 涉及专业领域
    QUESTION = "question"               # 领域内直接提问
    NONE = "none"


@dataclass
class TriggerResult:
    should_speak: bool
    trigger_type: TriggerType
    urgency: str  # "high" | "medium" | "low" | "none"
    reason: str = ""

    @classmethod
    def no_trigger(cls) -> "TriggerResult":
        return cls(False, TriggerType.NONE, "none")


class TriggerDetector:
    """Rule-based trigger detection for meeting participation."""

    def __init__(self, twin_name: str,
                 confident_domains: list[str],
                 min_domain_keywords: int = 1):
        self.twin_name = twin_name
        self.confident_domains = confident_domains
        self.min_domain_keywords = min_domain_keywords

        # Name variants (e.g., 张老师, 张三老师, etc.)
        parts = twin_name.split()
        surname = parts[0][0] if parts else twin_name[0]
        self._name_patterns = [
            twin_name,
            f"{surname}老师",
            f"{twin_name}老师",
            f"@{twin_name}",
        ]

    def check(self, utterance: str) -> TriggerResult:
        """Check if the latest utterance should trigger a response."""
        # 1. Direct mention (highest priority)
        for pattern in self._name_patterns:
            if pattern in utterance:
                return TriggerResult(
                    should_speak=True,
                    trigger_type=TriggerType.DIRECT_MENTION,
                    urgency="high",
                    reason=f"被点名：'{pattern}' 出现在发言中",
                )

        # 2. Domain keyword match
        matched = [d for d in self.confident_domains if d in utterance]
        if len(matched) >= self.min_domain_keywords:
            # Higher urgency if it's also a question
            is_question = "？" in utterance or "?" in utterance or \
                          any(w in utterance for w in ["怎么看", "如何", "有没有", "请问"])
            return TriggerResult(
                should_speak=True,
                trigger_type=TriggerType.DOMAIN_RELEVANT,
                urgency="medium" if is_question else "low",
                reason=f"涉及领域关键词：{matched}",
            )

        return TriggerResult.no_trigger()

    def check_transcript_window(self, utterances: list[dict],
                                 window: int = 3) -> TriggerResult:
        """Check the last N utterances for triggers."""
        recent = utterances[-window:] if len(utterances) >= window else utterances
        for utt in reversed(recent):
            result = self.check(utt.get("text", ""))
            if result.should_speak:
                return result
        return TriggerResult.no_trigger()
```

**Step 4: Run tests**

```bash
./venv/bin/python -m pytest tests/interface/test_trigger_detector.py -v
```
Expected: 5 passed

**Step 5: Commit**

```bash
git add interface/trigger_detector.py tests/interface/test_trigger_detector.py
git commit -m "feat(interface): add TriggerDetector for meeting participation timing"
```

---

### Task 3: multimodal/meeting_vision.py — 屏幕理解

**Files:**
- Create: `multimodal/meeting_vision.py`
- Create: `tests/multimodal/test_meeting_vision.py`

**Step 1: 安装 mss（轻量截图库）**

```bash
./venv/bin/pip install mss
```

追加到 requirements.txt：`mss>=9.0`

**Step 2: Write the failing test**

```python
# tests/multimodal/test_meeting_vision.py
import os
import pytest
from unittest.mock import MagicMock, patch
from multimodal.meeting_vision import ScreenCapture, analyze_screen_content


def test_screen_capture_init():
    cap = ScreenCapture(interval_seconds=10)
    assert cap.interval_seconds == 10
    assert cap.last_capture is None


def test_analyze_screen_content_calls_vision(tmp_path):
    """analyze_screen_content should call vision client."""
    from PIL import Image
    img_path = str(tmp_path / "screen.png")
    Image.new("RGB", (100, 50), color=(200, 200, 200)).save(img_path)

    mock_client = MagicMock()
    mock_client.vision_chat.return_value = "这是一张关于氢能催化剂的PPT幻灯片"

    result = analyze_screen_content(
        client=mock_client,
        image_path=img_path,
        meeting_topic="氢能催化剂进展",
    )
    assert "PPT" in result or "幻灯片" in result or len(result) > 0
    mock_client.vision_chat.assert_called_once()


def test_screen_capture_without_mss():
    """ScreenCapture.capture should raise if mss not installed."""
    with patch.dict("sys.modules", {"mss": None}):
        cap = ScreenCapture()
        with pytest.raises((ImportError, ModuleNotFoundError)):
            cap.capture()
```

**Step 3: Run test to verify it fails**

```bash
./venv/bin/python -m pytest tests/multimodal/test_meeting_vision.py -v
```
Expected: FAIL

**Step 4: Write implementation**

```python
# multimodal/meeting_vision.py
"""Screen capture and analysis for meeting participation."""
import base64
import os
import tempfile
import time

try:
    import mss
    import mss.tools
    HAS_MSS = True
except ImportError:
    mss = None
    HAS_MSS = False

SCREEN_ANALYSIS_PROMPT = """你正在参加一个学术会议（主题：{topic}）。
屏幕上当前显示的是共享的内容（可能是PPT、论文、数据图表或其他材料）。

请简洁描述：
1. 屏幕上显示了什么（内容类型、主要信息）
2. 与会议主题的关联
3. 是否有你（作为{domain}领域专家）需要关注的内容

如果无法识别内容或内容与会议无关，直接说明。"""


class ScreenCapture:
    """Periodically captures screen content for meeting analysis."""

    def __init__(self, interval_seconds: float = 30.0,
                 monitor: int = 1):
        self.interval_seconds = interval_seconds
        self.monitor = monitor
        self.last_capture: str | None = None
        self._last_capture_time: float = 0

    def capture(self, output_path: str | None = None) -> str:
        """Capture current screen. Returns path to saved image."""
        if not HAS_MSS:
            raise ImportError(
                "mss not installed. Run: pip install mss"
            )

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            output_path = tmp.name
            tmp.close()

        with mss.mss() as sct:
            monitors = sct.monitors
            mon = monitors[min(self.monitor, len(monitors) - 1)]
            screenshot = sct.grab(mon)
            mss.tools.to_png(screenshot.rgb, screenshot.size, output=output_path)

        self.last_capture = output_path
        self._last_capture_time = time.time()
        return output_path

    def should_capture(self) -> bool:
        """Returns True if enough time has passed since last capture."""
        return time.time() - self._last_capture_time >= self.interval_seconds

    def cleanup(self):
        """Delete temporary capture file."""
        if self.last_capture and os.path.exists(self.last_capture):
            try:
                os.unlink(self.last_capture)
            except OSError:
                pass
        self.last_capture = None


def analyze_screen_content(client, image_path: str,
                            meeting_topic: str = "",
                            domain: str = "材料科学/氢能") -> str:
    """Analyze a screen capture with meeting context."""
    from multimodal.vision import image_file_to_b64

    b64, media_type = image_file_to_b64(image_path)
    prompt = SCREEN_ANALYSIS_PROMPT.format(
        topic=meeting_topic or "学术讨论",
        domain=domain,
    )
    return client.vision_chat(
        prompt=prompt,
        image_b64=b64,
        media_type=media_type,
        max_tokens=400,
    )
```

**Step 5: Run tests**

```bash
./venv/bin/python -m pytest tests/multimodal/test_meeting_vision.py -v
```
Expected: 3 passed

**Step 6: Commit**

```bash
git add multimodal/meeting_vision.py tests/multimodal/test_meeting_vision.py requirements.txt
git commit -m "feat(multimodal): add screen capture and analysis for meeting participation"
```

---

### Task 4: interface/meeting.py — 会议主循环

**Files:**
- Create: `interface/meeting.py`
- Create: `tests/interface/test_meeting.py`

**依赖 Task 1、2、3 全部完成**

**Step 1: Write the failing test**

```python
# tests/interface/test_meeting.py
import os
import pytest
from unittest.mock import MagicMock, patch
from interface.meeting import MeetingBot, run_meeting_from_audio


def _make_bot(tmp_path):
    agent = MagicMock()
    agent.project_dir = str(tmp_path)
    agent.config = {"model": "test-model"}
    agent.chat = MagicMock(return_value="这个方向值得关注，稳定性是核心问题。")

    bot = MeetingBot(
        agent=agent,
        twin_name="张三",
        confident_domains=["氢能催化剂", "电解水"],
        voice="zh-CN-YunxiNeural",
    )
    return bot


def test_meeting_bot_init(tmp_path):
    bot = _make_bot(tmp_path)
    assert bot.twin_name == "张三"
    assert bot.meeting_context is not None
    assert bot.trigger_detector is not None


def test_process_utterance_no_trigger(tmp_path):
    """Utterance outside domain should not generate response."""
    bot = _make_bot(tmp_path)
    result = bot.process_utterance(
        speaker="李四",
        text="今天天气不错。",
    )
    assert result is None


def test_process_utterance_direct_mention(tmp_path):
    """Direct mention should generate response."""
    bot = _make_bot(tmp_path)
    with patch.object(bot, "_speak", return_value="audio.mp3") as mock_speak:
        result = bot.process_utterance(
            speaker="李四",
            text="张三你怎么看单原子催化剂的稳定性问题？",
        )
    assert result is not None
    mock_speak.assert_called_once()
    bot.agent.chat.assert_called_once()


def test_process_utterance_adds_to_context(tmp_path):
    """All utterances should be added to meeting context."""
    bot = _make_bot(tmp_path)
    bot.process_utterance("李四", "最近HER催化剂有什么新进展？")
    assert len(bot.meeting_context.recent_utterances) == 1


def test_run_meeting_from_audio(tmp_path):
    """run_meeting_from_audio should process a transcript file."""
    transcript_path = str(tmp_path / "transcript.txt")
    with open(transcript_path, "w") as f:
        f.write("李四：请问张三老师，氢能催化剂稳定性的问题怎么看？\n")
        f.write("王五：另外单原子催化的成本控制有什么思路？\n")

    agent = MagicMock()
    agent.project_dir = str(tmp_path)
    agent.config = {"model": "test-model"}
    agent.chat = MagicMock(return_value="稳定性是核心问题，1000圈是门槛。")

    responses = []
    with patch("interface.meeting.synthesize_speech") as mock_tts:
        mock_tts.return_value = "response.mp3"
        responses = run_meeting_from_audio(
            agent=agent,
            transcript_path=transcript_path,
            twin_name="张三",
            confident_domains=["氢能催化剂", "单原子催化"],
            auto_play=False,
        )

    assert len(responses) >= 1
    assert any("稳定性" in r["response"] for r in responses)
```

**Step 2: Run test to verify it fails**

```bash
./venv/bin/python -m pytest tests/interface/test_meeting.py -v
```
Expected: FAIL

**Step 3: Write implementation**

```python
# interface/meeting.py
"""Meeting participation: real-time or file-based meeting bot."""
import os
import tempfile
from interface.meeting_context import MeetingContext
from interface.trigger_detector import TriggerDetector
from multimodal.tts import synthesize_speech, play_audio, DEFAULT_VOICE


class MeetingBot:
    """Digital twin meeting participant."""

    def __init__(self, agent,
                 twin_name: str,
                 confident_domains: list[str],
                 voice: str = DEFAULT_VOICE,
                 auto_play: bool = True,
                 screen_interval: float = 30.0):
        self.agent = agent
        self.twin_name = twin_name
        self.confident_domains = confident_domains
        self.voice = voice
        self.auto_play = auto_play

        # Core components
        self.meeting_context = MeetingContext(
            compressor_fn=self._compress_context,
        )
        self.trigger_detector = TriggerDetector(
            twin_name=twin_name,
            confident_domains=confident_domains,
        )
        self._response_history: list[dict] = []

    def process_utterance(self, speaker: str, text: str) -> dict | None:
        """Process one utterance. Returns response dict if twin should speak."""
        # Always record to context
        if speaker != self.twin_name:
            self.meeting_context.add_utterance(speaker, text)

        # Check trigger
        trigger = self.trigger_detector.check(text)
        if not trigger.should_speak:
            return None

        # Build context-aware prompt
        meeting_ctx = self.meeting_context.format_for_agent()
        prompt = (
            f"{meeting_ctx}\n\n"
            f"**{speaker}刚才说：** {text}\n\n"
            f"请作为你自己（{self.twin_name}）在会议中自然地回应。"
            f"保持简洁，说1-3句话，适合口语表达。"
        )

        response = self.agent.chat(prompt)

        # Record own response
        self.meeting_context.add_utterance(self.twin_name, response)

        # Speak the response
        audio_path = self._speak(response)

        result = {
            "trigger": trigger.trigger_type.value,
            "urgency": trigger.urgency,
            "speaker": speaker,
            "input": text,
            "response": response,
            "audio_path": audio_path,
        }
        self._response_history.append(result)
        return result

    def update_screen(self, image_path: str):
        """Update meeting context with current screen content."""
        try:
            from multimodal.meeting_vision import analyze_screen_content
            description = analyze_screen_content(
                client=self.agent.client,
                image_path=image_path,
                meeting_topic=self.meeting_context.topic,
            )
            self.meeting_context.add_utterance(
                "[屏幕共享]", description
            )
        except Exception:
            pass  # Screen analysis is best-effort

    def _speak(self, text: str) -> str:
        """Synthesize and optionally play response."""
        audio_path = os.path.join(
            tempfile.gettempdir(), "twin_meeting_response.mp3"
        )
        synthesize_speech(text, audio_path, voice=self.voice)
        if self.auto_play:
            play_audio(audio_path)
        return audio_path

    def _compress_context(self, utterances: list[dict]) -> str:
        """Use agent LLM to compress old utterances."""
        text = "\n".join(f"{u['speaker']}: {u['text']}" for u in utterances)
        prompt = f"请将以下会议发言压缩成简洁摘要（不超过200字）：\n{text}\n摘要："
        return self.agent.client.simple_chat(prompt, max_tokens=250)


def run_meeting_from_audio(agent, transcript_path: str,
                            twin_name: str,
                            confident_domains: list[str],
                            voice: str = DEFAULT_VOICE,
                            auto_play: bool = True) -> list[dict]:
    """Process a meeting transcript file and generate responses.

    Transcript format (one utterance per line):
        Speaker: text
    """
    bot = MeetingBot(
        agent=agent,
        twin_name=twin_name,
        confident_domains=confident_domains,
        voice=voice,
        auto_play=auto_play,
    )

    responses = []
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Parse "Speaker: text" format
            if "：" in line:
                speaker, text = line.split("：", 1)
            elif ":" in line:
                speaker, text = line.split(":", 1)
            else:
                speaker, text = "未知", line

            result = bot.process_utterance(speaker.strip(), text.strip())
            if result:
                responses.append(result)

    return responses


def run_realtime_meeting(agent, twin_name: str,
                          confident_domains: list[str],
                          voice: str = DEFAULT_VOICE,
                          record_duration: float = 3.0):
    """Real-time meeting participation with microphone input."""
    from rich.console import Console
    from multimodal.stt import STTEngine

    console = Console()
    bot = MeetingBot(
        agent=agent,
        twin_name=twin_name,
        confident_domains=confident_domains,
        voice=voice,
        auto_play=True,
    )
    stt = STTEngine()

    console.print(f"[bold]实时会议模式[/bold] — 监听中 (Ctrl+C 退出)")
    console.print(f"分身名称：{twin_name} | 领域：{', '.join(confident_domains[:3])}")

    while True:
        try:
            # Record a short segment
            from multimodal.stt import record_once
            audio_path = record_once(duration=record_duration)
            text = stt.transcribe(audio_path)
            os.unlink(audio_path)

            if not text.strip():
                continue

            console.print(f"[dim]监听：{text[:80]}[/dim]")

            # Detect speaker (simple heuristic: not self)
            result = bot.process_utterance("参会者", text)
            if result:
                console.print(f"[cyan]发言：{result['response'][:100]}[/cyan]")

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]错误：{e}[/red]")
            continue

    agent.end_session()
    console.print("[dim]会议结束[/dim]")
```

**Step 4: Run tests**

```bash
./venv/bin/python -m pytest tests/interface/test_meeting.py -v
```
Expected: 5 passed

**Step 5: Commit**

```bash
git add interface/meeting.py tests/interface/test_meeting.py
git commit -m "feat(interface): add MeetingBot with real-time and file-based participation"
```

---

### Task 5: 全量回归测试 + push

**Step 1: Run all tests**

```bash
./venv/bin/python -m pytest tests/ -v
```
Expected: All passed (95+ tests)

**Step 2: Push**

```bash
git push origin main
```
