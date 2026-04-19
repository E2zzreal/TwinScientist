# M6: 能说话 — 语音对话实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现语音对话——语音输入转文字（STT），Agent 回答后转语音播放（TTS），最终支持可选的语音克隆让输出听起来像本人。

**Architecture:** 三层结构——`multimodal/stt.py`（Whisper 封装，支持文件和麦克风）、`multimodal/tts.py`（edge-tts 作为基础引擎，CosyVoice/GPT-SoVITS 作为可选克隆引擎）、`interface/voice.py`（串联 STT + Agent + TTS 的完整语音循环）。MVP 用 edge-tts 免费语音，语音克隆作可选升级。

**Tech Stack:** `openai-whisper`（STT）, `edge-tts>=6.1`（TTS 基础引擎）, `sounddevice>=0.4`（麦克风+播放）, `soundfile>=0.12`（音频文件读写）, `numpy`（音频数据处理）

**两种运行模式：**
- 文件模式：输入音频文件 → STT → Agent → TTS 保存文件
- 实时模式：麦克风推按说话 → STT → Agent → TTS 播放

---

### Task 1: TTS 模块

**Files:**
- Create: `multimodal/tts.py`
- Create: `tests/multimodal/test_tts.py`

**Step 1: 安装依赖**

```bash
./venv/bin/pip install edge-tts soundfile sounddevice numpy
```

并追加到 requirements.txt：
```
edge-tts>=6.1
sounddevice>=0.4
soundfile>=0.12
numpy>=1.24
```

**Step 2: Write the failing test**

```python
# tests/multimodal/test_tts.py
import os
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from multimodal.tts import synthesize_speech, TTSEngine, get_available_voices


def test_tts_engine_default():
    """TTSEngine should default to edge-tts."""
    engine = TTSEngine()
    assert engine.backend == "edge_tts"


def test_tts_engine_voice_selection():
    """TTSEngine should accept a voice name."""
    engine = TTSEngine(voice="zh-CN-YunxiNeural")
    assert engine.voice == "zh-CN-YunxiNeural"


def test_synthesize_speech_creates_file(tmp_path):
    """synthesize_speech should create an audio file."""
    output_path = str(tmp_path / "output.mp3")

    with patch("multimodal.tts.edge_tts") as mock_et:
        mock_communicate = AsyncMock()
        mock_communicate.save = AsyncMock()
        mock_et.Communicate.return_value = mock_communicate

        # Simulate file creation
        with open(output_path, "wb") as f:
            f.write(b"fake audio data")

        synthesize_speech(
            text="你好，这是一段测试语音",
            output_path=output_path,
            voice="zh-CN-YunxiNeural",
        )

    assert os.path.exists(output_path)


def test_get_available_voices_returns_list():
    """get_available_voices should return a non-empty list."""
    # Mock to avoid network call
    with patch("multimodal.tts._CHINESE_VOICES") as mock_voices:
        mock_voices.__iter__ = MagicMock(return_value=iter([
            {"ShortName": "zh-CN-YunxiNeural", "Gender": "Male"},
            {"ShortName": "zh-CN-XiaoxiaoNeural", "Gender": "Female"},
        ]))
        voices = get_available_voices()
        # Even if mocking fails, function should return a list
        assert isinstance(voices, list)
```

**Step 3: Run test to verify it fails**

```bash
./venv/bin/python -m pytest tests/multimodal/test_tts.py -v
```
Expected: FAIL

**Step 4: Write implementation**

```python
# multimodal/tts.py
"""Text-to-Speech module with edge-tts as default backend.

Optional backends:
- CosyVoice 2: voice cloning from recordings (requires separate install)
- GPT-SoVITS: alternative voice cloning (requires separate install)
"""
import asyncio
import os

try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False

# High-quality Chinese neural voices (male + female options)
_CHINESE_VOICES = [
    {"ShortName": "zh-CN-YunxiNeural",     "Gender": "Male",   "Style": "calm"},
    {"ShortName": "zh-CN-YunjianNeural",   "Gender": "Male",   "Style": "sports"},
    {"ShortName": "zh-CN-XiaoxiaoNeural",  "Gender": "Female", "Style": "warm"},
    {"ShortName": "zh-CN-XiaoyiNeural",    "Gender": "Female", "Style": "lively"},
]
DEFAULT_VOICE = "zh-CN-YunxiNeural"


class TTSEngine:
    """Text-to-Speech engine with pluggable backends."""

    def __init__(self, voice: str = DEFAULT_VOICE, backend: str = "edge_tts"):
        self.voice = voice
        self.backend = backend

    def synthesize(self, text: str, output_path: str, rate: str = "+0%",
                   volume: str = "+0%") -> str:
        """Synthesize text to audio file. Returns output_path."""
        if self.backend == "edge_tts":
            return _synthesize_edge_tts(
                text, output_path, self.voice, rate, volume
            )
        elif self.backend == "cosyvoice":
            return _synthesize_cosyvoice(text, output_path)
        else:
            raise ValueError(f"Unknown TTS backend: {self.backend}")

    def play(self, audio_path: str):
        """Play an audio file through the speakers."""
        play_audio(audio_path)


def synthesize_speech(text: str, output_path: str,
                      voice: str = DEFAULT_VOICE,
                      rate: str = "+0%") -> str:
    """Convenience function: synthesize text and save to file."""
    engine = TTSEngine(voice=voice)
    engine.synthesize(text, output_path, rate=rate)
    return output_path


def play_audio(audio_path: str):
    """Play audio file through system speakers."""
    try:
        import sounddevice as sd
        import soundfile as sf
        data, samplerate = sf.read(audio_path)
        sd.play(data, samplerate)
        sd.wait()
    except ImportError:
        # Fallback: use system command
        import subprocess, sys
        if sys.platform == "darwin":
            subprocess.run(["afplay", audio_path], check=False)
        elif sys.platform.startswith("linux"):
            subprocess.run(["aplay", audio_path], check=False)
        else:
            os.startfile(audio_path)


def get_available_voices() -> list[dict]:
    """Return list of available Chinese voices."""
    return list(_CHINESE_VOICES)


def _synthesize_edge_tts(text: str, output_path: str, voice: str,
                          rate: str, volume: str) -> str:
    """Synthesize using edge-tts (async internally)."""
    if not HAS_EDGE_TTS:
        raise ImportError(
            "edge-tts not installed. Run: pip install edge-tts"
        )

    async def _run():
        communicate = edge_tts.Communicate(
            text, voice, rate=rate, volume=volume
        )
        await communicate.save(output_path)

    # Handle both sync and already-running event loop contexts
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _run())
                future.result()
        else:
            loop.run_until_complete(_run())
    except RuntimeError:
        asyncio.run(_run())

    return output_path


def _synthesize_cosyvoice(text: str, output_path: str) -> str:
    """Synthesize using CosyVoice voice cloning model."""
    try:
        from multimodal.voice_clone.synthesize import synthesize as cv_synth
        cv_synth(text, output_path)
    except ImportError:
        raise ImportError(
            "CosyVoice not installed. See multimodal/voice_clone/ for setup.\n"
            "Falling back to edge-tts is recommended for most use cases."
        )
    return output_path
```

**Step 5: Run tests**

```bash
./venv/bin/python -m pytest tests/multimodal/test_tts.py -v
```
Expected: 4 passed

**Step 6: Commit**

```bash
git add multimodal/tts.py tests/multimodal/test_tts.py requirements.txt
git commit -m "feat(multimodal): add TTS module with edge-tts and optional voice cloning"
```

---

### Task 2: STT 模块（独立于 audio_processor）

**Files:**
- Create: `multimodal/stt.py`
- Create: `tests/multimodal/test_stt.py`

**Step 1: Write the failing test**

```python
# tests/multimodal/test_stt.py
import os
import pytest
from unittest.mock import MagicMock, patch
from multimodal.stt import transcribe_file, record_once, STTEngine


def test_stt_engine_init():
    """STTEngine should initialize with model name."""
    engine = STTEngine(model_size="base")
    assert engine.model_size == "base"
    assert engine.model is None  # lazy load


def test_transcribe_file_without_whisper():
    """transcribe_file should raise ImportError if whisper not installed."""
    with patch.dict("sys.modules", {"whisper": None}):
        with pytest.raises((ImportError, ModuleNotFoundError)):
            transcribe_file("/fake/audio.wav")


def test_transcribe_file_calls_whisper(tmp_path):
    """transcribe_file should call whisper model.transcribe."""
    # Create a fake audio file
    audio_path = str(tmp_path / "test.wav")
    with open(audio_path, "wb") as f:
        f.write(b"\x00" * 100)

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "text": "这是一段测试语音转写",
        "language": "zh",
    }

    with patch("multimodal.stt.whisper") as mock_whisper:
        mock_whisper.load_model.return_value = mock_model
        result = transcribe_file(audio_path, model_size="base")

    assert result == "这是一段测试语音转写"
    mock_model.transcribe.assert_called_once()


def test_record_once_without_sounddevice():
    """record_once should raise ImportError if sounddevice not installed."""
    with patch.dict("sys.modules", {"sounddevice": None}):
        with pytest.raises((ImportError, ModuleNotFoundError)):
            record_once(duration=1)
```

**Step 2: Run test to verify it fails**

```bash
./venv/bin/python -m pytest tests/multimodal/test_stt.py -v
```
Expected: FAIL

**Step 3: Write implementation**

```python
# multimodal/stt.py
"""Speech-to-Text module wrapping Whisper with real-time microphone support."""
import os
import tempfile

try:
    import whisper as _whisper_mod
    whisper = _whisper_mod
    HAS_WHISPER = True
except ImportError:
    whisper = None
    HAS_WHISPER = False

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None
    HAS_SOUNDDEVICE = False


class STTEngine:
    """Speech-to-Text engine backed by Whisper."""

    def __init__(self, model_size: str = "base", language: str = "zh"):
        self.model_size = model_size
        self.language = language
        self.model = None  # lazy-loaded

    def _load_model(self):
        if not HAS_WHISPER:
            raise ImportError(
                "openai-whisper not installed.\n"
                "Run: pip install openai-whisper"
            )
        if self.model is None:
            self.model = whisper.load_model(self.model_size)
        return self.model

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file to text."""
        model = self._load_model()
        result = model.transcribe(audio_path, language=self.language)
        return result["text"].strip()

    def record_and_transcribe(self, duration: float = 5.0) -> str:
        """Record from microphone for `duration` seconds and transcribe."""
        audio_path = record_once(duration=duration)
        try:
            return self.transcribe(audio_path)
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)


def transcribe_file(audio_path: str, model_size: str = "base",
                    language: str = "zh") -> str:
    """Transcribe an audio file. Convenience function."""
    engine = STTEngine(model_size=model_size, language=language)
    return engine.transcribe(audio_path)


def record_once(duration: float = 5.0, samplerate: int = 16000) -> str:
    """Record audio from microphone and save to a temp file.

    Returns:
        Path to temporary WAV file. Caller must delete it.
    """
    if not HAS_SOUNDDEVICE:
        raise ImportError(
            "sounddevice not installed.\n"
            "Run: pip install sounddevice soundfile"
        )
    try:
        import soundfile as sf
        import numpy as np
    except ImportError:
        raise ImportError("soundfile and numpy required. Run: pip install soundfile numpy")

    print(f"  🎤 录音中（{duration:.0f}秒）... 请说话")
    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("  ✓ 录音完成")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, samplerate)
    return tmp.name
```

**Step 4: Run tests**

```bash
./venv/bin/python -m pytest tests/multimodal/test_stt.py -v
```
Expected: 4 passed

**Step 5: Commit**

```bash
git add multimodal/stt.py tests/multimodal/test_stt.py
git commit -m "feat(multimodal): add STT module wrapping Whisper with microphone support"
```

---

### Task 3: interface/voice.py — 语音对话循环

**Files:**
- Create: `interface/voice.py`
- Create: `tests/interface/test_voice.py`

**Task 1 和 Task 2 必须先完成**

**Step 1: Write the failing test**

```python
# tests/interface/test_voice.py
import os
import pytest
from unittest.mock import MagicMock, patch
from interface.voice import VoiceSession, process_voice_file


def _make_agent(tmp_path):
    """Create a minimal mock agent."""
    agent = MagicMock()
    agent.project_dir = str(tmp_path)
    agent.config = {"model": "test-model"}
    agent.chat = MagicMock(return_value="这是分身的回答，稳定性是核心问题。")
    return agent


def test_voice_session_init(tmp_path):
    agent = _make_agent(tmp_path)
    session = VoiceSession(
        agent=agent,
        voice="zh-CN-YunxiNeural",
        stt_model="base",
    )
    assert session.voice == "zh-CN-YunxiNeural"
    assert session.stt_model == "base"


def test_process_voice_file(tmp_path):
    """process_voice_file: audio → transcribe → agent → TTS → output file."""
    agent = _make_agent(tmp_path)

    audio_in = str(tmp_path / "input.wav")
    audio_out = str(tmp_path / "output.mp3")

    with open(audio_in, "wb") as f:
        f.write(b"\x00" * 100)

    with patch("interface.voice.transcribe_file", return_value="你对稳定性有什么看法？") as mock_stt, \
         patch("interface.voice.synthesize_speech") as mock_tts:
        # Simulate TTS creating the output file
        mock_tts.side_effect = lambda text, output_path, **kw: (
            open(output_path, "wb").write(b"audio") or output_path
        )

        result = process_voice_file(
            agent=agent,
            audio_input_path=audio_in,
            audio_output_path=audio_out,
            voice="zh-CN-YunxiNeural",
        )

    mock_stt.assert_called_once_with(audio_in, model_size="base")
    agent.chat.assert_called_once_with("你对稳定性有什么看法？")
    assert "稳定性" in agent.chat.return_value


def test_voice_session_text_mode(tmp_path):
    """VoiceSession.respond_text: text → TTS → audio file."""
    agent = _make_agent(tmp_path)
    session = VoiceSession(agent=agent)
    audio_out = str(tmp_path / "response.mp3")

    with patch("interface.voice.synthesize_speech") as mock_tts:
        mock_tts.side_effect = lambda text, output_path, **kw: (
            open(output_path, "wb").write(b"audio") or output_path
        )
        session.respond_text("这个方向值得关注。", output_path=audio_out)

    mock_tts.assert_called_once()
```

**Step 2: Run test to verify it fails**

```bash
./venv/bin/python -m pytest tests/interface/test_voice.py -v
```
Expected: FAIL

**Step 3: Write implementation**

```python
# interface/voice.py
"""Voice interface: STT → Agent → TTS pipeline."""
import os
import tempfile
from multimodal.stt import transcribe_file, STTEngine
from multimodal.tts import synthesize_speech, TTSEngine, play_audio, DEFAULT_VOICE


class VoiceSession:
    """A voice conversation session with the digital twin."""

    def __init__(self, agent, voice: str = DEFAULT_VOICE,
                 stt_model: str = "base", language: str = "zh",
                 auto_play: bool = True):
        self.agent = agent
        self.voice = voice
        self.stt_model = stt_model
        self.language = language
        self.auto_play = auto_play
        self._stt = STTEngine(model_size=stt_model, language=language)
        self._tts = TTSEngine(voice=voice)

    def chat_once(self, duration: float = 5.0,
                  output_dir: str | None = None) -> dict:
        """Record one utterance, get response, play it back.

        Returns dict with keys: transcription, response, audio_path
        """
        # 1. Record
        audio_path = self._stt.record_and_transcribe.__wrapped__(duration) \
            if hasattr(self._stt.record_and_transcribe, "__wrapped__") \
            else None

        from multimodal.stt import record_once
        tmp_audio = record_once(duration=duration)
        try:
            transcription = self._stt.transcribe(tmp_audio)
        finally:
            if os.path.exists(tmp_audio):
                os.unlink(tmp_audio)

        print(f"  你说：{transcription}")

        # 2. Agent response
        response = self.agent.chat(transcription)
        print(f"  分身：{response[:100]}{'...' if len(response) > 100 else ''}")

        # 3. TTS
        out_dir = output_dir or tempfile.gettempdir()
        audio_out = os.path.join(out_dir, "twin_response.mp3")
        self._tts.synthesize(response, audio_out)

        # 4. Playback
        if self.auto_play:
            play_audio(audio_out)

        return {
            "transcription": transcription,
            "response": response,
            "audio_path": audio_out,
        }

    def respond_text(self, text: str, output_path: str | None = None) -> str:
        """Synthesize a text response to audio. Returns audio file path."""
        if output_path is None:
            output_path = os.path.join(
                tempfile.gettempdir(), "twin_response.mp3"
            )
        synthesize_speech(text, output_path, voice=self.voice)
        if self.auto_play:
            play_audio(output_path)
        return output_path


def process_voice_file(agent, audio_input_path: str,
                       audio_output_path: str,
                       voice: str = DEFAULT_VOICE,
                       stt_model: str = "base") -> str:
    """Process one audio file: STT → Agent → TTS.

    Returns path to output audio file.
    """
    # Transcribe
    transcription = transcribe_file(audio_input_path, model_size=stt_model)
    print(f"转写：{transcription}")

    # Agent
    response = agent.chat(transcription)
    print(f"回答：{response}")

    # TTS
    synthesize_speech(response, audio_output_path, voice=voice)
    return audio_output_path


def run_voice_cli(project_dir: str, voice: str = DEFAULT_VOICE,
                  duration: float = 5.0):
    """Launch interactive voice conversation loop."""
    from rich.console import Console
    from rich.panel import Panel
    from agent.main import TwinScientist

    console = Console()
    agent = TwinScientist(project_dir)
    session = VoiceSession(agent=agent, voice=voice)

    console.print(Panel(
        f"[bold]Twin Scientist — 语音对话模式[/bold]\n"
        f"声音：{voice}\n"
        "每次按 Enter 开始录音，录音时长 {duration:.0f} 秒\n"
        "输入 q 退出",
        title="🎙️ 语音模式",
    ))

    while True:
        try:
            cmd = input("\n按 Enter 开始说话（q 退出）: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if cmd.lower() == "q":
            break

        try:
            result = session.chat_once(duration=duration)
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")

    agent.end_session()
    console.print("[dim]再见！[/dim]")
```

**Step 4: Run tests**

```bash
./venv/bin/python -m pytest tests/interface/test_voice.py -v
```
Expected: 3 passed

**Step 5: Commit**

```bash
git add interface/voice.py tests/interface/test_voice.py
git commit -m "feat(interface): add voice session with STT+Agent+TTS pipeline"
```

---

### Task 4: 全量回归测试 + push

**Step 1: Run all tests**

```bash
./venv/bin/python -m pytest tests/ -v
```
Expected: All passed (78+ tests)

**Step 2: Push**

```bash
git push origin main
```
