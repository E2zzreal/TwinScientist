# M3: 能摄入 — 半自动数据摄入管道实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现论文PDF→认知印象、录音→风格提取、交互式校准三条摄入管道，让数字分身的知识库可以半自动增长。

**Architecture:** 四个独立模块——paper_processor（PDF→印象YAML）、persona_extractor（从转写文本提炼风格特征）、audio_processor（录音→Whisper转写→调用persona_extractor）、interactive_init（CLI问答补盲校准）。Whisper为可选依赖，未安装时audio_processor接受已有转写文本。

**Tech Stack:** pypdf（PDF解析）, openai-whisper（可选，音频转写）, anthropic SDK（LLM提取）, PyYAML, rich

**新增依赖（追加到 requirements.txt）：**
```
pypdf>=4.0
openai-whisper>=20231117  # optional, for audio transcription
```

---

### Task 1: paper_processor — PDF → 认知印象 YAML

**Files:**
- Create: `ingestion/paper_processor.py`
- Create: `tests/ingestion/test_paper_processor.py`

**Step 1: 更新 requirements.txt，追加 pypdf**

在 requirements.txt 末尾追加：
```
pypdf>=4.0
```

安装：`./venv/bin/pip install pypdf`

**Step 2: Write the failing test**

```python
# tests/ingestion/test_paper_processor.py
import os
import yaml
import pytest
from unittest.mock import MagicMock, patch
from ingestion.paper_processor import extract_pdf_text, generate_impression, process_paper


def _make_fake_pdf(tmp_path) -> str:
    """Create a minimal valid PDF for testing."""
    # Use pypdf to create a test PDF with text
    try:
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        pdf_path = str(tmp_path / "test_paper.pdf")
        with open(pdf_path, "wb") as f:
            writer.write(f)
        return pdf_path
    except Exception:
        # fallback: write a minimal PDF bytes
        pdf_bytes = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
        pdf_path = str(tmp_path / "test_paper.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        return pdf_path


def test_extract_pdf_text_returns_string(tmp_path):
    pdf_path = _make_fake_pdf(tmp_path)
    text = extract_pdf_text(pdf_path)
    assert isinstance(text, str)


def test_extract_pdf_text_file_not_found():
    with pytest.raises(FileNotFoundError):
        extract_pdf_text("/nonexistent/paper.pdf")


def test_generate_impression_structure():
    """generate_impression should return a dict matching the schema."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="""
impression:
  one_sentence: "用MOF衍生碳负载单原子Pt，思路好但稳定性不足"
  key_takeaway: "单原子分散可降低Pt用量"
  attitude: skeptical_but_interested
  relevance_to_me: high
memorable_details:
  - "Fig.3的EXAFS数据漂亮"
connections: []
""")]
    mock_client.messages.create.return_value = mock_response

    result = generate_impression(
        client=mock_client,
        model="claude-sonnet-4-20250514",
        paper_text="Abstract: We report single-atom Pt on MOF-derived carbon for HER...",
        metadata={"title": "SAC for HER", "year": 2024},
    )
    assert "impression" in result
    assert "one_sentence" in result["impression"]
    assert "memorable_details" in result


def test_process_paper_creates_yaml(tmp_path):
    """process_paper should write a YAML file to memory/papers/."""
    pdf_path = _make_fake_pdf(tmp_path)
    memory_dir = str(tmp_path / "memory")
    os.makedirs(os.path.join(memory_dir, "papers"), exist_ok=True)

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="""
impression:
  one_sentence: "测试论文印象"
  key_takeaway: "关键结论"
  attitude: neutral
  relevance_to_me: medium
memorable_details: []
connections: []
""")]
    mock_client.messages.create.return_value = mock_response

    output_path = process_paper(
        client=mock_client,
        model="claude-sonnet-4-20250514",
        pdf_path=pdf_path,
        memory_dir=memory_dir,
        metadata={"title": "Test Paper", "authors": ["Test Author"], "year": 2024,
                  "field": ["hydrogen", "catalyst"]},
    )

    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        data = yaml.safe_load(f)
    assert "source" in data
    assert "impression" in data
    assert data["source"]["title"] == "Test Paper"
```

**Step 3: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/ingestion/test_paper_processor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ingestion.paper_processor'`

**Step 4: Write implementation**

```python
# ingestion/paper_processor.py
"""Convert research papers (PDF) into structured cognitive impressions."""
import os
import re
import yaml
from datetime import datetime

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

IMPRESSION_PROMPT = """你正在帮助一位材料科学家（专攻氢能）建立论文认知档案。

以下是一篇论文的文本内容（可能被截断）：

---
{paper_text}
---

论文元数据：
标题：{title}
年份：{year}

请提取该论文的认知印象，按以下YAML格式输出（只输出YAML，不要其他文字）：

impression:
  one_sentence: "一句话总结这篇论文的核心贡献和你的评价"
  key_takeaway: "最值得记住的一个技术点"
  attitude: "excited|interested|neutral|skeptical_but_interested|skeptical|critical 之一"
  relevance_to_me: "high|medium|low 之一"
memorable_details:
  - "具体的数据点、图表发现或实验结果（引用具体数字）"
  - "（可有多条，最多5条）"
connections: []
figure_impressions: []
"""


def extract_pdf_text(pdf_path: str, max_chars: int = 8000) -> str:
    """Extract text from PDF, truncated to max_chars."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not HAS_PYPDF:
        raise ImportError("pypdf not installed. Run: pip install pypdf")

    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    text = "\n".join(parts)
    return text[:max_chars]


def generate_impression(client, model: str, paper_text: str,
                        metadata: dict) -> dict:
    """Use LLM to generate a cognitive impression from paper text."""
    prompt = IMPRESSION_PROMPT.format(
        paper_text=paper_text[:6000],
        title=metadata.get("title", "Unknown"),
        year=metadata.get("year", "Unknown"),
    )
    response = client.messages.create(
        model=model,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:yaml)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        return yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        return {"impression": {"one_sentence": raw[:200],
                               "key_takeaway": "", "attitude": "neutral",
                               "relevance_to_me": "medium"},
                "memorable_details": [], "connections": []}


def process_paper(client, model: str, pdf_path: str, memory_dir: str,
                  metadata: dict) -> str:
    """Full pipeline: PDF → impression YAML saved to memory/papers/."""
    # Extract text
    text = extract_pdf_text(pdf_path)

    # Generate impression
    impression_data = generate_impression(client, model, text, metadata)

    # Build full record
    record = {
        "source": {
            "title": metadata.get("title", "Unknown"),
            "authors": metadata.get("authors", []),
            "year": metadata.get("year"),
            "field": metadata.get("field", []),
            "pdf_path": pdf_path,
        },
        **impression_data,
        "ingested_at": datetime.now().strftime("%Y-%m-%d"),
    }

    # Generate filename from title and year
    slug = re.sub(r"[^\w\s-]", "", metadata.get("title", "paper").lower())
    slug = re.sub(r"\s+", "-", slug.strip())[:40]
    year = metadata.get("year", "0000")
    filename = f"{year}-{slug}.yaml"

    papers_dir = os.path.join(memory_dir, "papers")
    os.makedirs(papers_dir, exist_ok=True)
    output_path = os.path.join(papers_dir, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(record, f, allow_unicode=True, default_flow_style=False)

    return output_path
```

**Step 5: Install pypdf and run tests**

```bash
./venv/bin/pip install pypdf
./venv/bin/python -m pytest tests/ingestion/test_paper_processor.py -v
```

Expected: 4 passed

**Step 6: Commit**

```bash
git add ingestion/paper_processor.py tests/ingestion/test_paper_processor.py requirements.txt
git commit -m "feat(ingestion): add PDF paper processor with LLM impression extraction"
```

---

### Task 2: persona_extractor — 从转写文本提炼风格特征

**Files:**
- Create: `ingestion/persona_extractor.py`
- Create: `tests/ingestion/test_persona_extractor.py`

**Step 1: Write the failing test**

```python
# tests/ingestion/test_persona_extractor.py
import os
import yaml
import pytest
from unittest.mock import MagicMock
from ingestion.persona_extractor import (
    extract_style_exemplars,
    extract_verbal_habits,
    extract_reasoning_patterns,
    merge_into_persona,
)

SAMPLE_TRANSCRIPT = """
[张三发言]
这个方向我觉得有意思，但你看这个稳定性数据，100圈就衰减了15%，
离实用差太远了。你跟商业化的Pt/C比过没有？
[李四发言]
我们还没有做这个对比实验。
[张三发言]
那这个就说不清楚了。想法是好的，但control experiment要做干净。
你看Fig.3，没有跟bulk材料对比，怎么证明是single atom的贡献？
"""


def test_extract_style_exemplars_returns_list():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="""
exemplars:
  - context: "评价实验数据"
    good: "你看这个稳定性数据，100圈就衰减了15%，离实用差太远了。"
    note: "锚定具体数据，不说空话"
  - context: "质疑实验设计"
    good: "control experiment要做干净，没有跟bulk对比，怎么证明是single atom的贡献？"
    note: "精确指出实验缺陷"
""")]
    mock_client.messages.create.return_value = mock_response

    result = extract_style_exemplars(
        client=mock_client,
        model="claude-sonnet-4-20250514",
        transcript=SAMPLE_TRANSCRIPT,
        speaker="张三",
    )
    assert isinstance(result, list)
    assert len(result) > 0
    assert "context" in result[0]
    assert "good" in result[0]


def test_extract_verbal_habits_returns_dict():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="""
verbal_habits:
  high_frequency_phrases:
    - "你看这个数据"
    - "离实用差太远"
  sentence_style: "短句为主，直接给结论"
  language_mix: "术语用英文（single atom, control experiment），论述用中文"
""")]
    mock_client.messages.create.return_value = mock_response

    result = extract_verbal_habits(
        client=mock_client,
        model="claude-sonnet-4-20250514",
        transcript=SAMPLE_TRANSCRIPT,
        speaker="张三",
    )
    assert "verbal_habits" in result
    assert "high_frequency_phrases" in result["verbal_habits"]


def test_extract_reasoning_patterns_returns_dict():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="""
reasoning_patterns:
  evaluating_experiment:
    steps:
      - "先问有没有对照实验"
      - "再看具体数据是否支撑claim"
    bias: "对缺乏对照实验的工作天然不信任"
""")]
    mock_client.messages.create.return_value = mock_response

    result = extract_reasoning_patterns(
        client=mock_client,
        model="claude-sonnet-4-20250514",
        transcript=SAMPLE_TRANSCRIPT,
        speaker="张三",
    )
    assert "reasoning_patterns" in result


def test_merge_into_persona(tmp_path):
    """merge_into_persona should update style.yaml with new exemplars."""
    persona_dir = str(tmp_path / "persona")
    os.makedirs(persona_dir, exist_ok=True)

    # Existing style.yaml with one exemplar
    existing = {"voice": {"summary": "直接", "exemplars": [
        {"context": "old", "good": "old good", "note": "old note"}
    ]}}
    style_path = os.path.join(persona_dir, "style.yaml")
    with open(style_path, "w") as f:
        yaml.dump(existing, f, allow_unicode=True)

    new_exemplars = [
        {"context": "评价数据", "good": "100圈就衰减了15%，离实用差太远", "note": "锚定数据"}
    ]
    merge_into_persona(persona_dir, new_exemplars=new_exemplars)

    with open(style_path, "r") as f:
        updated = yaml.safe_load(f)

    assert len(updated["voice"]["exemplars"]) == 2
    assert any("100圈" in e["good"] for e in updated["voice"]["exemplars"])
```

**Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/ingestion/test_persona_extractor.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
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
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
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
```

**Step 4: Run tests**

Run: `./venv/bin/python -m pytest tests/ingestion/test_persona_extractor.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add ingestion/persona_extractor.py tests/ingestion/test_persona_extractor.py
git commit -m "feat(ingestion): add persona extractor from transcripts"
```

---

### Task 3: audio_processor — 录音转写 + 风格提取

**Files:**
- Create: `ingestion/audio_processor.py`
- Create: `tests/ingestion/test_audio_processor.py`

**依赖 Task 2（调用 persona_extractor）**

**Step 1: Write the failing test**

```python
# tests/ingestion/test_audio_processor.py
import os
import yaml
import pytest
from unittest.mock import MagicMock, patch
from ingestion.audio_processor import transcribe_audio, process_audio


def test_transcribe_audio_without_whisper():
    """If whisper not installed, should raise ImportError with helpful message."""
    with patch.dict("sys.modules", {"whisper": None}):
        with pytest.raises((ImportError, ModuleNotFoundError)):
            transcribe_audio("/fake/audio.mp3")


def test_process_audio_from_existing_transcript(tmp_path):
    """process_audio should accept pre-existing transcript text."""
    persona_dir = str(tmp_path / "persona")
    os.makedirs(persona_dir, exist_ok=True)
    style_path = os.path.join(persona_dir, "style.yaml")
    with open(style_path, "w") as f:
        yaml.dump({"voice": {"summary": "", "exemplars": []}}, f,
                  allow_unicode=True)

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="""
exemplars:
  - context: "评价实验"
    good: "你看这个数据，100圈就衰减了15%"
    note: "锚定数据"
""")]
    mock_client.messages.create.return_value = mock_response

    result = process_audio(
        client=mock_client,
        model="claude-sonnet-4-20250514",
        persona_dir=str(persona_dir),
        speaker="张三",
        transcript_text="[张三] 你看这个数据，100圈就衰减了15%，离实用差太远了。",
    )

    assert result["exemplars_added"] >= 0
    with open(style_path, "r") as f:
        style = yaml.safe_load(f)
    assert "exemplars" in style["voice"]


def test_process_audio_saves_transcript(tmp_path):
    """process_audio should save transcript to file if provided."""
    persona_dir = str(tmp_path / "persona")
    os.makedirs(persona_dir, exist_ok=True)
    with open(os.path.join(persona_dir, "style.yaml"), "w") as f:
        yaml.dump({"voice": {"summary": "", "exemplars": []}}, f)

    transcript_save_path = str(tmp_path / "transcript.txt")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="exemplars: []")]
    mock_client.messages.create.return_value = mock_response

    process_audio(
        client=mock_client,
        model="claude-sonnet-4-20250514",
        persona_dir=str(persona_dir),
        speaker="张三",
        transcript_text="some transcript",
        save_transcript_to=transcript_save_path,
    )

    assert os.path.exists(transcript_save_path)
```

**Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/ingestion/test_audio_processor.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# ingestion/audio_processor.py
"""Convert audio recordings into transcripts and extract persona features."""
from ingestion.persona_extractor import (
    extract_style_exemplars,
    extract_verbal_habits,
    extract_reasoning_patterns,
    merge_into_persona,
)


def transcribe_audio(audio_path: str, language: str = "zh") -> str:
    """Transcribe audio file using Whisper. Requires whisper to be installed."""
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "openai-whisper is not installed.\n"
            "Install it with: pip install openai-whisper\n"
            "Or provide transcript_text directly to process_audio()."
        )
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language=language)
    return result["text"]


def process_audio(client, model: str, persona_dir: str, speaker: str,
                  audio_path: str | None = None,
                  transcript_text: str | None = None,
                  save_transcript_to: str | None = None) -> dict:
    """Full pipeline: audio (or transcript) → persona features.

    Either audio_path or transcript_text must be provided.
    If audio_path is given, Whisper transcribes it first.
    """
    if transcript_text is None and audio_path is None:
        raise ValueError("Either audio_path or transcript_text must be provided.")

    # Transcribe if needed
    if transcript_text is None:
        transcript_text = transcribe_audio(audio_path)

    # Optionally save transcript
    if save_transcript_to:
        with open(save_transcript_to, "w", encoding="utf-8") as f:
            f.write(transcript_text)

    # Extract features
    exemplars = extract_style_exemplars(client, model, transcript_text, speaker)
    habits = extract_verbal_habits(client, model, transcript_text, speaker)
    patterns = extract_reasoning_patterns(client, model, transcript_text, speaker)

    # Merge into persona files
    merge_into_persona(
        persona_dir,
        new_exemplars=exemplars,
        verbal_habits=habits,
        reasoning_patterns=patterns,
    )

    return {
        "exemplars_added": len(exemplars),
        "verbal_habits_extracted": bool(habits),
        "reasoning_patterns_extracted": bool(patterns),
        "transcript_length": len(transcript_text),
    }
```

**Step 4: Run tests**

Run: `./venv/bin/python -m pytest tests/ingestion/test_audio_processor.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add ingestion/audio_processor.py tests/ingestion/test_audio_processor.py
git commit -m "feat(ingestion): add audio processor with Whisper transcription support"
```

---

### Task 4: interactive_init — CLI 交互式校准

**Files:**
- Create: `ingestion/interactive_init.py`
- Create: `tests/ingestion/test_interactive_init.py`

**Step 1: Write the failing test**

```python
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
        {"field": "research_focus", "value": ["氢能催化剂", "电解水"]},
        {"field": "personality_sketch",
         "value": "说话直接，喜欢用数据说话。"},
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
```

**Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/ingestion/test_interactive_init.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# ingestion/interactive_init.py
"""Interactive initialization: CLI Q&A to fill gaps and calibrate persona."""
import os
import yaml

# Questions targeting commonly empty fields
BASE_QUESTIONS = [
    {
        "question": "你的主要研究方向是什么？（用逗号分隔多个方向）",
        "field": "research_focus",
        "type": "list",
        "target_file": "identity.yaml",
    },
    {
        "question": "用一两句话描述你的说话和思维风格？",
        "field": "personality_sketch",
        "type": "text",
        "target_file": "identity.yaml",
    },
    {
        "question": "你最确信的学术观点或信念是什么？",
        "field": "core_beliefs",
        "type": "belief",
        "target_file": "identity.yaml",
    },
    {
        "question": "你非常熟悉的领域有哪些？（用逗号分隔）",
        "field": "confident_domains",
        "type": "list",
        "target_file": "boundaries.yaml",
    },
    {
        "question": "你了解但不深入的领域？（用逗号分隔）",
        "field": "familiar_but_not_expert",
        "type": "list",
        "target_file": "boundaries.yaml",
    },
    {
        "question": "评价论文时你最先看什么？",
        "field": "review_approach",
        "type": "text",
        "target_file": "style.yaml",
    },
    {
        "question": "什么样的研究工作让你感到兴奋？",
        "field": "excited_by",
        "type": "list",
        "target_file": "identity.yaml",
    },
    {
        "question": "什么样的论文或工作让你持怀疑态度？",
        "field": "skeptical_of",
        "type": "list",
        "target_file": "identity.yaml",
    },
]


def _is_field_empty(persona_dir: str, field: str, target_file: str) -> bool:
    """Check if a persona field is empty or unset."""
    path = os.path.join(persona_dir, target_file)
    if not os.path.exists(path):
        return True
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Navigate nested fields
    if target_file == "identity.yaml":
        val = data.get(field) or data.get("taste_profile", {}).get(field)
    else:
        val = data.get(field)

    if val is None:
        return True
    if isinstance(val, (list, str)) and len(val) == 0:
        return True
    if isinstance(val, str) and not val.strip():
        return True
    return False


def build_calibration_questions(persona_dir: str) -> list:
    """Return questions for fields that are empty in the current persona."""
    questions = []
    for q in BASE_QUESTIONS:
        if _is_field_empty(persona_dir, q["field"], q["target_file"]):
            questions.append(q)
    return questions


def _parse_answer(answer: str, q_type: str) -> object:
    """Parse raw string answer based on question type."""
    if q_type == "list":
        return [x.strip() for x in answer.split("，") + answer.split(",")
                if x.strip()]
    elif q_type == "belief":
        return [{"claim": answer.strip(), "confidence": "medium", "origin": "自述"}]
    else:
        return answer.strip()


def run_gap_filling(persona_dir: str, max_questions: int = 20) -> list:
    """Interactive CLI loop to collect answers for empty fields."""
    questions = build_calibration_questions(persona_dir)[:max_questions]
    corrections = []

    for q in questions:
        print(f"\n{q['question']}")
        print("（直接回车跳过）", end=" ")
        try:
            answer = input().strip()
        except EOFError:
            answer = ""

        if not answer:
            continue

        parsed = _parse_answer(answer, q["type"])
        corrections.append({
            "field": q["field"],
            "value": parsed,
            "target_file": q["target_file"],
        })

    return corrections


def apply_corrections(persona_dir: str, corrections: list):
    """Write user answers back into persona YAML files."""
    # Group by target file
    by_file: dict[str, list] = {}
    for c in corrections:
        by_file.setdefault(c["target_file"], []).append(c)

    for target_file, items in by_file.items():
        path = os.path.join(persona_dir, target_file)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}

        for item in items:
            field = item["field"]
            value = item["value"]

            # Handle taste_profile nested fields
            if field in ("excited_by", "skeptical_of"):
                if "taste_profile" not in data or not isinstance(data["taste_profile"], dict):
                    data["taste_profile"] = {}
                data["taste_profile"][field] = value
            else:
                data[field] = value

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def run_interactive_init(persona_dir: str):
    """Full interactive initialization flow."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print(Panel(
        "[bold]Twin Scientist — 交互式人格初始化[/bold]\n"
        "以下问题帮助建立你的数字分身基础档案。\n"
        "直接回车可跳过任何问题。",
    ))

    corrections = run_gap_filling(persona_dir)

    if corrections:
        apply_corrections(persona_dir, corrections)
        console.print(f"\n[green]已更新 {len(corrections)} 个人格字段。[/green]")
    else:
        console.print("\n[dim]没有需要补充的内容。[/dim]")
```

**Step 4: Run tests**

Run: `./venv/bin/python -m pytest tests/ingestion/test_interactive_init.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add ingestion/interactive_init.py tests/ingestion/test_interactive_init.py
git commit -m "feat(ingestion): add interactive persona initialization and calibration"
```

---

### Task 5: 全量回归测试 + push

**Step 1: Run all tests**

Run: `./venv/bin/python -m pytest tests/ -v`
Expected: All passed (40+ tests)

**Step 2: Update requirements.txt and push**

```bash
git add requirements.txt
git commit -m "chore: add pypdf to requirements for paper ingestion"
git push origin main
```
