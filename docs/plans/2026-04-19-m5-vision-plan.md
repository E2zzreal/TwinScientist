# M5: 能看图 — 图像理解实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让数字分身能看懂论文图表、实验截图和PPT，回答"这张XRD图说明了什么"、"这条极化曲线的数据如何"等问题。

**Architecture:** 在 LLMClient 中增加 `vision_chat()` 方法统一处理 Anthropic/OpenAI 两种图像格式；新建 `multimodal/vision.py` 封装图像描述逻辑；Agent 新增 `see` 工具（接收文件路径或 base64），输出带有科研人员视角的图像解读；ingestion 的 paper_processor 可选地提取 PDF 页面截图并生成图表印象。

**Tech Stack:** 复用 LLMClient 适配层，新增 `pdf2image`（可选，PDF页面转图片），pillow（图像处理），base64 编码。

**注意：** 并非所有 OpenAI-compatible 服务商都支持视觉模型。系统在不支持时应优雅降级，提示用户切换到支持视觉的模型（如 gpt-4o、qwen-vl-plus、deepseek-vl2）。

---

### Task 1: LLMClient 增加 vision_chat() 方法

**Files:**
- Modify: `agent/llm_client.py`
- Modify: `tests/agent/test_llm_client.py`（新建）

**Step 1: Write the failing test**

```python
# tests/agent/test_llm_client.py
import base64
import pytest
from unittest.mock import MagicMock, patch
from agent.llm_client import LLMClient


def _make_config(provider="anthropic"):
    base = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "api_key": "test-key",
    }
    if provider == "openai_compatible":
        base.update({
            "provider": "openai_compatible",
            "base_url": "https://api.test.com/v1",
            "model": "gpt-4o",
        })
    else:
        base["provider"] = "anthropic"
    return base


def _fake_image_b64() -> str:
    """1x1 white PNG as base64."""
    import io
    from PIL import Image
    img = Image.new("RGB", (1, 1), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_vision_chat_anthropic_format(tmp_path):
    """vision_chat should call Anthropic messages.create with image content."""
    config = _make_config("anthropic")

    with patch("agent.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="这是一张XRD图谱")]
        mock_client.messages.create.return_value = mock_response

        client = LLMClient(config)
        result = client.vision_chat(
            prompt="这张图显示了什么？",
            image_b64=_fake_image_b64(),
            media_type="image/png",
            max_tokens=200,
        )

    assert result == "这是一张XRD图谱"
    call_kwargs = mock_client.messages.create.call_args[1]
    # Verify image content block is in messages
    messages = call_kwargs["messages"]
    content = messages[0]["content"]
    assert any(
        isinstance(block, dict) and block.get("type") == "image"
        for block in content
    )


def test_vision_chat_openai_format():
    """vision_chat should call OpenAI with image_url content."""
    config = _make_config("openai_compatible")

    with patch("agent.llm_client.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "检测到Pt纳米颗粒"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        client = LLMClient(config)
        result = client.vision_chat(
            prompt="这张TEM图显示了什么？",
            image_b64=_fake_image_b64(),
            media_type="image/png",
            max_tokens=200,
        )

    assert result == "检测到Pt纳米颗粒"
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    # Should have image_url type in content
    content = messages[0]["content"]
    assert any(
        isinstance(block, dict) and block.get("type") == "image_url"
        for block in content
    )


def test_vision_chat_not_supported_raises():
    """When model doesn't support vision, should raise informative error."""
    config = _make_config("anthropic")

    with patch("agent.llm_client.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = Exception(
            "Input validation error: image not supported"
        )

        client = LLMClient(config)
        with pytest.raises(RuntimeError, match="视觉功能不支持"):
            client.vision_chat(
                prompt="这张图显示了什么？",
                image_b64=_fake_image_b64(),
                media_type="image/png",
            )
```

**Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/agent/test_llm_client.py -v`
Expected: FAIL — `AttributeError: 'LLMClient' object has no attribute 'vision_chat'`

**Step 3: Append vision_chat() to agent/llm_client.py**

在 `simple_chat` 方法之后追加：

```python
    def vision_chat(self, prompt: str, image_b64: str,
                    media_type: str = "image/jpeg",
                    max_tokens: int = 800) -> str:
        """Send an image + text prompt, return text description.

        Args:
            prompt: Question or instruction about the image.
            image_b64: Base64-encoded image bytes.
            media_type: MIME type, e.g. "image/png", "image/jpeg".
            max_tokens: Max tokens in response.

        Returns:
            Text description/analysis from the model.

        Raises:
            RuntimeError: If the model does not support vision.
        """
        try:
            if self.provider == "anthropic":
                return self._vision_anthropic(prompt, image_b64, media_type, max_tokens)
            else:
                return self._vision_openai(prompt, image_b64, media_type, max_tokens)
        except RuntimeError:
            raise
        except Exception as e:
            if "image" in str(e).lower() or "vision" in str(e).lower() or "multimodal" in str(e).lower():
                raise RuntimeError(
                    f"视觉功能不支持：当前模型 ({self.model}) 不支持图像输入。\n"
                    "请在 config.yaml 中切换到支持视觉的模型，例如：\n"
                    "  Anthropic: claude-sonnet-4-20250514\n"
                    "  OpenAI: gpt-4o\n"
                    "  Qwen: qwen-vl-plus\n"
                    "  DeepSeek: deepseek-vl2"
                ) from e
            raise

    def _vision_anthropic(self, prompt: str, image_b64: str,
                          media_type: str, max_tokens: int) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.content[0].text.strip()

    def _vision_openai(self, prompt: str, image_b64: str,
                       media_type: str, max_tokens: int) -> str:
        data_url = f"data:{media_type};base64,{image_b64}"
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return (response.choices[0].message.content or "").strip()
```

**Step 4: Install pillow and run tests**

```bash
./venv/bin/pip install pillow
./venv/bin/python -m pytest tests/agent/test_llm_client.py -v
```
Expected: 3 passed

**Step 5: Add pillow to requirements.txt**

在 requirements.txt 的 `# Ingestion` 部分追加：
```
pillow>=10.0
```

**Step 6: Commit**

```bash
git add agent/llm_client.py tests/agent/test_llm_client.py requirements.txt
git commit -m "feat(agent): add vision_chat() to LLMClient for image understanding"
```

---

### Task 2: multimodal/vision.py — 图像理解核心逻辑

**Files:**
- Create: `multimodal/vision.py`
- Create: `tests/multimodal/__init__.py`
- Create: `tests/multimodal/test_vision.py`

**Step 1: Write the failing test**

```python
# tests/multimodal/test_vision.py
import os
import base64
import pytest
from unittest.mock import MagicMock
from multimodal.vision import (
    image_file_to_b64,
    describe_image,
    analyze_figure_as_scientist,
)


def _make_test_image(tmp_path) -> str:
    """Create a minimal test PNG and return its path."""
    from PIL import Image
    img = Image.new("RGB", (10, 10), color=(200, 200, 200))
    path = str(tmp_path / "test.png")
    img.save(path)
    return path


def test_image_file_to_b64(tmp_path):
    path = _make_test_image(tmp_path)
    b64, media_type = image_file_to_b64(path)
    assert isinstance(b64, str)
    assert len(b64) > 0
    assert media_type == "image/png"
    # Verify it's valid base64
    decoded = base64.b64decode(b64)
    assert len(decoded) > 0


def test_image_file_to_b64_not_found():
    with pytest.raises(FileNotFoundError):
        image_file_to_b64("/nonexistent/image.png")


def test_describe_image(tmp_path):
    path = _make_test_image(tmp_path)
    mock_client = MagicMock()
    mock_client.vision_chat.return_value = "这是一张灰色的测试图片"

    result = describe_image(
        client=mock_client,
        image_path=path,
        prompt="这张图显示了什么？",
    )
    assert result == "这是一张灰色的测试图片"
    mock_client.vision_chat.assert_called_once()


def test_analyze_figure_as_scientist(tmp_path):
    """analyze_figure_as_scientist should inject scientific context."""
    path = _make_test_image(tmp_path)
    mock_client = MagicMock()
    mock_client.vision_chat.return_value = "XRD图谱显示Pt(111)主峰，无杂峰"

    result = analyze_figure_as_scientist(
        client=mock_client,
        image_path=path,
        figure_context="这是一张XRD图谱，来自氢能催化剂表征实验",
        persona_summary="专注氢能催化，重视实验验证，关注稳定性",
    )
    assert "XRD" in result or len(result) > 0
    # Verify scientific context was passed in prompt
    call_args = mock_client.vision_chat.call_args
    prompt_used = call_args[1]["prompt"] if "prompt" in call_args[1] else call_args[0][0]
    assert "催化" in prompt_used or "科研" in prompt_used
```

**Step 2: Run test to verify it fails**

```bash
mkdir -p tests/multimodal && touch tests/multimodal/__init__.py
./venv/bin/python -m pytest tests/multimodal/test_vision.py -v
```
Expected: FAIL

**Step 3: Write implementation**

```python
# multimodal/vision.py
"""Image understanding utilities for the Twin Scientist."""
import base64
import os

# Map file extension to MIME type
_MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

SCIENTIST_VISION_PROMPT = """你是一位材料科学领域的科研人员（专攻{domain}）。
请从科研人员的视角分析这张图像。

图像背景：{context}

请关注：
- 图像类型（XRD/SEM/TEM/极化曲线/EIS等）识别
- 关键数据点和趋势
- 数据质量评估（噪声、异常点等）
- 与你研究方向的关联性
- 值得关注或质疑的地方

用简洁直接的语言回答，像在组会上评论一样。"""


def image_file_to_b64(image_path: str) -> tuple[str, str]:
    """Read an image file and return (base64_string, media_type).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()
    media_type = _MIME_MAP.get(ext)
    if not media_type:
        raise ValueError(
            f"Unsupported image format: {ext}. "
            f"Supported: {list(_MIME_MAP.keys())}"
        )

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    return b64, media_type


def describe_image(client, image_path: str, prompt: str,
                   max_tokens: int = 800) -> str:
    """Describe an image using the LLM vision capability."""
    b64, media_type = image_file_to_b64(image_path)
    return client.vision_chat(
        prompt=prompt,
        image_b64=b64,
        media_type=media_type,
        max_tokens=max_tokens,
    )


def analyze_figure_as_scientist(client, image_path: str,
                                 figure_context: str = "",
                                 persona_summary: str = "",
                                 domain: str = "氢能催化",
                                 max_tokens: int = 600) -> str:
    """Analyze a figure with the researcher's scientific lens."""
    prompt = SCIENTIST_VISION_PROMPT.format(
        domain=domain,
        context=figure_context or "请描述这张图的内容和科学意义",
    )
    if persona_summary:
        prompt += f"\n\n你的研究背景：{persona_summary}"

    return describe_image(client, image_path, prompt, max_tokens)
```

**Step 4: Run tests**

```bash
./venv/bin/python -m pytest tests/multimodal/test_vision.py -v
```
Expected: 4 passed

**Step 5: Commit**

```bash
git add multimodal/vision.py tests/multimodal/__init__.py tests/multimodal/test_vision.py
git commit -m "feat(multimodal): add vision utilities for scientific figure analysis"
```

---

### Task 3: Agent 集成 see 工具

**Files:**
- Modify: `agent/main.py`（新增 see 工具定义和执行）
- Modify: `tests/agent/test_main.py`（追加测试）

**Step 1: 追加测试到 tests/agent/test_main.py**

```python
# 追加到 tests/agent/test_main.py
def test_see_tool_executes(tmp_path):
    """see tool should call vision_chat and return description."""
    from PIL import Image
    project_dir = _make_project(tmp_path)
    agent = TwinScientist(project_dir)

    # Create a test image
    img_path = str(tmp_path / "test_figure.png")
    Image.new("RGB", (10, 10), color=(100, 100, 100)).save(img_path)

    agent.client.vision_chat = MagicMock(return_value="这是一张XRD图谱，主峰对应Pt(111)")

    result = agent._execute_tool("see", {
        "image_path": img_path,
        "context": "来自氢能催化剂表征实验的XRD图",
    })

    assert "XRD" in result
    agent.client.vision_chat.assert_called_once()


def test_see_tool_file_not_found(tmp_path):
    """see tool should return error message when image not found."""
    project_dir = _make_project(tmp_path)
    agent = TwinScientist(project_dir)

    result = agent._execute_tool("see", {
        "image_path": "/nonexistent/figure.png",
        "context": "测试",
    })

    assert "找不到" in result or "not found" in result.lower() or "错误" in result
```

**Step 2: Run tests to verify they fail**

```bash
./venv/bin/python -m pytest tests/agent/test_main.py::test_see_tool_executes tests/agent/test_main.py::test_see_tool_file_not_found -v
```
Expected: FAIL

**Step 3: Modify agent/main.py**

在 `TOOL_DEFINITIONS` 末尾追加 `see` 工具定义（在最后的 `}` 和 `]` 之间）：

```python
    {
        "name": "see",
        "description": "观察并分析一张图片（论文图表、实验截图、PPT页面）。当用户分享图片或提到图表时调用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "图片文件的路径",
                },
                "context": {
                    "type": "string",
                    "description": "图片的背景信息，如来源、实验类型等",
                },
            },
            "required": ["image_path"],
        },
    },
```

在 `_execute_tool` 方法中，在最后的 `return f"Unknown tool: {tool_name}"` 之前追加：

```python
        elif tool_name == "see":
            return self._execute_see(tool_input)
```

在 `TwinScientist` 类末尾追加 `_execute_see` 方法：

```python
    def _execute_see(self, tool_input: dict) -> str:
        """Execute the see tool: analyze an image with scientific lens."""
        from multimodal.vision import analyze_figure_as_scientist
        import yaml as _yaml

        image_path = tool_input.get("image_path", "")
        context = tool_input.get("context", "")

        if not os.path.exists(image_path):
            return f"错误：找不到图片文件 {image_path}"

        # Build persona summary from identity.yaml
        try:
            with open(os.path.join(self.persona_dir, "identity.yaml"), "r") as f:
                identity = _yaml.safe_load(f) or {}
            research_focus = identity.get("research_focus", [])
            persona_summary = "、".join(research_focus[:3]) if research_focus else ""
        except Exception:
            persona_summary = ""

        try:
            return analyze_figure_as_scientist(
                client=self.client,
                image_path=image_path,
                figure_context=context,
                persona_summary=persona_summary,
            )
        except RuntimeError as e:
            return str(e)
```

**Step 4: Run all main tests**

```bash
./venv/bin/python -m pytest tests/agent/test_main.py -v
```
Expected: 9 passed

**Step 5: Commit**

```bash
git add agent/main.py tests/agent/test_main.py
git commit -m "feat(agent): add see tool for scientific image analysis"
```

---

### Task 4: paper_processor 增加图表印象提取

**Files:**
- Modify: `ingestion/paper_processor.py`（新增 extract_figure_impressions）
- Modify: `tests/ingestion/test_paper_processor.py`（追加测试）

**Step 1: 追加测试**

```python
# 追加到 tests/ingestion/test_paper_processor.py
from ingestion.paper_processor import extract_figure_impressions

def test_extract_figure_impressions_from_image(tmp_path):
    """extract_figure_impressions should return list of figure analyses."""
    from PIL import Image
    img_path = str(tmp_path / "fig1.png")
    Image.new("RGB", (50, 50), color=(180, 180, 180)).save(img_path)

    mock_client = MagicMock()
    mock_client.vision_chat.return_value = """figure: "Fig.1 XRD patterns"
saw: "主峰对应Pt(111)，2θ约39.8度，无杂峰"
judgment: "晶相纯净，数据可信"
"""

    result = extract_figure_impressions(
        client=mock_client,
        image_paths=[img_path],
        paper_title="Test Paper",
    )

    assert isinstance(result, list)
    assert len(result) == 1
    assert "saw" in result[0] or len(result[0]) > 0


def test_extract_figure_impressions_empty():
    """Empty image list should return empty list."""
    mock_client = MagicMock()
    result = extract_figure_impressions(
        client=mock_client,
        image_paths=[],
        paper_title="Test",
    )
    assert result == []
```

**Step 2: Run tests to verify they fail**

```bash
./venv/bin/python -m pytest tests/ingestion/test_paper_processor.py::test_extract_figure_impressions_from_image tests/ingestion/test_paper_processor.py::test_extract_figure_impressions_empty -v
```
Expected: FAIL

**Step 3: Append to ingestion/paper_processor.py**

在文件末尾追加：

```python
FIGURE_PROMPT = """你正在分析一篇材料科学论文（标题：{title}）中的图表。
请从科研人员视角简洁描述这张图，输出YAML格式（只输出YAML，不要其他文字）：

figure: "图的标题或编号（如果能识别）"
saw: "你看到了什么（具体数据、趋势、特征）"
judgment: "这个数据的质量和可信度评估"
"""


def extract_figure_impressions(client, image_paths: list[str],
                                paper_title: str = "") -> list[dict]:
    """Analyze figure images and return impression dicts."""
    if not image_paths:
        return []

    import re
    results = []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            continue
        try:
            from multimodal.vision import image_file_to_b64
            b64, media_type = image_file_to_b64(img_path)
            prompt = FIGURE_PROMPT.format(title=paper_title or "未知")
            raw = client.vision_chat(
                prompt=prompt,
                image_b64=b64,
                media_type=media_type,
                max_tokens=300,
            )
            raw = re.sub(r"^```(?:yaml)?\n?", "", raw.strip())
            raw = re.sub(r"\n?```$", "", raw)
            try:
                impression = yaml.safe_load(raw) or {"raw": raw}
            except yaml.YAMLError:
                impression = {"raw": raw}
            results.append(impression)
        except Exception as e:
            results.append({"error": str(e), "path": img_path})

    return results
```

**Step 4: Run all paper_processor tests**

```bash
./venv/bin/python -m pytest tests/ingestion/test_paper_processor.py -v
```
Expected: 6 passed

**Step 5: Commit**

```bash
git add ingestion/paper_processor.py tests/ingestion/test_paper_processor.py
git commit -m "feat(ingestion): add figure impression extraction from images"
```

---

### Task 5: 全量回归测试 + push

**Step 1: Run all tests**

```bash
./venv/bin/python -m pytest tests/ -v
```
Expected: All passed (70+ tests)

**Step 2: Push**

```bash
git push origin main
```
