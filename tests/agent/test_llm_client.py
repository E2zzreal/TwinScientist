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