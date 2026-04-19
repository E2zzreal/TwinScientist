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