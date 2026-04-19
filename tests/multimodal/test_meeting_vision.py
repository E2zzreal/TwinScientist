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