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