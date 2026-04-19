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