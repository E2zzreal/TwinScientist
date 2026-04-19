#!/usr/bin/env python3
"""
会议字幕清洗脚本
================
功能：
  1. 解析腾讯会议、飞书、Zoom 导出的字幕文件
  2. 合并被错误分断的句子
  3. 过滤语气词、重复词（"那个那个那个"→"那个"）
  4. 自动识别并统一说话人名称（处理"主持人"/"与会者"等模糊标注）
  5. 输出标准格式（适配 audio_processor.py）

支持的输入格式：
  - 腾讯会议导出 .txt（格式: "发言人名\n时间: 内容"）
  - 飞书妙记 .txt（格式: "HH:MM 发言人: 内容"）
  - 标准 SRT 字幕 .srt
  - 通用格式（自动检测）

用法：
  python preprocess/transcript_clean.py --input 会议字幕.txt --speaker 张三
  python preprocess/transcript_clean.py --input subtitles.srt --format srt
  python preprocess/transcript_clean.py --input transcripts/ --batch
"""

import argparse
import os
import re
import sys
from pathlib import Path


# ─── 常见语气词/填充词（清洗掉但保留语义）───────────────────────────────────
FILLER_PATTERNS = [
    r"\b(那个|这个|嗯|呃|啊|哦|对对对|好好好|是是是)\s*\1{1,4}\b",  # 重复语气词
    r"^(嗯+|呃+|啊+)[，。\s]*$",     # 纯语气词段落
    r"\[.*?\]",                       # [笑声] [掌声] 等标注
]

# 腾讯会议格式识别
TENCENT_PATTERN = re.compile(
    r"^(.+?)\s*\n\s*(\d{2}:\d{2}:\d{2})\s*(.+?)$",
    re.MULTILINE,
)

# 飞书妙记格式
FEISHU_PATTERN = re.compile(
    r"^(\d{1,2}:\d{2}(?::\d{2})?)\s+([^\n:]+?)：\s*(.+?)$",
    re.MULTILINE,
)

# 标准 "说话人: 内容" 格式
GENERIC_PATTERN = re.compile(
    r"^([^\n:：]{1,20})[：:]\s*(.+?)$",
    re.MULTILINE,
)


def detect_format(content: str) -> str:
    """Auto-detect transcript format."""
    if content.startswith("WEBVTT") or re.search(r"\d{2}:\d{2}:\d{2},\d{3}", content):
        return "srt"
    if TENCENT_PATTERN.search(content[:2000]):
        return "tencent"
    if FEISHU_PATTERN.search(content[:2000]):
        return "feishu"
    return "generic"


def parse_srt(content: str) -> list[dict]:
    """Parse SRT subtitle format."""
    segments = []
    blocks = re.split(r"\n\n+", content.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        # Try to extract time and text
        time_line = lines[1] if len(lines) > 1 else ""
        time_match = re.search(r"(\d{2}:\d{2}:\d{2})", time_line)
        timestamp = time_match.group(1) if time_match else ""
        text = " ".join(lines[2:]).strip()
        if text:
            segments.append({"speaker": "???", "timestamp": timestamp, "text": text})
    return segments


def parse_tencent(content: str) -> list[dict]:
    """
    Parse Tencent Meeting transcript format.
    Example:
        张三（主持人）
        00:01:23 这是发言内容
    """
    segments = []
    # Split by speaker blocks
    blocks = re.split(r"\n(?=[^\d\s])", content.strip())
    for block in blocks:
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
        if not lines:
            continue
        speaker = lines[0]
        # Remove role labels: （主持人）, (与会者) etc.
        speaker = re.sub(r"[（(][^）)]*[）)]", "", speaker).strip()

        for line in lines[1:]:
            time_match = re.match(r"(\d{2}:\d{2}:\d{2})\s*(.*)", line)
            if time_match:
                timestamp, text = time_match.groups()
                if text.strip():
                    segments.append({
                        "speaker": speaker,
                        "timestamp": timestamp,
                        "text": text.strip(),
                    })
    return segments


def parse_feishu(content: str) -> list[dict]:
    """
    Parse Feishu (Lark) meeting minutes format.
    Example:
        00:01 张三：这是发言内容
    """
    segments = []
    for match in FEISHU_PATTERN.finditer(content):
        timestamp, speaker, text = match.groups()
        segments.append({
            "speaker": speaker.strip(),
            "timestamp": timestamp,
            "text": text.strip(),
        })
    return segments


def parse_generic(content: str) -> list[dict]:
    """Parse generic 'Speaker: text' format (one per line)."""
    segments = []
    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Try "Speaker: text" pattern
        match = re.match(r"^([^\n:：]{1,20})[：:]\s*(.+)$", line)
        if match:
            speaker, text = match.groups()
            segments.append({
                "speaker": speaker.strip(),
                "timestamp": "",
                "text": text.strip(),
            })
        else:
            # No speaker label — try to continue previous
            if segments:
                segments[-1]["text"] += " " + line
    return segments


def clean_text(text: str) -> str:
    """Clean individual utterance text."""
    # Remove filler repetitions
    for pattern in FILLER_PATTERNS:
        text = re.sub(pattern, "", text)

    # Clean extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove standalone punctuation
    text = re.sub(r"^[，。！？、…\s]+$", "", text)

    return text


def merge_short_segments(segments: list[dict],
                          min_chars: int = 10,
                          same_speaker_only: bool = True) -> list[dict]:
    """
    Merge segments that are too short into the previous one.
    Auto-subtitles often split sentences mid-way.
    """
    if not segments:
        return segments

    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        # Merge if: same speaker + previous segment too short + no sentence boundary
        prev_text = prev["text"]
        same_speaker = (prev["speaker"] == seg["speaker"]) or not same_speaker_only
        prev_short = len(prev_text) < min_chars
        no_boundary = not re.search(r"[。！？…]\s*$", prev_text)

        if same_speaker and (prev_short or no_boundary):
            merged[-1]["text"] = prev_text + seg["text"]
        else:
            merged.append(seg.copy())

    return merged


def normalize_speakers(segments: list[dict],
                        speaker_map: dict[str, str] | None = None) -> list[dict]:
    """
    Normalize speaker names.
    speaker_map example: {"主持人": "张三", "与会者1": "李四"}
    """
    if not speaker_map:
        return segments

    for seg in segments:
        seg["speaker"] = speaker_map.get(seg["speaker"], seg["speaker"])
    return segments


def filter_low_value_segments(segments: list[dict],
                               min_chars: int = 5) -> list[dict]:
    """Remove segments that are too short or contain only filler words."""
    FILLERS = {"好的", "嗯", "对", "好", "谢谢", "是的", "对对", "可以", "明白"}
    result = []
    for seg in segments:
        text = seg["text"].strip()
        if len(text) < min_chars:
            continue
        if text in FILLERS:
            continue
        result.append(seg)
    return result


def to_standard_format(segments: list[dict], speaker_focus: str = "") -> str:
    """
    Convert to standard format for audio_processor.py:
        [HH:MM] Speaker: Text

    Args:
        speaker_focus: If set, mark other speakers differently for clarity
    """
    lines = []
    for seg in segments:
        ts = seg.get("timestamp", "")
        ts_str = f"[{ts}] " if ts else ""
        speaker = seg["speaker"]
        text = seg["text"]

        # Highlight the target speaker
        if speaker_focus and speaker == speaker_focus:
            lines.append(f"{ts_str}{speaker}：{text}")
        else:
            lines.append(f"{ts_str}{speaker}：{text}")

    return "\n".join(lines)


def get_speaker_stats(segments: list[dict]) -> dict:
    """Compute speaking time/turns per speaker."""
    stats: dict[str, dict] = {}
    for seg in segments:
        sp = seg["speaker"]
        if sp not in stats:
            stats[sp] = {"turns": 0, "chars": 0}
        stats[sp]["turns"] += 1
        stats[sp]["chars"] += len(seg["text"])

    # Sort by turns
    return dict(sorted(stats.items(), key=lambda x: -x[1]["turns"]))


def process_transcript(input_path: str,
                        output_path: str | None = None,
                        fmt: str = "auto",
                        speaker_focus: str = "",
                        speaker_map: dict | None = None,
                        merge: bool = True) -> str:
    """
    Full pipeline: parse → clean → merge → normalize → save.

    Returns cleaned transcript as string.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Detect format
    if fmt == "auto":
        fmt = detect_format(content)
    print(f"  检测到格式: {fmt}")

    # Parse
    parsers = {
        "srt": parse_srt,
        "tencent": parse_tencent,
        "feishu": parse_feishu,
        "generic": parse_generic,
    }
    parser = parsers.get(fmt, parse_generic)
    segments = parser(content)
    print(f"  解析出 {len(segments)} 个发言片段")

    # Clean text
    for seg in segments:
        seg["text"] = clean_text(seg["text"])

    # Filter empty after cleaning
    segments = [s for s in segments if s["text"].strip()]

    # Merge short segments
    if merge:
        segments = merge_short_segments(segments)
        print(f"  合并后: {len(segments)} 个片段")

    # Normalize speakers
    if speaker_map:
        segments = normalize_speakers(segments, speaker_map)

    # Filter low-value
    before = len(segments)
    segments = filter_low_value_segments(segments)
    filtered = before - len(segments)
    if filtered:
        print(f"  过滤低价值片段: {filtered} 个")

    # Speaker stats
    stats = get_speaker_stats(segments)
    print("\n  说话人统计:")
    for sp, data in stats.items():
        marker = " ← 目标" if sp == speaker_focus else ""
        print(f"    {sp}: {data['turns']} 次发言, {data['chars']} 字{marker}")

    # Generate output
    cleaned = to_standard_format(segments, speaker_focus)

    # Save
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_cleaned.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# 清洗后的会议记录\n")
        f.write(f"# 来源: {os.path.basename(input_path)}\n")
        f.write(f"# 目标发言人: {speaker_focus}\n\n")
        f.write(cleaned)

    print(f"\n  ✅ 已保存到: {output_path}")
    return cleaned


def batch_process(input_dir: str, output_dir: str,
                  speaker_focus: str = "",
                  speaker_map: dict | None = None):
    """Process all transcript files in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    supported = {".txt", ".srt", ".vtt"}
    files = [
        f for f in Path(input_dir).iterdir()
        if f.suffix.lower() in supported
        and "_cleaned" not in f.stem
    ]

    if not files:
        print(f"在 {input_dir} 中没有找到字幕/转写文件")
        return

    print(f"找到 {len(files)} 个文件\n")
    for f in sorted(files):
        output_path = os.path.join(output_dir, f"{f.stem}_cleaned.txt")
        print(f"\n{'='*40}")
        print(f"处理: {f.name}")
        print('='*40)
        try:
            process_transcript(
                str(f), output_path, "auto",
                speaker_focus, speaker_map,
            )
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="会议字幕清洗工具")
    parser.add_argument("--input", required=True,
                        help="输入文件或目录")
    parser.add_argument("--output", default=None,
                        help="输出文件路径（默认: 原文件名_cleaned.txt）")
    parser.add_argument("--output_dir", default="data/transcripts_cleaned",
                        help="批量模式输出目录")
    parser.add_argument("--speaker", default="",
                        help="目标说话人姓名（用于标注）")
    parser.add_argument("--format", default="auto",
                        choices=["auto", "tencent", "feishu", "srt", "generic"],
                        help="字幕格式（默认自动检测）")
    parser.add_argument("--batch", action="store_true",
                        help="批量处理整个目录")
    parser.add_argument("--speaker_map", default=None,
                        help='说话人名称映射 JSON，如 {"主持人":"张三","与会者1":"李四"}')
    args = parser.parse_args()

    speaker_map = None
    if args.speaker_map:
        import json
        speaker_map = json.loads(args.speaker_map)

    if args.batch or os.path.isdir(args.input):
        batch_process(args.input, args.output_dir, args.speaker, speaker_map)
    else:
        process_transcript(
            args.input, args.output,
            args.format, args.speaker, speaker_map,
        )


if __name__ == "__main__":
    main()
