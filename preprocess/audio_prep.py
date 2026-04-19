#!/usr/bin/env python3
"""
录音预处理脚本
==============
功能：
  1. 格式转换（mp4/m4a/ogg → wav/mp3）
  2. 静音检测与过滤（去除纯静音片段）
  3. 音频分割（将长录音按静音断点分割成块）
  4. 质量评分（信噪比估算，过滤低质量片段）
  5. 说话人标注辅助（输出带时间戳的转写，方便手动标注）

依赖：
  pip install pydub numpy scipy
  # pydub 需要 ffmpeg：sudo apt-get install ffmpeg（Linux）
  #                     brew install ffmpeg（Mac）

用法：
  python preprocess/audio_prep.py --input 会议录音.mp4 --output_dir data/audio/
  python preprocess/audio_prep.py --input recordings/ --output_dir data/audio/ --batch
"""

import argparse
import json
import os
import sys


def check_ffmpeg():
    """Check if ffmpeg is available."""
    import shutil
    if shutil.which("ffmpeg") is None:
        print("❌ 未找到 ffmpeg。请先安装：")
        print("   Linux: sudo apt-get install ffmpeg")
        print("   Mac:   brew install ffmpeg")
        sys.exit(1)


def convert_to_wav(input_path: str, output_path: str) -> str:
    """Convert audio to 16kHz mono WAV (optimal for Whisper)."""
    try:
        from pydub import AudioSegment
    except ImportError:
        print("请先安装: pip install pydub")
        sys.exit(1)

    print(f"  转换格式: {os.path.basename(input_path)} → WAV")
    audio = AudioSegment.from_file(input_path)
    # 16kHz mono — Whisper 最佳格式
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_path, format="wav")
    duration_min = len(audio) / 60000
    print(f"  时长: {duration_min:.1f} 分钟")
    return output_path


def estimate_snr(audio_segment) -> float:
    """Estimate Signal-to-Noise Ratio (higher is better, >20dB is good)."""
    try:
        import numpy as np
        samples = np.array(audio_segment.get_array_of_samples(), dtype=float)
        if len(samples) == 0:
            return 0.0
        rms = np.sqrt(np.mean(samples ** 2))
        # Estimate noise from the quietest 10% of the signal
        sorted_samples = np.sort(np.abs(samples))
        noise_samples = sorted_samples[:max(1, len(sorted_samples) // 10)]
        noise_rms = np.sqrt(np.mean(noise_samples ** 2))
        if noise_rms < 1e-6:
            return 60.0  # Very clean signal
        snr = 20 * np.log10(rms / noise_rms)
        return round(snr, 1)
    except ImportError:
        return -1.0  # numpy not available


def split_on_silence(input_wav: str, output_dir: str,
                     min_silence_ms: int = 1500,
                     silence_thresh_db: int = -40,
                     min_chunk_min: float = 1.0,
                     max_chunk_min: float = 15.0) -> list[str]:
    """
    Split audio on silence into chunks suitable for Whisper.

    Args:
        min_silence_ms:    Minimum silence duration to split on (ms)
        silence_thresh_db: Audio below this level counts as silence (dBFS)
        min_chunk_min:     Minimum chunk duration (minutes)
        max_chunk_min:     Maximum chunk duration (minutes)

    Returns:
        List of output file paths.
    """
    try:
        from pydub import AudioSegment
        from pydub.silence import split_on_silence as _split
    except ImportError:
        print("请先安装: pip install pydub")
        sys.exit(1)

    print(f"  分割音频（静音阈值: {silence_thresh_db}dBFS）...")
    audio = AudioSegment.from_wav(input_wav)
    total_min = len(audio) / 60000

    if total_min <= max_chunk_min:
        print(f"  时长 {total_min:.1f} 分钟，无需分割")
        return [input_wav]

    chunks = _split(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=silence_thresh_db,
        keep_silence=200,
    )

    min_ms = int(min_chunk_min * 60 * 1000)
    max_ms = int(max_chunk_min * 60 * 1000)

    # Merge too-short chunks, split too-long ones
    merged = []
    current = AudioSegment.empty()
    for chunk in chunks:
        current += chunk
        if len(current) >= min_ms:
            if len(current) > max_ms:
                # Force split at max_ms boundaries
                while len(current) > max_ms:
                    merged.append(current[:max_ms])
                    current = current[max_ms:]
            merged.append(current)
            current = AudioSegment.empty()
    if len(current) > 0:
        merged.append(current)

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_wav))[0]
    paths = []
    for i, chunk in enumerate(merged):
        path = os.path.join(output_dir, f"{base}_chunk{i+1:03d}.wav")
        chunk.export(path, format="wav")
        duration = len(chunk) / 60000
        snr = estimate_snr(chunk)
        print(f"    块 {i+1}: {duration:.1f}分钟, 信噪比: {snr:.0f}dB")
        paths.append(path)

    print(f"  共分割为 {len(paths)} 块")
    return paths


def quality_report(wav_path: str) -> dict:
    """Generate quality report for an audio file."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(wav_path)
        snr = estimate_snr(audio)
        duration_min = len(audio) / 60000

        # Estimate speech ratio (non-silent portions)
        from pydub.silence import detect_nonsilent
        non_silent = detect_nonsilent(audio, min_silence_len=300,
                                      silence_thresh=-40)
        if len(audio) > 0:
            speech_duration = sum(end - start for start, end in non_silent)
            speech_ratio = speech_duration / len(audio)
        else:
            speech_ratio = 0.0

        quality = "好" if snr > 20 and speech_ratio > 0.4 else \
                  "中" if snr > 10 and speech_ratio > 0.2 else "差"

        return {
            "file": os.path.basename(wav_path),
            "duration_min": round(duration_min, 1),
            "snr_db": snr,
            "speech_ratio": round(speech_ratio, 2),
            "quality": quality,
            "recommendation": "可用于人格提取" if quality in ("好", "中")
                              else "建议跳过（噪音太大）",
        }
    except Exception as e:
        return {"file": os.path.basename(wav_path), "error": str(e)}


def transcribe_with_timestamps(wav_path: str, model_size: str = "base",
                                language: str = "zh") -> list[dict]:
    """
    Transcribe audio with word-level timestamps using Whisper.
    Helps with manual speaker labeling.

    Returns list of segments: {start, end, text}
    """
    try:
        import whisper
    except ImportError:
        print("请先安装: pip install openai-whisper")
        return []

    print(f"  转写中（模型: {model_size}）...")
    model = whisper.load_model(model_size)
    result = model.transcribe(
        wav_path,
        language=language,
        word_timestamps=True,
        verbose=False,
    )

    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": round(seg["start"], 1),
            "end": round(seg["end"], 1),
            "text": seg["text"].strip(),
        })
    return segments


def save_transcript_for_labeling(segments: list[dict],
                                  output_path: str,
                                  speaker_name: str = "张三"):
    """
    Save transcript in a format easy to manually add speaker labels.

    Format:
        [00:12] 张三: 这是转写的文字内容
        [00:45] ???: 另一段发言（需要手动填写说话人）
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 说话人标注文件\n")
        f.write("# 请将 '???' 替换为实际说话人姓名\n")
        f.write("# 格式：[时间] 姓名: 内容\n\n")
        for seg in segments:
            m, s = divmod(int(seg["start"]), 60)
            timestamp = f"{m:02d}:{s:02d}"
            # Default first speaker to twin_name, others to ???
            speaker = speaker_name if seg["text"] else "???"
            f.write(f"[{timestamp}] {speaker}: {seg['text']}\n")

    print(f"  转写已保存到 {output_path}")
    print(f"  请手动标注说话人后，再运行摄入流程")


def process_single(input_path: str, output_dir: str,
                   speaker_name: str = "张三",
                   skip_transcribe: bool = False) -> dict:
    """Process a single audio file through the full pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]

    print(f"\n{'='*50}")
    print(f"处理: {os.path.basename(input_path)}")
    print('='*50)

    # Step 1: Convert to WAV
    wav_path = os.path.join(output_dir, f"{base}.wav")
    if not input_path.endswith(".wav"):
        convert_to_wav(input_path, wav_path)
    else:
        import shutil
        shutil.copy(input_path, wav_path)

    # Step 2: Quality report
    print("\n质量评估:")
    report = quality_report(wav_path)
    for k, v in report.items():
        print(f"  {k}: {v}")

    if report.get("quality") == "差":
        print("  ⚠️  质量较差，建议检查录音设备后重新录制")

    # Step 3: Split into chunks
    chunks_dir = os.path.join(output_dir, f"{base}_chunks")
    chunk_paths = split_on_silence(wav_path, chunks_dir)

    # Step 4: Transcribe with timestamps (optional)
    transcript_path = None
    if not skip_transcribe:
        print("\n生成标注用转写文件:")
        segments = transcribe_with_timestamps(wav_path)
        if segments:
            transcript_path = os.path.join(
                output_dir, f"{base}_for_labeling.txt"
            )
            save_transcript_for_labeling(segments, transcript_path, speaker_name)

    return {
        "original": input_path,
        "wav": wav_path,
        "chunks": chunk_paths,
        "transcript_for_labeling": transcript_path,
        "quality": report,
    }


def batch_process(input_dir: str, output_dir: str,
                  speaker_name: str = "张三") -> list[dict]:
    """Process all audio files in a directory."""
    supported = {".mp3", ".mp4", ".m4a", ".wav", ".ogg", ".flac", ".aac"}
    files = [
        os.path.join(input_dir, f)
        for f in sorted(os.listdir(input_dir))
        if os.path.splitext(f)[1].lower() in supported
    ]

    if not files:
        print(f"在 {input_dir} 中没有找到音频文件")
        return []

    print(f"找到 {len(files)} 个音频文件")
    results = []
    for f in files:
        result = process_single(f, output_dir, speaker_name,
                                skip_transcribe=len(files) > 10)
        results.append(result)

    # Summary report
    report_path = os.path.join(output_dir, "preprocessing_report.json")
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\n✅ 预处理完成。报告: {report_path}")

    good = sum(1 for r in results if r["quality"].get("quality") in ("好", "中"))
    print(f"   可用文件: {good}/{len(results)}")
    return results


def main():
    parser = argparse.ArgumentParser(description="录音预处理工具")
    parser.add_argument("--input", required=True,
                        help="输入音频文件或目录")
    parser.add_argument("--output_dir", default="data/audio_processed",
                        help="输出目录（默认: data/audio_processed）")
    parser.add_argument("--speaker", default="张三",
                        help="主要说话人姓名（用于转写标注）")
    parser.add_argument("--batch", action="store_true",
                        help="批量处理整个目录")
    parser.add_argument("--skip_transcribe", action="store_true",
                        help="跳过转写步骤（只做格式转换和分割）")
    args = parser.parse_args()

    check_ffmpeg()

    if args.batch or os.path.isdir(args.input):
        batch_process(args.input, args.output_dir, args.speaker)
    else:
        process_single(args.input, args.output_dir, args.speaker,
                       args.skip_transcribe)


if __name__ == "__main__":
    main()
