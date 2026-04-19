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