# interface/voice.py
"""Voice interface: STT → Agent → TTS pipeline."""
import os
import tempfile
from multimodal.stt import transcribe_file, STTEngine
from multimodal.tts import synthesize_speech, TTSEngine, play_audio, DEFAULT_VOICE


class VoiceSession:
    """A voice conversation session with the digital twin."""

    def __init__(self, agent, voice: str = DEFAULT_VOICE,
                 stt_model: str = "base", language: str = "zh",
                 auto_play: bool = True):
        self.agent = agent
        self.voice = voice
        self.stt_model = stt_model
        self.language = language
        self.auto_play = auto_play
        self._stt = STTEngine(model_size=stt_model, language=language)
        self._tts = TTSEngine(voice=voice)

    def chat_once(self, duration: float = 5.0,
                  output_dir: str | None = None) -> dict:
        """Record one utterance, get response, play it back.

        Returns dict with keys: transcription, response, audio_path
        """
        # 1. Record
        audio_path = self._stt.record_and_transcribe.__wrapped__(duration) \
            if hasattr(self._stt.record_and_transcribe, "__wrapped__") \
            else None

        from multimodal.stt import record_once
        tmp_audio = record_once(duration=duration)
        try:
            transcription = self._stt.transcribe(tmp_audio)
        finally:
            if os.path.exists(tmp_audio):
                os.unlink(tmp_audio)

        print(f"  你说：{transcription}")

        # 2. Agent response
        response = self.agent.chat(transcription)
        print(f"  分身：{response[:100]}{'...' if len(response) > 100 else ''}")

        # 3. TTS
        out_dir = output_dir or tempfile.gettempdir()
        audio_out = os.path.join(out_dir, "twin_response.mp3")
        self._tts.synthesize(response, audio_out)

        # 4. Playback
        if self.auto_play:
            play_audio(audio_out)

        return {
            "transcription": transcription,
            "response": response,
            "audio_path": audio_out,
        }

    def respond_text(self, text: str, output_path: str | None = None) -> str:
        """Synthesize a text response to audio. Returns audio file path."""
        if output_path is None:
            output_path = os.path.join(
                tempfile.gettempdir(), "twin_response.mp3"
            )
        synthesize_speech(text, output_path, voice=self.voice)
        if self.auto_play:
            play_audio(output_path)
        return output_path


def process_voice_file(agent, audio_input_path: str,
                       audio_output_path: str,
                       voice: str = DEFAULT_VOICE,
                       stt_model: str = "base") -> str:
    """Process one audio file: STT → Agent → TTS.

    Returns path to output audio file.
    """
    # Transcribe
    transcription = transcribe_file(audio_input_path, model_size=stt_model)
    print(f"转写：{transcription}")

    # Agent
    response = agent.chat(transcription)
    print(f"回答：{response}")

    # TTS
    synthesize_speech(response, audio_output_path, voice=voice)
    return audio_output_path


def run_voice_cli(project_dir: str, voice: str = DEFAULT_VOICE,
                  duration: float = 5.0):
    """Launch interactive voice conversation loop."""
    from rich.console import Console
    from rich.panel import Panel
    from agent.main import TwinScientist

    console = Console()
    agent = TwinScientist(project_dir)
    session = VoiceSession(agent=agent, voice=voice)

    console.print(Panel(
        f"[bold]Twin Scientist — 语音对话模式[/bold]\n"
        f"声音：{voice}\n"
        "每次按 Enter 开始录音，录音时长 {duration:.0f} 秒\n"
        "输入 q 退出",
        title="语音模式",
    ))

    while True:
        try:
            cmd = input("\n按 Enter 开始说话（q 退出）: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if cmd.lower() == "q":
            break

        try:
            result = session.chat_once(duration=duration)
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")

    agent.end_session()
    console.print("[dim]再见！[/dim]")