# interface/meeting.py
"""Meeting participation: real-time or file-based meeting bot."""
import os
import tempfile
from interface.meeting_context import MeetingContext
from interface.trigger_detector import TriggerDetector
from multimodal.tts import synthesize_speech, play_audio, DEFAULT_VOICE


class MeetingBot:
    """Digital twin meeting participant."""

    def __init__(self, agent,
                 twin_name: str,
                 confident_domains: list[str],
                 voice: str = DEFAULT_VOICE,
                 auto_play: bool = True,
                 screen_interval: float = 30.0):
        self.agent = agent
        self.twin_name = twin_name
        self.confident_domains = confident_domains
        self.voice = voice
        self.auto_play = auto_play

        # Core components
        self.meeting_context = MeetingContext(
            compressor_fn=self._compress_context,
        )
        self.trigger_detector = TriggerDetector(
            twin_name=twin_name,
            confident_domains=confident_domains,
        )
        self._response_history: list[dict] = []

    def process_utterance(self, speaker: str, text: str) -> dict | None:
        """Process one utterance. Returns response dict if twin should speak."""
        # Always record to context
        if speaker != self.twin_name:
            self.meeting_context.add_utterance(speaker, text)

        # Check trigger
        trigger = self.trigger_detector.check(text)
        if not trigger.should_speak:
            return None

        # Build context-aware prompt
        meeting_ctx = self.meeting_context.format_for_agent()
        prompt = (
            f"{meeting_ctx}\n\n"
            f"**{speaker}刚才说：** {text}\n\n"
            f"请作为你自己（{self.twin_name}）在会议中自然地回应。"
            f"保持简洁，说1-3句话，适合口语表达。"
        )

        response = self.agent.chat(prompt)

        # Record own response
        self.meeting_context.add_utterance(self.twin_name, response)

        # Speak the response
        audio_path = self._speak(response)

        result = {
            "trigger": trigger.trigger_type.value,
            "urgency": trigger.urgency,
            "speaker": speaker,
            "input": text,
            "response": response,
            "audio_path": audio_path,
        }
        self._response_history.append(result)
        return result

    def update_screen(self, image_path: str):
        """Update meeting context with current screen content."""
        try:
            from multimodal.meeting_vision import analyze_screen_content
            description = analyze_screen_content(
                client=self.agent.client,
                image_path=image_path,
                meeting_topic=self.meeting_context.topic,
            )
            self.meeting_context.add_utterance(
                "[屏幕共享]", description
            )
        except Exception:
            pass  # Screen analysis is best-effort

    def _speak(self, text: str) -> str:
        """Synthesize and optionally play response."""
        audio_path = os.path.join(
            tempfile.gettempdir(), "twin_meeting_response.mp3"
        )
        synthesize_speech(text, audio_path, voice=self.voice)
        if self.auto_play:
            play_audio(audio_path)
        return audio_path

    def _compress_context(self, utterances: list[dict]) -> str:
        """Use agent LLM to compress old utterances."""
        text = "\n".join(f"{u['speaker']}: {u['text']}" for u in utterances)
        prompt = f"请将以下会议发言压缩成简洁摘要（不超过200字）：\n{text}\n摘要："
        return self.agent.client.simple_chat(prompt, max_tokens=250)


def run_meeting_from_audio(agent, transcript_path: str,
                            twin_name: str,
                            confident_domains: list[str],
                            voice: str = DEFAULT_VOICE,
                            auto_play: bool = True) -> list[dict]:
    """Process a meeting transcript file and generate responses.

    Transcript format (one utterance per line):
        Speaker: text
    """
    bot = MeetingBot(
        agent=agent,
        twin_name=twin_name,
        confident_domains=confident_domains,
        voice=voice,
        auto_play=auto_play,
    )

    responses = []
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Parse "Speaker: text" format
            if "：" in line:
                speaker, text = line.split("：", 1)
            elif ":" in line:
                speaker, text = line.split(":", 1)
            else:
                speaker, text = "未知", line

            result = bot.process_utterance(speaker.strip(), text.strip())
            if result:
                responses.append(result)

    return responses


def run_realtime_meeting(agent, twin_name: str,
                          confident_domains: list[str],
                          voice: str = DEFAULT_VOICE,
                          record_duration: float = 3.0):
    """Real-time meeting participation with microphone input."""
    from rich.console import Console
    from multimodal.stt import STTEngine

    console = Console()
    bot = MeetingBot(
        agent=agent,
        twin_name=twin_name,
        confident_domains=confident_domains,
        voice=voice,
        auto_play=True,
    )
    stt = STTEngine()

    console.print(f"[bold]实时会议模式[/bold] — 监听中 (Ctrl+C 退出)")
    console.print(f"分身名称：{twin_name} | 领域：{', '.join(confident_domains[:3])}")

    while True:
        try:
            # Record a short segment
            from multimodal.stt import record_once
            audio_path = record_once(duration=record_duration)
            text = stt.transcribe(audio_path)
            os.unlink(audio_path)

            if not text.strip():
                continue

            console.print(f"[dim]监听：{text[:80]}[/dim]")

            # Detect speaker (simple heuristic: not self)
            result = bot.process_utterance("参会者", text)
            if result:
                console.print(f"[cyan]发言：{result['response'][:100]}[/cyan]")

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]错误：{e}[/red]")
            continue

    agent.end_session()
    console.print("[dim]会议结束[/dim]")