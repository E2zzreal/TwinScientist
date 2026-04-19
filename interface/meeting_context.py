# interface/meeting_context.py
"""Meeting context manager: maintains sliding window + rolling summary."""
import time
from agent.tokens import count_tokens


class MeetingContext:
    """Maintains meeting state within a fixed token budget.

    Keeps:
    - Meeting metadata (topic, participants, discussed topics)
    - Recent utterances as raw text (sliding window)
    - Rolling summary of older utterances
    """

    def __init__(self, window_minutes: float = 5.0,
                 max_tokens: int = 4000,
                 compressor_fn=None):
        """
        Args:
            window_minutes: How long to keep raw utterances before compressing.
            max_tokens: Max tokens for recent_utterances before triggering compression.
            compressor_fn: Optional fn(utterances) -> str for LLM compression.
        """
        self.window_minutes = window_minutes
        self.max_tokens = max_tokens
        self.compressor_fn = compressor_fn

        # Meeting metadata
        self.topic: str = ""
        self.participants: list[str] = []
        self.discussed_topics: list[str] = []

        # Content
        self.recent_utterances: list[dict] = []  # {speaker, text, timestamp}
        self.rolling_summary: str = ""

    def add_utterance(self, speaker: str, text: str):
        """Record a new utterance and compress if needed."""
        self.recent_utterances.append({
            "speaker": speaker,
            "text": text,
            "timestamp": time.time(),
        })
        # Add speaker to participants if new
        if speaker and speaker not in self.participants:
            self.participants.append(speaker)

        # Compress if over budget
        if self._count_recent_tokens() > self.max_tokens:
            self._compress_old_utterances()

    def get_snapshot(self) -> dict:
        """Return current meeting state as a dict."""
        recent_text = "\n".join(
            f"{u['speaker']}: {u['text']}"
            for u in self.recent_utterances[-20:]  # last 20 utterances max
        )
        return {
            "topic": self.topic,
            "participants": self.participants,
            "discussed_topics": self.discussed_topics,
            "rolling_summary": self.rolling_summary,
            "recent_text": recent_text,
        }

    def format_for_agent(self) -> str:
        """Format meeting context for injection into agent system prompt."""
        snap = self.get_snapshot()
        parts = ["## 当前会议上下文\n"]

        if snap["topic"]:
            parts.append(f"**会议主题：** {snap['topic']}\n")
        if snap["participants"]:
            parts.append(f"**参会者：** {', '.join(snap['participants'])}\n")
        if snap["discussed_topics"]:
            parts.append(
                f"**已讨论话题：** {', '.join(snap['discussed_topics'])}\n"
            )
        if snap["rolling_summary"]:
            parts.append(f"**早期讨论摘要：**\n{snap['rolling_summary']}\n")
        if snap["recent_text"]:
            parts.append(f"**最近发言：**\n{snap['recent_text']}\n")

        return "\n".join(parts)

    def _count_recent_tokens(self) -> int:
        text = " ".join(u["text"] for u in self.recent_utterances)
        return count_tokens(text)

    def _compress_old_utterances(self):
        """Compress the oldest half of utterances into rolling_summary."""
        if len(self.recent_utterances) < 4:
            return

        cutoff = len(self.recent_utterances) // 2
        to_compress = self.recent_utterances[:cutoff]
        self.recent_utterances = self.recent_utterances[cutoff:]

        compressed_text = "\n".join(
            f"{u['speaker']}: {u['text']}" for u in to_compress
        )

        if self.compressor_fn:
            new_summary = self.compressor_fn(to_compress)
        else:
            # Naive fallback: keep first 200 chars of each
            lines = [
                f"{u['speaker']}: {u['text'][:100]}"
                for u in to_compress
            ]
            new_summary = "\n".join(lines)

        if self.rolling_summary:
            self.rolling_summary = self.rolling_summary + "\n" + new_summary
        else:
            self.rolling_summary = new_summary