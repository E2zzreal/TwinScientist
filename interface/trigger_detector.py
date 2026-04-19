# interface/trigger_detector.py
"""Determines when the digital twin should speak in a meeting."""
import re
from dataclasses import dataclass
from enum import Enum


class TriggerType(Enum):
    DIRECT_MENTION = "direct_mention"   # 被点名
    DOMAIN_RELEVANT = "domain_relevant" # 涉及专业领域
    QUESTION = "question"               # 领域内直接提问
    NONE = "none"


@dataclass
class TriggerResult:
    should_speak: bool
    trigger_type: TriggerType
    urgency: str  # "high" | "medium" | "low" | "none"
    reason: str = ""

    @classmethod
    def no_trigger(cls) -> "TriggerResult":
        return cls(False, TriggerType.NONE, "none")


class TriggerDetector:
    """Rule-based trigger detection for meeting participation."""

    def __init__(self, twin_name: str,
                 confident_domains: list[str],
                 min_domain_keywords: int = 1):
        self.twin_name = twin_name
        self.confident_domains = confident_domains
        self.min_domain_keywords = min_domain_keywords

        # Name variants (e.g., 张老师, 张三老师, etc.)
        parts = twin_name.split()
        surname = parts[0][0] if parts else twin_name[0]
        self._name_patterns = [
            twin_name,
            f"{surname}老师",
            f"{twin_name}老师",
            f"@{twin_name}",
        ]

    def check(self, utterance: str) -> TriggerResult:
        """Check if the latest utterance should trigger a response."""
        # 1. Direct mention (highest priority)
        for pattern in self._name_patterns:
            if pattern in utterance:
                return TriggerResult(
                    should_speak=True,
                    trigger_type=TriggerType.DIRECT_MENTION,
                    urgency="high",
                    reason=f"被点名：'{pattern}' 出现在发言中",
                )

        # 2. Domain keyword match
        matched = [d for d in self.confident_domains if d in utterance]
        if len(matched) >= self.min_domain_keywords:
            # Higher urgency if it's also a question
            is_question = "？" in utterance or "?" in utterance or \
                          any(w in utterance for w in ["怎么看", "如何", "有没有", "请问"])
            return TriggerResult(
                should_speak=True,
                trigger_type=TriggerType.DOMAIN_RELEVANT,
                urgency="medium" if is_question else "low",
                reason=f"涉及领域关键词：{matched}",
            )

        return TriggerResult.no_trigger()

    def check_transcript_window(self, utterances: list[dict],
                                 window: int = 3) -> TriggerResult:
        """Check the last N utterances for triggers."""
        recent = utterances[-window:] if len(utterances) >= window else utterances
        for utt in reversed(recent):
            result = self.check(utt.get("text", ""))
            if result.should_speak:
                return result
        return TriggerResult.no_trigger()