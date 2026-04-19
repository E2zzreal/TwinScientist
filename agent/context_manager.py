# agent/context_manager.py
from agent.tokens import count_tokens


class ContextManager:
    """Manages conversation history, dynamic memory, and context budget."""

    def __init__(self, config: dict):
        budget = config.get("context_budget", {})
        self.total_budget = budget.get("total", 100000)
        self.fixed_budget = budget.get("fixed_zone", 8000)
        self.dynamic_budget = budget.get("dynamic_zone", 12000)
        self.conversation_budget = budget.get("conversation_zone", 80000)

        conv = config.get("conversation", {})
        self.recent_window = conv.get("recent_window", 10)
        self.compression_trigger = conv.get("compression_trigger", 10)
        self.emergency_threshold = conv.get("emergency_threshold", 70000)

        # Conversation state
        self._turns: list[tuple[str, str]] = []  # (user_msg, assistant_msg)
        self._summary: str = ""  # compressed older turns

        # Dynamic zone
        self._dynamic: dict[str, str] = {}  # topic -> content
        self._dynamic_access_order: list[str] = []

    def add_turn(self, user_msg: str, assistant_msg: str):
        """Record a completed conversation turn."""
        self._turns.append((user_msg, assistant_msg))

    def prepare(self, user_message: str):
        """Prepare context before a new LLM call. Compress if needed."""
        if len(self._turns) > self.recent_window:
            self._compress()

    def get_history(self) -> list[dict]:
        """Return message history formatted for the API."""
        messages = []

        # Add summary of older turns if exists
        if self._summary:
            messages.append({
                "role": "user",
                "content": f"[之前的对话摘要]\n{self._summary}",
            })
            messages.append({
                "role": "assistant",
                "content": "好的，我记住了之前讨论的内容。",
            })

        # Add recent turns
        for user_msg, assistant_msg in self._turns[-self.recent_window:]:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})

        return messages

    def _compress(self):
        """Compress older turns into a summary."""
        overflow_count = len(self._turns) - self.recent_window
        if overflow_count <= 0:
            return

        old_turns = self._turns[:overflow_count]
        # Simple compression: concatenate key points
        new_summary_parts = []
        if self._summary:
            new_summary_parts.append(self._summary)
        for user_msg, assistant_msg in old_turns:
            new_summary_parts.append(f"- 用户问: {user_msg[:100]}")
            new_summary_parts.append(f"  回答要点: {assistant_msg[:100]}")

        self._summary = "\n".join(new_summary_parts)
        self._turns = self._turns[overflow_count:]

    # --- Dynamic zone management ---

    def load_dynamic(self, topic: str, content: str):
        """Load content into the dynamic zone."""
        tokens = count_tokens(content)
        # Evict LRU if over budget
        while (self._dynamic_tokens() + tokens > self.dynamic_budget
               and self._dynamic_access_order):
            evict = self._dynamic_access_order.pop(0)
            self._dynamic.pop(evict, None)

        self._dynamic[topic] = content
        if topic in self._dynamic_access_order:
            self._dynamic_access_order.remove(topic)
        self._dynamic_access_order.append(topic)

    def unload_dynamic(self, topic: str):
        """Remove content from the dynamic zone."""
        self._dynamic.pop(topic, None)
        if topic in self._dynamic_access_order:
            self._dynamic_access_order.remove(topic)

    def get_loaded_topics(self) -> list[str]:
        """Return currently loaded dynamic topics."""
        return list(self._dynamic.keys())

    def get_dynamic_content(self) -> str:
        """Return all dynamic zone content for injection."""
        if not self._dynamic:
            return ""
        parts = ["## 当前加载的详细记忆\n"]
        for topic, content in self._dynamic.items():
            parts.append(f"### {topic}\n{content}\n")
        return "\n".join(parts)

    def _dynamic_tokens(self) -> int:
        return sum(count_tokens(v) for v in self._dynamic.values())

    # --- Budget status ---

    def get_budget_status(self) -> dict:
        conversation_used = sum(
            count_tokens(u) + count_tokens(a) for u, a in self._turns
        ) + count_tokens(self._summary)
        dynamic_used = self._dynamic_tokens()

        return {
            "fixed_zone": self.fixed_budget,
            "dynamic_zone": {"budget": self.dynamic_budget, "used": dynamic_used},
            "conversation_zone": {
                "budget": self.conversation_budget,
                "used": conversation_used,
            },
            "conversation_used": conversation_used,
        }