# agent/tools.py
import os
import yaml


class RecallTool:
    """Progressive disclosure memory recall — reads structured YAML files."""

    def __init__(self, memory_dir: str):
        self.memory_dir = memory_dir

    def _load_topic_index(self) -> dict:
        """Hot-reload topic index on every call."""
        path = os.path.join(self.memory_dir, "topic_index.yaml")
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("topics", {})

    def execute(self, topic: str, depth: str = "summary",
                paper_id: str | None = None) -> str:
        topic_index = self._load_topic_index()

        if depth == "conversations":
            return self._recall_conversations(topic)

        if topic not in topic_index:
            return f"关于「{topic}」没有找到相关记忆。"

        if depth == "summary":
            return self._recall_summary(topic, topic_index)
        elif depth == "detail":
            return self._recall_detail(topic, topic_index)
        elif depth == "specific_paper":
            return self._recall_paper(topic, paper_id, topic_index)
        else:
            return self._recall_summary(topic, topic_index)

    def _recall_summary(self, topic: str, topic_index: dict) -> str:
        info = topic_index[topic]
        return yaml.dump(info, allow_unicode=True, default_flow_style=False)

    def _recall_detail(self, topic: str, topic_index: dict) -> str:
        detail_path = os.path.join(
            self.memory_dir, "topics", f"{topic}.yaml"
        )
        if not os.path.exists(detail_path):
            return self._recall_summary(topic, topic_index)
        with open(detail_path, "r", encoding="utf-8") as f:
            return f.read()

    def _recall_paper(self, topic: str, paper_id: str | None,
                      topic_index: dict) -> str:
        if not paper_id:
            return self._recall_detail(topic, topic_index)
        info = topic_index.get(topic, {})
        for detail_file in info.get("detail_files", []):
            if paper_id in detail_file:
                paper_path = os.path.join(self.memory_dir, detail_file)
                if os.path.exists(paper_path):
                    with open(paper_path, "r", encoding="utf-8") as f:
                        return f.read()
        return f"关于「{paper_id}」没有找到具体论文记忆。"

    def _recall_conversations(self, topic: str) -> str:
        """Search conversation memories for a given topic."""
        conv_dir = os.path.join(self.memory_dir, "conversations")
        if not os.path.exists(conv_dir):
            return f"没有找到关于「{topic}」的对话记忆。"

        results = []
        for filename in sorted(os.listdir(conv_dir)):
            if not filename.endswith(".yaml"):
                continue
            filepath = os.path.join(conv_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                entry = yaml.safe_load(f) or {}
            if entry.get("topic") == topic:
                results.append(
                    f"[{entry.get('timestamp', '')}] {entry.get('content', '')}"
                )

        if not results:
            return f"没有找到关于「{topic}」的对话记忆。"
        return "\n---\n".join(results)


class SaveToMemoryTool:
    """Persist new insights from conversation to memory files."""

    def __init__(self, memory_dir: str):
        self.memory_dir = memory_dir

    def execute(self, topic: str, content: str, source: str = "conversation") -> str:
        conv_dir = os.path.join(self.memory_dir, "conversations")
        os.makedirs(conv_dir, exist_ok=True)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}-{topic}.yaml"
        filepath = os.path.join(conv_dir, filename)

        entry = {
            "topic": topic,
            "content": content,
            "source": source,
            "timestamp": timestamp,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(entry, f, allow_unicode=True)

        return f"已保存到 {filename}"