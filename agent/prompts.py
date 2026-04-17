# agent/prompts.py
import os

TOOL_INSTRUCTIONS = """
## 可用工具

你有以下工具可以使用：

- **recall**: 回忆特定话题的详细知识。当对话涉及具体论文、数据或需要深入观点时调用。
  参数: topic (话题ID), depth ("summary" | "detail" | "specific_paper")
- **save_to_memory**: 将对话中产生的新观点存入长期记忆。
  参数: topic (话题ID), content (内容), source (来源)

## 回答原则

- 用你自己的风格和语气说话，不要用学术论文腔
- 观点要能追溯到已有的立场或经验
- 如果话题超出你的专业范围，明确说"这个我不太熟"
- 宁可说"不知道"也不编造
"""

def _read_file(path: str) -> str:
    """Read file content, return empty string if not found."""
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_system_prompt(persona_dir: str, memory_dir: str) -> str:
    """Assemble the system prompt from persona and memory files."""
    identity = _read_file(os.path.join(persona_dir, "identity.yaml"))
    frameworks = _read_file(os.path.join(persona_dir, "thinking_frameworks.yaml"))
    boundaries = _read_file(os.path.join(persona_dir, "boundaries.yaml"))
    topic_index = _read_file(os.path.join(memory_dir, "topic_index.yaml"))

    sections = [
        "# 你是一位科研人员的数字分身\n",
        "## 身份\n",
        identity,
        "\n## 能力边界\n",
        boundaries,
        "\n## 知识领域\n",
        topic_index,
        "\n## 思维方式\n",
        frameworks,
        "\n",
        TOOL_INSTRUCTIONS,
    ]
    return "\n".join(sections)