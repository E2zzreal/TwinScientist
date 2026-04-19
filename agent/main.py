# agent/main.py
import os
import anthropic
from agent.config import load_config
from agent.prompts import build_system_prompt
from agent.context_manager import ContextManager
from agent.tools import RecallTool, SaveToMemoryTool

# Tool definitions for the Claude API
TOOL_DEFINITIONS = [
    {
        "name": "recall",
        "description": "回忆特定话题的详细知识和立场。当需要具体论文、数据或深入观点时调用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "话题ID，如 hydrogen_catalyst",
                },
                "depth": {
                    "type": "string",
                    "enum": ["summary", "detail", "specific_paper"],
                    "description": "检索深度",
                },
                "paper_id": {
                    "type": "string",
                    "description": "具体论文ID，仅当 depth=specific_paper 时使用",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "save_to_memory",
        "description": "将对话中产生的新观点或知识存入长期记忆。",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "话题ID"},
                "content": {"type": "string", "description": "要保存的内容"},
                "source": {"type": "string", "description": "来源标记"},
            },
            "required": ["topic", "content"],
        },
    },
]


class TwinScientist:
    """Core agent: the digital twin of a researcher."""

    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        config_path = os.path.join(project_dir, "config.yaml")
        self.config = load_config(config_path)

        self.persona_dir = os.path.join(
            project_dir, self.config.get("paths", {}).get("persona_dir", "persona")
        )
        self.memory_dir = os.path.join(
            project_dir, self.config.get("paths", {}).get("memory_dir", "memory")
        )

        self.context = ContextManager(self.config)
        self.client = anthropic.Anthropic()

        # Tools
        self.recall_tool = RecallTool(self.memory_dir)
        self.save_tool = SaveToMemoryTool(self.memory_dir)

        # Inject LLM compressor into ContextManager
        self.context.set_llm_compressor(self._llm_compress)

    def build_system_prompt(self) -> str:
        base = build_system_prompt(self.persona_dir, self.memory_dir)
        dynamic = self.context.get_dynamic_content()
        if dynamic:
            base += "\n" + dynamic
        return base

    def chat(self, user_message: str) -> str:
        """Process one user message and return the agent's response."""
        self.context.prepare(user_message)

        messages = self.context.get_history()
        messages.append({"role": "user", "content": user_message})

        # Agent loop: may have multiple tool calls before final response
        while True:
            response = self.client.messages.create(
                model=self.config["model"],
                max_tokens=self.config.get("max_tokens", 4096),
                system=self.build_system_prompt(),
                messages=messages,
                tools=TOOL_DEFINITIONS,
            )

            if response.stop_reason == "end_turn":
                break

            # Process tool calls
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    result = self._execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})

        # Extract text from final response
        answer = ""
        for block in response.content:
            if hasattr(block, "text"):
                answer += block.text

        self.context.add_turn(user_message, answer)
        return answer

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool call and return the result as string."""
        if tool_name == "recall":
            result = self.recall_tool.execute(
                topic=tool_input.get("topic", ""),
                depth=tool_input.get("depth", "summary"),
                paper_id=tool_input.get("paper_id"),
            )
            # Load into dynamic zone for context awareness
            self.context.load_dynamic(tool_input.get("topic", ""), result)
            return result

        elif tool_name == "save_to_memory":
            return self.save_tool.execute(
                topic=tool_input.get("topic", ""),
                content=tool_input.get("content", ""),
                source=tool_input.get("source", "conversation"),
            )

        return f"Unknown tool: {tool_name}"

    def _llm_compress(self, turns: list[tuple[str, str]]) -> str:
        """Use Claude to compress conversation turns into a concise summary."""
        turns_text = "\n".join(
            f"用户: {u}\n回答: {a}" for u, a in turns
        )
        prompt = f"""以下是一段对话记录，请压缩成简洁摘要（不超过300字）。
保留：话题、关键结论、重要观点、提到的具体数据或论文。
舍弃：重复内容、寒暄、过渡语句。

对话记录：
{turns_text}

摘要："""

        response = self.client.messages.create(
            model=self.config["model"],
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def end_session(self):
        """Persist current session summary to disk for next session."""
        from agent.session import save_session_summary
        # Build a summary of this entire session
        all_turns = list(self.context._turns)
        if not all_turns and not self.context._summary:
            return

        summary_parts = []
        if self.context._summary:
            summary_parts.append(self.context._summary)

        # Use LLM to compress remaining turns if any
        if all_turns:
            compressed = self._llm_compress(all_turns)
            summary_parts.append(compressed)

        final_summary = "\n".join(summary_parts)
        save_session_summary(self.memory_dir, final_summary)