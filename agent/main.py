# agent/main.py
import os
from agent.config import load_config
from agent.prompts import build_system_prompt
from agent.context_manager import ContextManager
from agent.tools import RecallTool, SaveToMemoryTool
from agent.llm_client import LLMClient

# Tool definitions (Anthropic format — adapter converts for OpenAI backends)
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
                    "enum": ["summary", "detail", "specific_paper", "conversations"],
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
    {
        "name": "give_feedback",
        "description": "记录用户对回答的反馈，触发风格校正或立场更新。当用户说'不像我'时用feedback_type=style；当用户说'我现在不这么看了'时用feedback_type=stance。",
        "input_schema": {
            "type": "object",
            "properties": {
                "feedback_type": {
                    "type": "string",
                    "enum": ["style", "stance"],
                    "description": "style=风格校正，stance=立场更新",
                },
                "feedback": {"type": "string", "description": "用户的反馈内容（style类型必填）"},
                "original_response": {"type": "string", "description": "被反馈的原始回答（style类型必填）"},
                "context": {"type": "string", "description": "对话上下文简述"},
                "topic": {"type": "string", "description": "相关话题ID（stance类型必填）"},
                "new_stance": {"type": "string", "description": "新的立场（stance类型必填）"},
                "reason": {"type": "string", "description": "立场变化的原因"},
            },
            "required": ["feedback_type"],
        },
    },
    {
        "name": "see",
        "description": "观察并分析一张图片（论文图表、实验截图、PPT页面）。当用户分享图片或提到图表时调用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "图片文件的路径",
                },
                "context": {
                    "type": "string",
                    "description": "图片的背景信息，如来源、实验类型等",
                },
            },
            "required": ["image_path"],
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
        self.client = LLMClient(self.config)

        # Tools
        self.recall_tool = RecallTool(self.memory_dir)
        self.save_tool = SaveToMemoryTool(self.memory_dir)

        # Inject LLM compressor into ContextManager
        self.context.set_llm_compressor(self._llm_compress)

        # Load previous session summary
        from agent.session import load_latest_session_summary
        prev_summary = load_latest_session_summary(self.memory_dir)
        if prev_summary:
            self.context._summary = f"[上次会话摘要]\n{prev_summary}"

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
            response = self.client.chat(
                system=self.build_system_prompt(),
                messages=messages,
                tools=TOOL_DEFINITIONS,
                max_tokens=self.config.get("max_tokens", 4096),
            )

            if response.stop_reason == "end_turn":
                break

            # Append assistant message to history
            messages.append(response.raw_assistant_message)

            # Execute tools and collect results
            results = [
                self._execute_tool(tc.name, tc.input)
                for tc in response.tool_calls
            ]

            # Append tool results
            for msg in self.client.tool_result_message(response.tool_calls, results):
                messages.append(msg)

        self.context.add_turn(user_message, response.text)
        return response.text

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool call and return the result as string."""
        if tool_name == "recall":
            result = self.recall_tool.execute(
                topic=tool_input.get("topic", ""),
                depth=tool_input.get("depth", "summary"),
                paper_id=tool_input.get("paper_id"),
            )
            self.context.load_dynamic(tool_input.get("topic", ""), result)
            return result

        elif tool_name == "save_to_memory":
            return self.save_tool.execute(
                topic=tool_input.get("topic", ""),
                content=tool_input.get("content", ""),
                source=tool_input.get("source", "conversation"),
            )

        elif tool_name == "give_feedback":
            return self._handle_feedback(tool_input)

        elif tool_name == "see":
            return self._execute_see(tool_input)

        return f"Unknown tool: {tool_name}"

    def _llm_compress(self, turns: list[tuple[str, str]]) -> str:
        """Compress conversation turns into a concise summary."""
        turns_text = "\n".join(f"用户: {u}\n回答: {a}" for u, a in turns)
        prompt = f"""以下是一段对话记录，请压缩成简洁摘要（不超过300字）。
保留：话题、关键结论、重要观点、提到的具体数据或论文。
舍弃：重复内容、寒暄、过渡语句。

对话记录：
{turns_text}

摘要："""
        return self.client.simple_chat(prompt, max_tokens=400)

    def end_session(self):
        """Persist current session summary to disk for next session."""
        from agent.session import save_session_summary
        all_turns = list(self.context._turns)
        if not all_turns and not self.context._summary:
            return
        summary_parts = []
        if self.context._summary:
            summary_parts.append(self.context._summary)
        if all_turns:
            summary_parts.append(self._llm_compress(all_turns))
        save_session_summary(self.memory_dir, "\n".join(summary_parts))

    def _handle_feedback(self, tool_input: dict) -> str:
        """Route feedback to the appropriate evolution handler."""
        from agent.evolution import apply_style_correction, apply_stance_update

        changelog_path = os.path.join(
            self.project_dir,
            self.config.get("paths", {}).get("evolution_dir", "evolution"),
            "changelog.yaml",
        )
        if not os.path.exists(changelog_path):
            import yaml
            os.makedirs(os.path.dirname(changelog_path), exist_ok=True)
            with open(changelog_path, "w") as f:
                yaml.dump({"changes": []}, f)

        feedback_type = tool_input.get("feedback_type", "style")

        if feedback_type == "style":
            apply_style_correction(
                client=self.client,
                model=self.config["model"],
                persona_dir=self.persona_dir,
                changelog_path=changelog_path,
                original_response=tool_input.get("original_response", ""),
                feedback=tool_input.get("feedback", ""),
                context=tool_input.get("context", ""),
            )
            return "已记录风格反馈，下次对话将体现改进。"

        elif feedback_type == "stance":
            apply_stance_update(
                memory_dir=self.memory_dir,
                changelog_path=changelog_path,
                topic=tool_input.get("topic", ""),
                new_stance=tool_input.get("new_stance", ""),
                reason=tool_input.get("reason", ""),
            )
            self.recall_tool = self.recall_tool.__class__(self.memory_dir)
            return f"已更新「{tool_input.get('topic', '')}」的立场。"

        return "未知的反馈类型。"

    def _execute_see(self, tool_input: dict) -> str:
        """Execute the see tool: analyze an image with scientific lens."""
        from multimodal.vision import analyze_figure_as_scientist
        import yaml as _yaml

        image_path = tool_input.get("image_path", "")
        context = tool_input.get("context", "")

        if not os.path.exists(image_path):
            return f"错误：找不到图片文件 {image_path}"

        # Build persona summary from identity.yaml
        try:
            with open(os.path.join(self.persona_dir, "identity.yaml"), "r") as f:
                identity = _yaml.safe_load(f) or {}
            research_focus = identity.get("research_focus", [])
            persona_summary = "、".join(research_focus[:3]) if research_focus else ""
        except Exception:
            persona_summary = ""

        try:
            return analyze_figure_as_scientist(
                client=self.client,
                image_path=image_path,
                figure_context=context,
                persona_summary=persona_summary,
            )
        except RuntimeError as e:
            return str(e)
