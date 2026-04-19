# agent/llm_client.py
"""
Unified LLM client adapter supporting Anthropic and OpenAI-compatible APIs.

Usage in config.yaml:
  provider: anthropic          # default, uses ANTHROPIC_API_KEY
  provider: openai_compatible  # uses OPENAI_API_KEY + base_url
  base_url: https://api.deepseek.com/v1
  model: deepseek-chat
"""
import json
import os
from dataclasses import dataclass, field

import anthropic
from openai import OpenAI


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class ChatResponse:
    """Unified response from either backend."""
    stop_reason: str          # "end_turn" | "tool_use"
    text: str                 # populated when stop_reason == "end_turn"
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_assistant_message: object = None  # backend-specific, for appending to history


def _anthropic_tools_to_openai(tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool format to OpenAI function-calling format."""
    result = []
    for t in tools:
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return result


class LLMClient:
    """Unified chat client. Hides Anthropic vs OpenAI format differences."""

    def __init__(self, config: dict):
        self.provider = config.get("provider", "anthropic")
        self.model = config["model"]
        self._config = config

        if self.provider == "anthropic":
            self._client = anthropic.Anthropic(
                api_key=config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
            )
        else:
            # openai_compatible: DeepSeek, Qwen, Together, etc.
            self._client = OpenAI(
                api_key=config.get("api_key") or os.environ.get("OPENAI_API_KEY"),
                base_url=config.get("base_url", "https://api.openai.com/v1"),
            )

    # ── Public interface ────────────────────────────────────────────────────

    def chat(self, system: str, messages: list[dict],
             tools: list[dict], max_tokens: int) -> ChatResponse:
        """Send a chat request and return a unified ChatResponse."""
        if self.provider == "anthropic":
            return self._chat_anthropic(system, messages, tools, max_tokens)
        else:
            return self._chat_openai(system, messages, tools, max_tokens)

    def tool_result_message(self, tool_calls: list[ToolCall],
                            results: list[str]) -> list[dict]:
        """Build the tool-result message(s) to append after tool calls."""
        if self.provider == "anthropic":
            return [{
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": result,
                    }
                    for tc, result in zip(tool_calls, results)
                ],
            }]
        else:
            # OpenAI: one message per tool result
            return [
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
                for tc, result in zip(tool_calls, results)
            ]

    def simple_chat(self, prompt: str, max_tokens: int = 400) -> str:
        """Single-turn chat without tools (used for compression, extraction)."""
        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        else:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return (response.choices[0].message.content or "").strip()

    # ── Anthropic backend ───────────────────────────────────────────────────

    def _chat_anthropic(self, system, messages, tools, max_tokens) -> ChatResponse:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools,
        )

        if response.stop_reason == "end_turn":
            text = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )
            return ChatResponse(
                stop_reason="end_turn",
                text=text,
                raw_assistant_message={"role": "assistant", "content": response.content},
            )
        else:
            # tool_use
            calls = [
                ToolCall(id=block.id, name=block.name, input=block.input)
                for block in response.content
                if block.type == "tool_use"
            ]
            return ChatResponse(
                stop_reason="tool_use",
                text="",
                tool_calls=calls,
                raw_assistant_message={"role": "assistant", "content": response.content},
            )

    # ── OpenAI-compatible backend ───────────────────────────────────────────

    def _chat_openai(self, system, messages, tools, max_tokens) -> ChatResponse:
        # Prepend system as first message
        full_messages = [{"role": "system", "content": system}] + messages

        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=full_messages,
        )
        if tools:
            kwargs["tools"] = _anthropic_tools_to_openai(tools)
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        finish = choice.finish_reason
        msg = choice.message

        if finish == "tool_calls" and msg.tool_calls:
            calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    input=json.loads(tc.function.arguments or "{}"),
                )
                for tc in msg.tool_calls
            ]
            # Convert to dict for history appending
            raw = {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
            return ChatResponse(
                stop_reason="tool_use",
                text="",
                tool_calls=calls,
                raw_assistant_message=raw,
            )
        else:
            return ChatResponse(
                stop_reason="end_turn",
                text=(msg.content or "").strip(),
                raw_assistant_message={"role": "assistant", "content": msg.content},
            )

    # ── Vision support ─────────────────────────────────────────────────────

    def vision_chat(self, prompt: str, image_b64: str,
                    media_type: str = "image/jpeg",
                    max_tokens: int = 800) -> str:
        """Send an image + text prompt, return text description.

        Args:
            prompt: Question or instruction about the image.
            image_b64: Base64-encoded image bytes.
            media_type: MIME type, e.g. "image/png", "image/jpeg".
            max_tokens: Max tokens in response.

        Returns:
            Text description/analysis from the model.

        Raises:
            RuntimeError: If the model does not support vision.
        """
        try:
            if self.provider == "anthropic":
                return self._vision_anthropic(prompt, image_b64, media_type, max_tokens)
            else:
                return self._vision_openai(prompt, image_b64, media_type, max_tokens)
        except RuntimeError:
            raise
        except Exception as e:
            if "image" in str(e).lower() or "vision" in str(e).lower() or "multimodal" in str(e).lower():
                raise RuntimeError(
                    f"视觉功能不支持：当前模型 ({self.model}) 不支持图像输入。\n"
                    "请在 config.yaml 中切换到支持视觉的模型，例如：\n"
                    "  Anthropic: claude-sonnet-4-20250514\n"
                    "  OpenAI: gpt-4o\n"
                    "  Qwen: qwen-vl-plus\n"
                    "  DeepSeek: deepseek-vl2"
                ) from e
            raise

    def _vision_anthropic(self, prompt: str, image_b64: str,
                          media_type: str, max_tokens: int) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.content[0].text.strip()

    def _vision_openai(self, prompt: str, image_b64: str,
                       media_type: str, max_tokens: int) -> str:
        data_url = f"data:{media_type};base64,{image_b64}"
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return (response.choices[0].message.content or "").strip()
