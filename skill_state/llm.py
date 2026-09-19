"""LLM interface. The runtime only needs `complete(prompt) -> LLMResult`.

Each call is a fresh, single-turn request: SKILL.state never sends a
conversation transcript, so any chat API works with one user message.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass
class LLMResult:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLM(Protocol):
    def complete(self, prompt: str) -> LLMResult: ...


class CallableLLM:
    """Wrap any `fn(prompt) -> str` (Gemini, vLLM, Ollama, ...)."""

    def __init__(self, fn: Callable[[str], str]):
        self.fn = fn

    def complete(self, prompt: str) -> LLMResult:
        return LLMResult(self.fn(prompt))


class AnthropicLLM:
    """Adapter for the Anthropic Messages API (`pip install anthropic`)."""

    def __init__(self, model: str = "claude-sonnet-5", max_tokens: int = 2048,
                 temperature: float = 0.0):
        import anthropic  # lazy import: optional dependency
        self.client = anthropic.Anthropic()
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature

    def complete(self, prompt: str) -> LLMResult:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return LLMResult(text, msg.usage.input_tokens, msg.usage.output_tokens)
