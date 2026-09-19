"""Prompt construction A_t = (P, Σ_t, O_t) and response parsing (R_t, ΔΣ_t, a_t).

The template follows Appendix A.4 of the paper. Note that the model never
sees previous observations, actions, or reasoning; only an optional
single-line error from a rejected attempt *within the current step* is
added on retry, which keeps the prompt bounded.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

PROMPT_TEMPLATE = """Instructions:
{instructions}

Skill Execution State:
```json
{state}
```

Latest Observation: {observation}
{retry}
Provide your response with:
1. Step-by-step reasoning (will be discarded after execution)
2. A JSON block fenced with ```json ... ``` containing both your State Patch and your Action. The JSON block MUST have exactly these two keys: {{ "state_patch": {{ <dict: your state updates, set keys to null to delete> }}, "action": "<string: the exact command you want to execute>" }}
State patches are deep-merged: include only the keys that change; nested objects merge key-by-key; lists replace wholesale."""


def build_prompt(instructions: str, state: dict, observation: str,
                 retry_error: str | None = None) -> str:
    retry = ""
    if retry_error:
        retry = (f"\nYour previous response for this step was rejected: {retry_error}\n"
                 "The state shown above is unchanged. Try again.\n")
    return PROMPT_TEMPLATE.format(
        instructions=instructions,
        state=json.dumps(state, separators=(",", ":"), sort_keys=True),
        observation=observation,
        retry=retry,
    )


class ParseError(ValueError):
    pass


@dataclass
class ParsedResponse:
    reasoning: str            # R_t (discarded by the runtime after the step)
    state_patch: dict[str, Any]  # ΔΣ_t
    action: str               # a_t


_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_response(text: str) -> ParsedResponse:
    """Extract the last fenced JSON block holding exactly {state_patch, action}."""
    blocks = _FENCE.findall(text)
    if not blocks:
        raise ParseError("no ```json fenced block found")
    raw = blocks[-1].strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ParseError(f"invalid JSON ({e.msg} at line {e.lineno} col {e.colno})") from None
    if not isinstance(obj, dict) or set(obj) != {"state_patch", "action"}:
        keys = sorted(obj) if isinstance(obj, dict) else type(obj).__name__
        raise ParseError(f"JSON block must have exactly keys state_patch and action, got {keys}")
    if not isinstance(obj["state_patch"], dict):
        raise ParseError("state_patch must be an object")
    if not isinstance(obj["action"], str) or not obj["action"].strip():
        raise ParseError("action must be a non-empty string")
    reasoning = _FENCE.sub("", text).strip()
    return ParsedResponse(reasoning, obj["state_patch"], obj["action"].strip())
