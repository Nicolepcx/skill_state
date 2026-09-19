"""SKILL.state runtime (paper Algorithm 1).

    for t = 0..T:
        receive O_t
        prompt  <- (P, Σ_t, O_t)
        (R_t, ΔΣ_t, a_t) <- LLM(prompt)
        validate ΔΣ_t              # rollback-retry on failure (paper §7)
        Σ_{t+1} <- Σ_t ⊕ ΔΣ_t
        execute a_t                # R_t is dropped here, never re-fed
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .llm import LLM
from .prompt import ParseError, build_prompt, parse_response
from .schema import validate as schema_validate
from .state import State, merge

StateCheck = Callable[[State], list[str]]


class Environment(Protocol):
    def reset(self) -> str: ...
    def step(self, action: str) -> tuple[str, bool]: ...  # (O_{t+1}, done)


@dataclass
class Skill:
    """P: the immutable procedural specification plus its domain schema."""
    name: str
    instructions: str
    schema: dict | None = None
    initial_state: State = field(default_factory=dict)
    checks: list[StateCheck] = field(default_factory=list)  # extra domain invariants
    fallback_action: str | None = None  # used if all retries fail; None = abort run


@dataclass
class StepRecord:
    t: int
    observation: str
    action: str
    attempts: int
    fell_back: bool
    prompt_chars: int


@dataclass
class RunResult:
    final_state: State
    steps: list[StepRecord] = field(default_factory=list)
    llm_calls: int = 0
    rejected_patches: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    aborted: bool = False

    @property
    def avg_prompt_chars(self) -> float:
        return sum(s.prompt_chars for s in self.steps) / max(len(self.steps), 1)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class SkillStateRuntime:
    def __init__(self, skill: Skill, llm: LLM, *, max_retries: int = 2,
                 max_state_chars: int | None = None,
                 on_step: Callable[[dict[str, Any]], None] | None = None):
        """
        max_state_chars: optional hard cap on serialized Σ. The O(1) prompt
            bound only holds if |Σ| is bounded, so enforce it for open-ended domains.
        on_step: optional logging hook. It receives the step's reasoning for
            audit/debugging; that trace is never fed back into any prompt.
        """
        self.skill, self.llm = skill, llm
        self.max_retries = max_retries
        self.max_state_chars = max_state_chars
        self.on_step = on_step

    # -- validation ---------------------------------------------------------
    def _check(self, state: State) -> list[str]:
        errors: list[str] = []
        if self.skill.schema is not None:
            errors += schema_validate(state, self.skill.schema)
            if errors:
                return errors  # domain checks may assume schema-valid shape
        for check in self.skill.checks:
            try:
                errors += check(state)
            except Exception as e:  # a buggy check must reject, not crash the run
                errors.append(f"check {getattr(check, '__name__', check)} failed: {e!r}")
        if self.max_state_chars is not None:
            size = len(json.dumps(state, separators=(",", ":")))
            if size > self.max_state_chars:
                errors.append(f"state size {size} exceeds cap {self.max_state_chars}; "
                              "delete keys that are no longer needed")
        return errors

    # -- main loop ----------------------------------------------------------
    def run(self, env: Environment, max_steps: int) -> RunResult:
        state: State = copy.deepcopy(self.skill.initial_state)
        init_errors = self._check(state)
        if init_errors:
            raise ValueError(f"initial state invalid: {init_errors}")

        result = RunResult(final_state=state)
        observation = env.reset()

        for t in range(max_steps):
            retry_error: str | None = None
            committed = None
            prompt_chars = 0

            for attempt in range(1, self.max_retries + 2):
                prompt = build_prompt(self.skill.instructions, state, observation, retry_error)
                prompt_chars = len(prompt)
                reply = self.llm.complete(prompt)
                result.llm_calls += 1
                result.input_tokens += reply.input_tokens or len(prompt) // 4
                result.output_tokens += reply.output_tokens or len(reply.text) // 4

                try:
                    parsed = parse_response(reply.text)
                    candidate = merge(state, parsed.state_patch)
                    errors = self._check(candidate)
                    if errors:
                        raise ParseError("; ".join(errors[:5]))
                except (ParseError, TypeError) as e:
                    result.rejected_patches += 1
                    retry_error = str(e)          # Σ_t untouched: rollback
                    continue

                committed = (candidate, parsed)
                break

            if committed is None:
                if self.skill.fallback_action is None:
                    result.aborted = True
                    break
                action, fell_back, reasoning = self.skill.fallback_action, True, None
            else:
                state, parsed = committed         # Σ_{t+1} = Σ_t ⊕ ΔΣ_t
                action, fell_back, reasoning = parsed.action, False, parsed.reasoning

            if self.on_step:
                self.on_step({"t": t, "observation": observation, "action": action,
                              "reasoning": reasoning, "state": copy.deepcopy(state),
                              "attempts": attempt, "fell_back": fell_back})
            del reasoning                         # R_t discarded permanently

            result.steps.append(StepRecord(t, observation, action, attempt,
                                           fell_back, prompt_chars))
            observation, done = env.step(action)
            if done:
                break

        result.final_state = state
        return result
