# skill_state

A dependency-free implementation of the SKILL.state runtime (Badhe et al., 2026):
each step the model sees only `(P, Σ_t, O_t)`, returns reasoning + a state patch +
an action, the runtime validates and merges the patch, and the reasoning is discarded.

```
skill_state/
  state.py      Σ ⊕ ΔΣ  (deep merge, null deletes = RFC 7386 JSON Merge Patch)
  schema.py     deterministic validator (JSON Schema subset)
  prompt.py     Appendix A.4 prompt + strict response parser
  runtime.py    Algorithm 1 loop, rollback-retry, metrics, state-size cap
  llm.py        LLM protocol, CallableLLM (any provider), AnthropicLLM
  envs/warehouse.py   SkillExecBench Env 1 (Appendix B) + noise (App. C) + drift (Exp. 3)
examples/run_warehouse.py
tests/test_core.py
```

## Plugging into your harness

```python
from skill_state import Skill, SkillStateRuntime, CallableLLM

skill = Skill(
    name="ctf",
    instructions=open("SKILL.md").read(),          # P, immutable
    schema={...},                                   # authored once per domain
    initial_state={"discovered_flags": [], "tested_hypotheses": [],
                   "active_files": [], "working_dir": "/", "cmd_summary": ""},
    checks=[my_invariant],                          # optional: state -> [errors]
    fallback_action=None,                           # None = abort after retries
)
runtime = SkillStateRuntime(skill, CallableLLM(my_model_fn),
                            max_retries=2, max_state_chars=8000)
result = runtime.run(env, max_steps=100)   # env: reset() -> obs; step(a) -> (obs, done)
```

## Decisions where the paper is underspecified

- **⊕ is a deep merge.** A shallow merge would make the paper's own B.3 example
  (`{"inventory": {"shelf_42": null}}`) wipe every other shelf. Lists replace
  wholesale, so patch list fields by sending the full new list.
- **Validation is on the merged state**, not the patch alone, so invariants
  (e.g. no item on two shelves) are checked against what would actually be committed.
- **Retry feedback.** On a rejected patch Σ_t is unchanged and the retry prompt
  carries only that single error string, which keeps prompts bounded.
- **State is committed before the action runs** (Algorithm 1 order). If the action
  then fails, the model learns it from O_{t+1} and must correct Σ itself. If your
  environment gives synchronous results, consider committing only on success.
- **Bounded prompt needs bounded Σ.** The O(1) claim assumes |Σ| is bounded;
  `max_state_chars` enforces it by rejecting patches that grow state past the cap.
- **Reasoning goes to `on_step` only**, for audit logs, and never into a prompt,
  which covers the paper's third limitation (tasks about the trajectory itself).

## Run

```
python examples/run_warehouse.py                     # scripted offline policy
python examples/run_warehouse.py --faulty --noise 20 # injected bad patches + telemetry
python examples/run_warehouse.py --anthropic         # real model, needs ANTHROPIC_API_KEY
python -m pytest -q tests
```
# skill_state
