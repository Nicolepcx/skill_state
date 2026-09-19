import json
import pytest
from skill_state import (LLMResult, ParseError, Skill, SkillStateRuntime,
                         build_prompt, merge, parse_response, validate)


# ---- ⊕ merge operator ----------------------------------------------------
def test_merge_is_deep_and_null_deletes():
    s = {"inventory": {"shelf_1": "a", "shelf_2": "b"}, "n": 1}
    out = merge(s, {"inventory": {"shelf_1": None, "shelf_3": "c"}})
    assert out == {"inventory": {"shelf_2": "b", "shelf_3": "c"}, "n": 1}
    assert s["inventory"]["shelf_1"] == "a"          # input not mutated

def test_merge_lists_replace_and_top_level_delete():
    assert merge({"l": [1, 2], "x": 1}, {"l": [3], "x": None}) == {"l": [3]}

def test_merge_new_subobject_strips_nulls():
    assert merge({}, {"a": {"b": None, "c": 1}}) == {"a": {"c": 1}}


# ---- parsing -------------------------------------------------------------
def test_parse_takes_last_block_and_separates_reasoning():
    text = 'think\n```json\n{"state_patch":{},"action":"A"}\n```\nmore\n```json\n{"state_patch":{"k":1},"action":"B"}\n```'
    p = parse_response(text)
    assert p.action == "B" and p.state_patch == {"k": 1} and "think" in p.reasoning

@pytest.mark.parametrize("bad", [
    "no block",
    '```json\n{"state_patch":{},}\n```',
    '```json\n{"state_patch":{},"action":"A","extra":1}\n```',
    '```json\n{"state_patch":[],"action":"A"}\n```',
    '```json\n{"state_patch":{},"action":""}\n```',
])
def test_parse_rejects(bad):
    with pytest.raises(ParseError):
        parse_response(bad)


# ---- schema --------------------------------------------------------------
def test_schema_subset():
    schema = {"type": "object", "required": ["a"], "additionalProperties": False,
              "properties": {"a": {"type": "array", "items": {"type": "integer"}}}}
    assert validate({"a": [1, 2]}, schema) == []
    assert validate({"a": [1, "x"], "b": 0}, schema)  # two errors
    assert validate({}, schema)


# ---- prompt is bounded: no history leaks in -------------------------------
def test_prompt_contains_only_P_state_obs():
    p = build_prompt("INSTR", {"k": 1}, "OBS")
    assert "INSTR" in p and '{"k":1}' in p and "OBS" in p and "History" not in p


# ---- runtime -------------------------------------------------------------
class Env:
    def __init__(self, n): self.n = n
    def reset(self): self.t = 0; self.actions = []; return "obs0"
    def step(self, a):
        self.actions.append(a); self.t += 1
        return f"obs{self.t}", self.t >= self.n

class Scripted:
    def __init__(self, replies): self.replies, self.prompts = list(replies), []
    def complete(self, prompt):
        self.prompts.append(prompt); return LLMResult(self.replies.pop(0))

def ok(patch, action):
    return "r\n```json\n" + json.dumps({"state_patch": patch, "action": action}) + "\n```"

SCHEMA = {"type": "object", "properties": {"count": {"type": "integer"}}, "additionalProperties": False}

def test_rollback_retry_keeps_state_and_feeds_back_error():
    llm = Scripted([ok({"count": 1}, "A"), ok({"bogus": 1}, "B"), ok({"count": 2}, "C")])
    rt = SkillStateRuntime(Skill("s", "P", SCHEMA, {"count": 0}), llm)
    env = Env(2)
    res = rt.run(env, max_steps=5)
    assert env.actions == ["A", "C"]
    assert res.final_state == {"count": 2} and res.rejected_patches == 1
    assert "was rejected" in llm.prompts[2] and '{"count":1}' in llm.prompts[2]
    assert "was rejected" not in llm.prompts[0]

def test_reasoning_never_reaches_later_prompts():
    secret = "SECRET_REASONING"
    llm = Scripted([secret + ok({}, "A"), ok({}, "B")])
    SkillStateRuntime(Skill("s", "P"), llm).run(Env(2), 5)
    assert secret not in llm.prompts[1]

def test_fallback_and_abort():
    bad = ["garbage"] * 3
    env = Env(1)
    res = SkillStateRuntime(Skill("s", "P", fallback_action="Wait"), Scripted(bad)).run(env, 5)
    assert env.actions == ["Wait"] and res.steps[0].fell_back
    res = SkillStateRuntime(Skill("s", "P"), Scripted(bad)).run(Env(1), 5)
    assert res.aborted and res.steps == []

def test_state_size_cap():
    llm = Scripted([ok({"blob": "x" * 500}, "A"), ok({}, "B")])
    res = SkillStateRuntime(Skill("s", "P"), llm, max_state_chars=100).run(Env(1), 5)
    assert res.final_state == {} and res.rejected_patches == 1
