"""End-to-end demo of the SKILL.state runtime on the Warehouse environment.

    python examples/run_warehouse.py                 # offline scripted policy
    python examples/run_warehouse.py --faulty        # + injected bad patches
    python examples/run_warehouse.py --anthropic     # real model (needs API key)

The scripted policy sees ONLY the prompt the runtime builds (P, Σ_t, O_t),
so a perfect score shows Σ_t is a sufficient statistic for this domain.
"""
import argparse, json, os, random, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from skill_state import AnthropicLLM, LLMResult, SkillStateRuntime
from skill_state.envs import WarehouseEnv, make_warehouse_skill


def reply(reasoning, patch, action):
    body = json.dumps({"state_patch": patch, "action": action})
    return f"{reasoning}\n```json\n{body}\n```"


class ScriptedPolicy:
    def __init__(self, n_shelves):
        self.n = n_shelves

    def complete(self, prompt: str) -> LLMResult:
        state = json.loads(re.search(r"```json\n(.*?)\n```", prompt, re.S).group(1))
        obs = prompt.split("Latest Observation:", 1)[1]
        event = re.search(r"Event: (.*)", obs).group(1)
        inv = state["inventory"]
        where = {item: shelf for shelf, item in inv.items()}
        empty = [f"shelf_{i}" for i in range(self.n) if f"shelf_{i}" not in inv]

        if m := re.match(r"Shipment arrived containing (\S+)\.", event):
            item, dst = m[1], empty[0]
            text = reply(f"Store {item} on first empty shelf.", {"inventory": {dst: item}}, f"Store {item} {dst}")
        elif m := re.match(r"Customer ordered (\S+)\.", event):
            item = m[1]; src = where[item]
            text = reply(f"{item} is on {src}; ship it.", {"inventory": {src: None}}, f"Ship {item} {src}")
        elif m := re.match(r"Maintenance required on (shelf_\d+)\.", event):
            src = m[1]; item = inv[src]; dst = empty[0]
            text = reply(f"Evacuate {src}.", {"inventory": {src: None, dst: item}}, f"Move {item} {src} {dst}")
        elif m := re.match(r"ALERT: (\S+) was moved from (\S+) to (\S+)", event):
            item, src, dst = m.groups()
            text = reply("External move; sync state.", {"inventory": {src: None, dst: item}}, "Wait")
        else:
            text = reply("Nothing to do.", {}, "Wait")
        return LLMResult(text)


class FaultInjector:
    """Corrupts ~25% of first attempts to exercise validation + rollback-retry."""
    def __init__(self, inner, seed=0):
        self.inner, self.rng = inner, random.Random(seed)

    def complete(self, prompt):
        good = self.inner.complete(prompt)
        if "was rejected" in prompt or self.rng.random() > 0.25:
            return good
        return LLMResult(self.rng.choice([
            "```json\n{\"state_patch\": {\"inventory\": {}},}\n```",           # JSON slip
            reply("oops", {"inventory": ["shelf_1"]}, "Wait"),                 # type coercion
            reply("oops", {"inventory": {"shelf_999": "x"}}, "Wait"),          # invalid shelf
            reply("oops", {"inv": {}}, "Wait"),                                # schema violation
        ]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--shelves", type=int, default=20)
    ap.add_argument("--noise", type=int, default=0)
    ap.add_argument("--faulty", action="store_true")
    ap.add_argument("--anthropic", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    skill = make_warehouse_skill(a.shelves)
    env = WarehouseEnv(a.horizon, a.shelves, seed=42, noise_per_turn=a.noise,
                       drift_at=(a.horizon // 3, 2 * a.horizon // 3))
    llm = AnthropicLLM() if a.anthropic else ScriptedPolicy(a.shelves)
    if a.faulty:
        llm = FaultInjector(llm)

    log = (lambda s: print(f"[t={s['t']}] {s['action']}  (attempts={s['attempts']})")) if a.verbose else None
    res = SkillStateRuntime(skill, llm, max_retries=2, on_step=log).run(env, max_steps=a.horizon)

    sizes = [s.prompt_chars for s in res.steps]
    print(f"score            {env.score:.3f}")
    print(f"steps            {len(res.steps)}  llm_calls {res.llm_calls}  rejected {res.rejected_patches}")
    print(f"prompt chars     avg {res.avg_prompt_chars:.0f}  min {min(sizes)}  max {max(sizes)}")
    print(f"total tokens     {res.total_tokens:,}")
    print(f"state == world   {res.final_state['inventory'] == env.world}")


if __name__ == "__main__":
    main()
