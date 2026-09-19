"""SkillExecBench Environment 1: Warehouse Management (paper Appendix B.1).

Events are pre-generated from a seeded RNG (Algorithm 2) so every runtime
sees the same event sequence. One deviation from the paper, for fairness:
"Maintenance required on [shelf]" is generated per *item* and resolved to
whichever shelf the agent actually put that item on, so a different (but
valid) storage choice doesn't invalidate later events.

Options:
  noise_per_turn  -- Appendix C background telemetry (Experiment 2)
  drift_at        -- steps where an external actor silently moves an item and
                     only a corrective alert reveals it (Experiment 3)
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..runtime import Skill

INSTRUCTIONS = """You are a warehouse controller managing shelves shelf_0 .. shelf_{last}.
Each shelf holds at most one item. Handle exactly one event per turn:
- "Shipment arrived containing <item>"  -> Store <item> <empty_shelf>
- "Customer ordered <item>"             -> Ship <item> <shelf_holding_it>
- "Maintenance required on <shelf>"     -> Move <item> <shelf> <empty_shelf>
- "ALERT: <item> was moved from <a> to <b>" -> Wait (just record the change)
- nothing to do                         -> Wait
Ignore anything under BACKGROUND TELEMETRY.
State schema: {{"inventory": {{"<shelf_id>": "<item_id>"}}}}; shelves not listed are empty.
Record every change to inventory in your state patch (set a shelf to null when it empties)."""

SCHEMA = {
    "type": "object",
    "required": ["inventory"],
    "additionalProperties": False,
    "properties": {
        "inventory": {"type": "object", "additionalProperties": {"type": "string"}},
    },
}


def make_skill(n_shelves: int = 20) -> Skill:
    valid = {f"shelf_{i}" for i in range(n_shelves)}

    def shelves_exist(state):
        bad = [k for k in state.get("inventory", {}) if k not in valid]
        return [f"unknown shelf ids: {bad[:5]}"] if bad else []

    def no_duplicate_items(state):
        items = list(state.get("inventory", {}).values())
        dupes = {i for i in items if items.count(i) > 1}
        return [f"items on multiple shelves: {sorted(dupes)}"] if dupes else []

    return Skill(
        name="warehouse",
        instructions=INSTRUCTIONS.format(last=n_shelves - 1),
        schema=SCHEMA,
        initial_state={"inventory": {}},
        checks=[shelves_exist, no_duplicate_items],
        fallback_action="Wait",
    )


@dataclass
class _Event:
    kind: str          # receive | order | maintenance | drift
    item: str


class WarehouseEnv:
    def __init__(self, horizon: int, n_shelves: int = 20, seed: int = 42,
                 noise_per_turn: int = 0, drift_at: tuple[int, ...] = ()):
        self.horizon, self.n_shelves, self.seed = horizon, n_shelves, seed
        self.noise_per_turn, self.drift_at = noise_per_turn, set(drift_at)

    # Algorithm 2 -------------------------------------------------------------
    def _generate(self) -> list[_Event]:
        rng = random.Random(self.seed)
        stocked: list[str] = []
        events, next_id = [], 0
        for i in range(self.horizon):
            has_room = len(stocked) < self.n_shelves   # Move/drift need an empty shelf
            if i in self.drift_at and stocked and has_room:
                events.append(_Event("drift", rng.choice(stocked)))
                continue
            kinds = ["receive", "maintenance"] if has_room else []
            if stocked:
                kinds.append("order")
            else:
                kinds = ["receive"]
            kind = rng.choice(kinds)
            if kind == "receive":
                item = f"item_{next_id}"; next_id += 1
                stocked.append(item)
            else:
                item = rng.choice(stocked)
                if kind == "order":
                    stocked.remove(item)
            events.append(_Event(kind, item))
        return events

    # Environment protocol ----------------------------------------------------
    def reset(self) -> str:
        self.rng = random.Random(self.seed + 1)
        self.world: dict[str, str] = {}          # true shelf -> item
        self.events = self._generate()
        self.i = 0
        self.actionable = sum(e.kind != "drift" for e in self.events)
        self.successes = 0
        return self._render(self._emit())

    def step(self, action: str) -> tuple[str, bool]:
        ok, msg = self._apply(action, self.current)
        if ok and self.current.kind != "drift":
            self.successes += 1
        self.i += 1
        if self.i >= len(self.events):
            return f"{msg}\nAll events processed.", True
        return self._render(self._emit(), prefix=msg), False

    @property
    def score(self) -> float:
        return self.successes / max(self.actionable, 1)

    # internals ---------------------------------------------------------------
    def _shelf_of(self, item):
        return next((s for s, it in self.world.items() if it == item), None)

    def _empty(self):
        return [f"shelf_{i}" for i in range(self.n_shelves) if f"shelf_{i}" not in self.world]

    def _emit(self) -> str:
        ev = self.current = self.events[self.i]
        if ev.kind == "receive":
            return f"Shipment arrived containing {ev.item}."
        if ev.kind == "order":
            return f"Customer ordered {ev.item}."
        src = self._shelf_of(ev.item)
        if src is None:  # agent lost the item earlier; event is unsatisfiable
            return f"Maintenance required on a shelf holding {ev.item}."
        if ev.kind == "maintenance":
            return f"Maintenance required on {src}."
        dst = self.rng.choice(self._empty())      # silent external move
        self.world[dst] = self.world.pop(src)
        return f"ALERT: {ev.item} was moved from {src} to {dst} by an external operator."

    def _render(self, event: str, prefix: str = "") -> str:
        parts = [prefix] if prefix else []
        parts.append(f"Event: {event}")
        if self.noise_per_turn:
            parts.append("--- BACKGROUND TELEMETRY ---")
            parts += [self._noise() for _ in range(self.noise_per_turn)]
        return "\n".join(parts)

    def _noise(self) -> str:
        r = self.rng
        return r.choice([
            f"Battery: {r.randint(10, 99)}%, Temperature: {r.randint(30, 60)}C, "
            f"CPU Load: {r.randint(5, 99)}%, Speed: {r.uniform(0, 2):.1f} m/s",
            f"[Sensor] Humidity: {r.randint(30, 60)}%, Temp: {r.uniform(18, 26):.1f}C, "
            f"CO2: {r.randint(400, 800)} ppm",
            f"[Camera OCR] {r.choice(['Forklift parked.', 'Worker entered Zone A.', 'Safety Vest Detected.'])}",
        ])

    def _apply(self, action: str, ev: _Event) -> tuple[bool, str]:
        parts = action.split()
        verb = parts[0] if parts else ""
        if verb == "Wait":
            return ev.kind == "drift", "Result: Waited."
        if verb == "Store" and len(parts) == 3:
            item, shelf = parts[1:]
            if shelf in self.world or not shelf.startswith("shelf_"):
                return False, f"Error: {shelf} is occupied or invalid."
            if ev.kind != "receive" or item != ev.item:
                return False, f"Error: no incoming shipment of {item}."
            self.world[shelf] = item
            return True, f"Success: Stored {item} on {shelf}."
        if verb == "Ship" and len(parts) == 3:
            item, shelf = parts[1:]
            if self.world.get(shelf) != item:
                return False, f"Error: {item} is not on {shelf}."
            if ev.kind != "order" or item != ev.item:
                return False, f"Error: {item} was not ordered."
            del self.world[shelf]
            return True, f"Success: Shipped {item} from {shelf}."
        if verb == "Move" and len(parts) == 4:
            item, src, dst = parts[1:]
            if self.world.get(src) != item or dst in self.world:
                return False, f"Error: cannot move {item} from {src} to {dst}."
            self.world[dst] = self.world.pop(src)
            correct = ev.kind == "maintenance" and ev.item == item
            return correct, f"Success: Moved {item} from {src} to {dst}."
        return False, f"Error: unrecognized action '{action}'."
