"""Execution state Σ and the merge operator ⊕ (paper Eq. 4).

The paper defines ⊕ as "the runtime's dictionary merge operator with
null-deletion semantics". It must be a *deep* merge: the Appendix B.3
example patches {"inventory": {"shelf_42": null}}, which under a shallow
merge would wipe the other 499 shelves. These semantics are exactly
RFC 7386 (JSON Merge Patch):

  * dict values merge recursively
  * a null value deletes the key
  * any other value (including lists) replaces wholesale
"""
from __future__ import annotations

import copy
from typing import Any

State = dict[str, Any]


def merge(state: State, patch: State) -> State:
    """Return Σ ⊕ ΔΣ without mutating either argument."""
    if not isinstance(patch, dict):
        raise TypeError("state patch must be a JSON object")
    out = copy.deepcopy(state)
    _merge_into(out, patch)
    return out


def _merge_into(target: dict, patch: dict) -> None:
    for key, value in patch.items():
        if value is None:
            target.pop(key, None)
        elif isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_into(target[key], value)
        elif isinstance(value, dict):
            # New sub-object: strip nulls so deletions of absent keys are no-ops.
            fresh: dict = {}
            _merge_into(fresh, value)
            target[key] = fresh
        else:
            target[key] = copy.deepcopy(value)
