"""Deterministic state validation (paper §3, §7: schema ownership lives in
the runtime, not the model, so a malformed patch can never corrupt Σ).

A dependency-free subset of JSON Schema is supported:
  type, properties, required, additionalProperties (bool | schema),
  items, enum, minimum, maximum.
If you already use `jsonschema`, pass `validator=` to the runtime instead.

Schemas are authored once per *domain*, not per task (paper §3.1).
"""
from __future__ import annotations

from typing import Any, Callable

_TYPES: dict[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def validate(value: Any, schema: dict, path: str = "$") -> list[str]:
    """Return a list of human-readable errors (empty list == valid)."""
    errors: list[str] = []

    expected = schema.get("type")
    if expected is not None:
        options = expected if isinstance(expected, list) else [expected]
        if not any(_TYPES[t](value) for t in options):
            return [f"{path}: expected {'|'.join(options)}, got {type(value).__name__}"]

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in {schema['enum']}")
    if "minimum" in schema and isinstance(value, (int, float)) and value < schema["minimum"]:
        errors.append(f"{path}: {value} < minimum {schema['minimum']}")
    if "maximum" in schema and isinstance(value, (int, float)) and value > schema["maximum"]:
        errors.append(f"{path}: {value} > maximum {schema['maximum']}")

    if isinstance(value, dict):
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"{path}: missing required key '{req}'")
        extra = schema.get("additionalProperties", True)
        for key, sub in value.items():
            child = f"{path}.{key}"
            if key in props:
                errors += validate(sub, props[key], child)
            elif extra is False:
                errors.append(f"{child}: key not allowed by schema")
            elif isinstance(extra, dict):
                errors += validate(sub, extra, child)

    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors += validate(item, schema["items"], f"{path}[{i}]")

    return errors
