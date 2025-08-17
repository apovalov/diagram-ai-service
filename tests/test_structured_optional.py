from __future__ import annotations

from pydantic import BaseModel

from app.core.structured import _json_schema_from_pydantic


class Demo(BaseModel):
    req: int
    opt: str | None = None  # optional


def _extract_obj_schema(schema: dict) -> dict:
    # Pydantic v2 may wrap via allOf/$ref; find the first object node
    if schema.get("type") == "object":
        return schema
    for key in ("allOf", "anyOf", "oneOf"):
        if key in schema:
            for part in schema[key]:
                if isinstance(part, dict):
                    found = _extract_obj_schema(part)
                    if found:
                        return found
    # Fallback: search any nested dicts
    for v in schema.values():
        if isinstance(v, dict):
            found = _extract_obj_schema(v)
            if found:
                return found
        elif isinstance(v, list):
            for it in v:
                if isinstance(it, dict):
                    found = _extract_obj_schema(it)
                    if found:
                        return found
    return {}


def test_optional_field_is_not_forced_required():
    schema = _json_schema_from_pydantic(Demo)
    obj = _extract_obj_schema(schema)
    assert obj, "object schema not found"

    required = set(obj.get("required", []))
    assert "req" in required, "req must be required"
    assert "opt" not in required, "opt must remain optional"
    assert obj.get("additionalProperties") is False
