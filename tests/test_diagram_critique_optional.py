from __future__ import annotations

from app.core.schemas import DiagramCritique
from app.core.structured import _json_schema_from_pydantic


def test_diagram_critique_accepts_done_without_critique():
    payload = {"done": True}
    # Pydantic validation of the model should pass:
    m = DiagramCritique.model_validate(payload)
    assert m.done is True
    assert m.critique is None


def test_diagram_critique_schema_allows_omitting_critique():
    # Ensure our schema does not force 'critique' into required
    schema = _json_schema_from_pydantic(DiagramCritique)

    # A very basic structural check: there should not be a top-level required containing 'critique'
    # Find object node:
    def find_obj(s):
        if isinstance(s, dict):
            if s.get("type") == "object":
                return s
            for v in s.values():
                res = find_obj(v)
                if res:
                    return res
        elif isinstance(s, list):
            for it in s:
                res = find_obj(it)
                if res:
                    return res
        return None

    obj = find_obj(schema)
    assert obj is not None
    assert "critique" not in set(obj.get("required", []))
