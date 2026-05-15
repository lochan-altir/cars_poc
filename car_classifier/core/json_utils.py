import ast
import json
from typing import Optional


def strip_code_fences(text: str) -> str:
    """
    Remove surrounding markdown code fences like ```json ... ```.
    """
    t = text.strip()
    if t.startswith("```"):
        first_newline = t.find("\n")
        if first_newline != -1:
            t = t[first_newline + 1 :]
        if t.endswith("```"):
            t = t[: -3]
    return t.strip()


def extract_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def escape_invalid_backslashes_in_json(candidate: str) -> str:
    """
    Best-effort fix for model outputs like:
      {"key":"bad\escape"}
    that break json.loads with `Invalid \escape`.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    for ch in candidate:
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
                escaped = False
            continue

        if escaped:
            out.append(ch)
            escaped = False
            continue

        if ch == "\\":
            out.append("\\")
            escaped = True
            continue

        if ch == '"':
            out.append(ch)
            in_string = False
            continue

        out.append(ch)

    fixed: list[str] = []
    in_string = False
    i = 0
    while i < len(out):
        ch = out[i]
        if ch == '"' and (i == 0 or out[i - 1] != "\\"):
            in_string = not in_string
            fixed.append(ch)
            i += 1
            continue
        if in_string and ch == "\\" and i + 1 < len(out):
            nxt = out[i + 1]
            if nxt not in ['"', "\\", "/", "b", "f", "n", "r", "t", "u"]:
                fixed.append("\\")
            fixed.append("\\")
            i += 1
            continue
        fixed.append(ch)
        i += 1
    return "".join(fixed)


def parse_json_lenient(text: str) -> dict:
    """
    Parse JSON from LLM output that may contain:
    - markdown code fences
    - leading/trailing non-JSON text
    - invalid backslash escapes
    - python dict-ish literals (fallback)
    """
    if text is None or not str(text).strip():
        raise ValueError("Empty model response")

    cleaned = strip_code_fences(str(text))
    candidates: list[str] = [cleaned]

    extracted = extract_first_json_object(cleaned)
    if extracted and extracted not in candidates:
        candidates.append(extracted)

    errors: list[str] = []
    for candidate in candidates:
        variants = [candidate]
        escaped_variant = escape_invalid_backslashes_in_json(candidate)
        if escaped_variant != candidate:
            variants.append(escaped_variant)

        for variant in variants:
            try:
                parsed = json.loads(variant)
                if isinstance(parsed, dict):
                    return parsed
                errors.append(f"Expected object JSON, got {type(parsed).__name__}")
                continue
            except json.JSONDecodeError as exc:
                errors.append(str(exc))

            try:
                literal = ast.literal_eval(variant)
                if isinstance(literal, dict):
                    return literal
                errors.append(f"Expected object literal, got {type(literal).__name__}")
            except Exception as literal_exc:
                errors.append(str(literal_exc))

    raise ValueError(f"Invalid JSON: {' | '.join(errors[-3:])}")

