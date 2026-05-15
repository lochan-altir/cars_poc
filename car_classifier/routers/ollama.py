import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ollama
from fastapi import APIRouter, File, HTTPException, UploadFile

from car_classifier.config import OLLAMA_BASE_URL, OLLAMA_MODEL_LLAVA_7B
from car_classifier.core.image_utils import convert_to_base64
from car_classifier.core.prompt import SYSTEM_PROMPT
from car_classifier.models.schemas import CarClassificationRequest, CarClassificationResponse

router = APIRouter()
ollama_client = ollama.AsyncClient(host=OLLAMA_BASE_URL)


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        first_newline = t.find("\n")
        if first_newline != -1:
            t = t[first_newline + 1 :]
        if t.endswith("```"):
            t = t[: -3]
    return t.strip()


def _escape_invalid_backslashes_in_json(candidate: str) -> str:
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


def _extract_first_json_object(text: str) -> Optional[str]:
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


def _normalize_keys(value):
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            if isinstance(key, str):
                fixed_key = key.replace("\\_", "_").strip()
            else:
                fixed_key = key
            normalized[fixed_key] = _normalize_keys(item)
        return normalized
    if isinstance(value, list):
        return [_normalize_keys(item) for item in value]
    return value


def _coerce_optional_string(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if item is not None and str(item).strip()]
        return " ".join(parts) if parts else None
    if isinstance(value, dict):
        text_parts = []
        for key, item in value.items():
            if item is None:
                continue
            piece = str(item).strip()
            if piece:
                text_parts.append(f"{key}: {piece}")
        return " | ".join(text_parts) if text_parts else None
    cleaned = str(value).strip()
    return cleaned or None


def _coerce_optional_bool(value) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1", "present"}:
        return True
    if text in {"false", "no", "n", "0", "absent"}:
        return False
    return None


def _coerce_confidence_score(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return 0.0
    match = re.search(r"-?\d+(\.\d+)?", text)
    if not match:
        return 0.0
    return float(match.group(0))


def _coerce_optional_string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            normalized = _coerce_optional_string(item)
            if normalized:
                out.append(normalized)
        return out
    normalized = _coerce_optional_string(value)
    if not normalized:
        return []
    if "," in normalized:
        parts = [chunk.strip() for chunk in normalized.split(",") if chunk.strip()]
        return parts or [normalized]
    return [normalized]


def _normalize_section(payload: dict, section_name: str, string_fields: list[str], bool_fields: list[str], list_fields: list[str]) -> None:
    section = payload.get(section_name)
    if section is None:
        section = {}
    elif not isinstance(section, dict):
        section = {"notes": _coerce_optional_string(section)}

    for field in string_fields:
        if field in section:
            section[field] = _coerce_optional_string(section.get(field))
    for field in bool_fields:
        if field in section:
            section[field] = _coerce_optional_bool(section.get(field))
    for field in list_fields:
        if field in section:
            section[field] = _coerce_optional_string_list(section.get(field))

    payload[section_name] = section


def _parse_ollama_json(generated_text: str) -> dict:
    if not generated_text.strip():
        raise HTTPException(status_code=502, detail="Invalid JSON from Ollama: empty response")

    candidates = [generated_text]
    json_obj = _extract_first_json_object(generated_text)
    if json_obj and json_obj not in candidates:
        candidates.append(json_obj)

    errors: list[str] = []
    for candidate in candidates:
        variants = [candidate]
        escaped_variant = _escape_invalid_backslashes_in_json(candidate)
        if escaped_variant != candidate:
            variants.append(escaped_variant)

        for variant in variants:
            try:
                parsed = json.loads(variant)
                if isinstance(parsed, dict):
                    return parsed
                errors.append(f"Expected object JSON, got {type(parsed).__name__}")
            except json.JSONDecodeError as exc:
                errors.append(str(exc))
                try:
                    literal = ast.literal_eval(variant)
                    if isinstance(literal, dict):
                        return literal
                    errors.append(f"Expected object literal, got {type(literal).__name__}")
                except Exception as literal_exc:
                    errors.append(str(literal_exc))

    raise HTTPException(status_code=502, detail=f"Invalid JSON from Ollama: {' | '.join(errors[-3:])}")


def _normalize_for_schema(payload: dict, source_filename: Optional[str]) -> dict:
    payload = _normalize_keys(payload)
    if not payload.get("image_id"):
        if source_filename:
            payload["image_id"] = Path(source_filename).stem
        else:
            payload["image_id"] = "unknown_image"
    payload["image_id"] = _coerce_optional_string(payload.get("image_id")) or "unknown_image"
    if not payload.get("source_file"):
        payload["source_file"] = source_filename or "unknown_file"
    payload["source_file"] = _coerce_optional_string(payload.get("source_file")) or "unknown_file"
    if not payload.get("classified_at"):
        payload["classified_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        payload["classified_at"] = _coerce_optional_string(payload.get("classified_at"))
    if payload.get("confidence_score") is None:
        payload["confidence_score"] = 0.0
    payload["confidence_score"] = _coerce_confidence_score(payload.get("confidence_score"))
    if payload.get("raw_labels") is None:
        payload["raw_labels"] = []
    payload["raw_labels"] = _coerce_optional_string_list(payload.get("raw_labels"))

    _normalize_section(
        payload,
        "vehicle",
        [
            "brand",
            "model",
            "category",
            "trim_tier",
            "exterior_color",
            "view_angle",
            "vehicle_count",
            "notes",
        ],
        [],
        [],
    )
    _normalize_section(
        payload,
        "location",
        ["environment_type", "terrain", "road_type", "region_inference", "background_feature", "notes"],
        ["is_studio"],
        [],
    )
    _normalize_section(payload, "scene", ["time_of_day", "season", "weather", "sky_prominence", "notes"], [], [])
    _normalize_section(
        payload,
        "visual",
        [
            "color_palette",
            "saturation",
            "brightness",
            "shot_type",
            "depth_of_field",
            "motion_energy",
            "production_type",
            "notes",
        ],
        [],
        ["dominant_hues"],
    )
    _normalize_section(
        payload,
        "creative",
        [
            "lifestyle_segment",
            "emotional_tone",
            "campaign_theme",
            "activity",
            "compositional_technique",
            "notes",
        ],
        ["people_present"],
        [],
    )
    _normalize_section(
        payload,
        "format",
        ["ad_format", "aspect_ratio", "platform_target", "orientation", "logo_badge_visible"],
        ["copy_present"],
        [],
    )
    return payload


@router.post("/ollama", response_model=CarClassificationResponse)
async def classify_ollama(
    image: UploadFile = File(...),
) -> CarClassificationResponse:
    try:
        await image.seek(0)
        image_bytes = await image.read()
        if not image_bytes:
            raise ValueError("Uploaded file is empty")
        CarClassificationRequest(image=image_bytes, model=None)
        await image.seek(0)
        await convert_to_base64(image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    selected_model = OLLAMA_MODEL_LLAVA_7B
    preferred_models = [
        OLLAMA_MODEL_LLAVA_7B,
        "llava:latest",
        "llava:13b",
        "llava-phi3:latest",
        "llava-phi3",
    ]
    try:
        response = await ollama_client.chat(
            model=selected_model,
            messages=[
                {
                    "role": "user",
                    "content": SYSTEM_PROMPT,
                    "images": [image_bytes],
                }
            ],
        )
    except ollama.ResponseError as exc:
        if exc.status_code != 404:
            raise HTTPException(
                status_code=500,
                detail=f"Ollama inference error at {OLLAMA_BASE_URL}: {exc}",
            )
        try:
            model_list = await ollama_client.list()
            available_models = [
                model.get("model") or model.get("name")
                for model in model_list.get("models", [])
                if model.get("model") or model.get("name")
            ]
        except Exception as list_exc:
            raise HTTPException(
                status_code=500,
                detail=f"Ollama model listing failed at {OLLAMA_BASE_URL}: {list_exc}",
            )
        fallback_model = next((model for model in preferred_models if model in available_models), None)
        if fallback_model is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Ollama model '{selected_model}' not found. "
                    f"Available models: {available_models or 'none'}. "
                    "Run: ollama pull llava:7b"
                ),
            )
        try:
            response = await ollama_client.chat(
                model=fallback_model,
                messages=[
                    {
                        "role": "user",
                        "content": SYSTEM_PROMPT,
                        "images": [image_bytes],
                    }
                ],
            )
        except Exception as fallback_exc:
            raise HTTPException(
                status_code=500,
                detail=f"Ollama fallback inference error at {OLLAMA_BASE_URL}: {fallback_exc}",
            )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Ollama inference error at {OLLAMA_BASE_URL}: {exc}",
        )
    generated_text = _strip_code_fences(response.get("message", {}).get("content", ""))
    result_json = _parse_ollama_json(generated_text)
    result_json = _normalize_for_schema(result_json, image.filename)
    try:
        return CarClassificationResponse(**result_json)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Schema validation error: {exc}")

