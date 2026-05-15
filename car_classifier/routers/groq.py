import ast
import json

import httpx
from fastapi import APIRouter, File, HTTPException, UploadFile

from car_classifier.config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL_LLAMA4_SCOUT
from car_classifier.core.image_utils import convert_to_base64
from car_classifier.core.json_utils import parse_json_lenient
from car_classifier.core.prompt import SYSTEM_PROMPT
from car_classifier.models.schemas import CarClassificationResponse
from car_classifier.routers.ollama import _normalize_for_schema

router = APIRouter()

# Groq endpoint details
# - Route: POST /classify/groq
# - Provider API: Groq OpenAI-compatible chat completions
#   (calls: {GROQ_BASE_URL}/chat/completions with Authorization: Bearer <GROQ_API_KEY>)
# - Model: meta-llama/llama-4-scout-17b-16e-instruct
# - Input: uploaded image + SYSTEM_PROMPT (see car_classifier/core/prompt.py)
# - Output: JSON parsed from model response, then normalized to CarClassificationResponse


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    return cleaned.strip()


def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
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
                return text[start : idx + 1]
    return ""


def _parse_groq_json(text: str) -> dict:
    normalized = _strip_code_fences(text)
    if not normalized:
        raise HTTPException(status_code=502, detail="Invalid JSON from Groq: empty response")

    candidates = [normalized]
    extracted = _extract_first_json_object(normalized)
    if extracted and extracted not in candidates:
        candidates.append(extracted)

    errors: list[str] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
            errors.append(f"Expected object JSON, got {type(parsed).__name__}")
            continue
        except json.JSONDecodeError as exc:
            errors.append(str(exc))

        try:
            parsed_literal = ast.literal_eval(candidate)
            if isinstance(parsed_literal, dict):
                return parsed_literal
            errors.append(f"Expected object literal, got {type(parsed_literal).__name__}")
        except Exception as exc:
            errors.append(str(exc))

    raise HTTPException(status_code=502, detail=f"Invalid JSON from Groq: {' | '.join(errors[-3:])}")


@router.post("/groq", response_model=CarClassificationResponse)
async def classify_groq(
    image: UploadFile = File(...),
) -> CarClassificationResponse:
    """
    Classify a car image using Groq's OpenAI-compatible API.

    Uses:
    - Model: `meta-llama/llama-4-scout-17b-16e-instruct` (override via `.env` -> GROQ_MODEL_LLAMA4_SCOUT)
    - Endpoint: `{GROQ_BASE_URL}/chat/completions` (override via `.env` -> GROQ_BASE_URL)
    - Expected response format: JSON matching `CarClassificationResponse` schema.
    """
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is missing. Add it to your .env file.",
        )
    try:
        image_b64, mime_type = await convert_to_base64(image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    payload = {
        "model": GROQ_MODEL_LLAMA4_SCOUT,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                    },
                ],
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{GROQ_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    body = response.json()
    text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    text = str(text).strip()
    if not text:
        raise HTTPException(status_code=502, detail="Invalid JSON from Groq: empty response content")
    try:
        result_json = parse_json_lenient(text)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"Invalid JSON from Groq: {exc}")
    result_json = _normalize_for_schema(result_json, image.filename)
    try:
        return CarClassificationResponse(**result_json)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Schema validation error: {exc}")
