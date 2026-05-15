import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from car_classifier.config import ANTHROPIC_API_KEY
from car_classifier.core.image_utils import convert_to_base64
from car_classifier.core.json_utils import parse_json_lenient
from car_classifier.core.prompt import SYSTEM_PROMPT
from car_classifier.models.schemas import (
    AnthropicModelChoice,
    CarClassificationRequest,
    CarClassificationResponse,
)
from car_classifier.routers.ollama import _normalize_for_schema

router = APIRouter()


@router.post("/anthropic", response_model=CarClassificationResponse)
async def classify_anthropic(
    image: UploadFile = File(...),
    model: AnthropicModelChoice = Form(...),
) -> CarClassificationResponse:
    try:
        image_b64, mime_type = await convert_to_base64(image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    request_payload = CarClassificationRequest(image=image_b64.encode("utf-8"), model=model.value)
    payload = {
        "model": request_payload.model,
        "max_tokens": 2000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": SYSTEM_PROMPT},
                ],
            }
        ],
    }
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
        )
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    body = response.json()

    # Anthropic can return different "content" parts; we try to extract any text-like parts.
    text_parts = []
    for part in body.get("content", []):
        if isinstance(part, dict):
            part_text = part.get("text")
            if part_text:
                text_parts.append(str(part_text))
    text = "".join(text_parts).strip()
    if not text:
        raise HTTPException(status_code=502, detail="Invalid JSON from Anthropic: empty response content")

    try:
        result_json = parse_json_lenient(text)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"Invalid JSON from Anthropic: {exc}")

    result_json = _normalize_for_schema(result_json, image.filename)
    try:
        return CarClassificationResponse(**result_json)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Schema validation error: {exc}")
