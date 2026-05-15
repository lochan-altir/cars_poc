import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from car_classifier.config import OPENAI_API_KEY
from car_classifier.core.image_utils import convert_to_base64
from car_classifier.core.json_utils import parse_json_lenient
from car_classifier.core.prompt import SYSTEM_PROMPT
from car_classifier.models.schemas import (
    CarClassificationRequest,
    CarClassificationResponse,
    OpenAIModelChoice,
)
from car_classifier.routers.ollama import _normalize_for_schema

router = APIRouter()


@router.post("/openai", response_model=CarClassificationResponse)
async def classify_openai(
    image: UploadFile = File(...),
    model: OpenAIModelChoice = Form(...),
) -> CarClassificationResponse:
    try:
        image_b64, mime_type = await convert_to_base64(image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    request_payload = CarClassificationRequest(image=image_b64.encode("utf-8"), model=model.value)
    payload = {
        "model": request_payload.model,
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
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
        )
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    body = response.json()
    text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    text = str(text).strip()
    if not text:
        raise HTTPException(status_code=502, detail="Invalid JSON from OpenAI: empty response content")

    try:
        result_json = parse_json_lenient(text)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"Invalid JSON from OpenAI: {exc}")

    result_json = _normalize_for_schema(result_json, image.filename)
    try:
        return CarClassificationResponse(**result_json)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Schema validation error: {exc}")
