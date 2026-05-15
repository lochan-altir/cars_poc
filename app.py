"""
Single-process Streamlit app for car image classification.

    streamlit run app.py

No separate FastAPI backend needed — calls provider APIs directly.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import io
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# ---------------------------------------------------------------------------
# .env loading — check current dir, then parent
# ---------------------------------------------------------------------------
_here = Path(__file__).resolve().parent
for _candidate in (_here / ".env", _here.parent / ".env"):
    if _candidate.is_file():
        load_dotenv(dotenv_path=_candidate, override=True)
        break

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL_LLAMA4_SCOUT", "meta-llama/llama-4-scout-17b-16e-instruct")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# ---------------------------------------------------------------------------
# SYSTEM_PROMPT
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a car listing image classifier for an automotive advertising platform.

STRICT RULES:
- You only classify cars and automotive vehicles
- If the image does not contain a car, SUV, truck, van, motorcycle, or other road vehicle, return this exact JSON and nothing else:
  {"error": "not_a_vehicle", "message": "Image does not contain a car or road vehicle"}
- Return ONLY valid JSON — no markdown, no backticks, no explanation
- Every field must be filled — no empty strings, no null except activity field
- notes fields must be detailed, specific, observational sentences like a professional automotive photographer would write
- confidence_score must reflect actual certainty based on image clarity

Return exactly this schema:

{
  "image_id": "derived from filename without extension",
  "classified_at": "current ISO timestamp",
  "confidence_score": 0.0,
  "vehicle": {
    "brand": "exact manufacturer name e.g. Cadillac, BMW, Toyota",
    "model": "exact model name e.g. Escalade IQ, 3 Series, Camry",
    "category": "one of: sedan, suv, truck, coupe, hatchback, van, convertible, wagon, motorcycle, unknown",
    "trim_tier": "trim level if visible e.g. Sport, Blackout, Platinum, base",
    "exterior_color": "specific color e.g. silver_metallic, pearl_white, midnight_black",
    "view_angle": "one of: front, rear, side, front_three_quarter, rear_three_quarter, interior, aerial, unknown",
    "vehicle_count": "one of: single, multiple",
    "notes": "2-3 sentences describing specific visible details — badging, wheels, lights, distinctive features"
  },
  "location": {
    "environment_type": "one of: urban, suburban, rural, studio, highway, off_road, unknown",
    "terrain": "specific terrain e.g. desert_gravel, wet_asphalt, mountain_road, studio_floor",
    "road_type": "e.g. paved, gravel, dirt, none",
    "region_inference": "inferred geographic region e.g. southwest_us, europe, southeast_asia, unknown",
    "background_feature": "dominant background element e.g. mountains, city_skyline, trees, seamless_white",
    "is_studio": true or false,
    "notes": "2-3 sentences describing the shooting location and environment context"
  },
  "scene": {
    "time_of_day": "one of: day, golden_hour, dusk, night, overcast, unknown",
    "season": "one of: spring, summer, autumn, winter, indeterminate",
    "weather": "one of: clear, cloudy, rain, snow, fog, unknown",
    "sky_prominence": "one of: none, minimal, moderate, dramatic",
    "notes": "2-3 sentences describing lighting conditions, sky, atmosphere"
  },
  "visual": {
    "color_palette": "e.g. cool_with_warm_accent, monochromatic_dark, warm_earthy, high_contrast",
    "dominant_hues": ["list of 2-4 color names present in image"],
    "saturation": "one of: muted, neutral, vivid",
    "brightness": "one of: dark_moody, medium, bright_airy",
    "shot_type": "e.g. medium_establishing, hero_shot, detail_close_up, wide_environmental",
    "depth_of_field": "one of: shallow, medium, deep_sharp",
    "motion_energy": "one of: static, implied, dynamic",
    "production_type": "one of: studio, cgi, live_photo, composite",
    "notes": "2-3 sentences describing cinematography, composition, visual treatment"
  },
  "creative": {
    "lifestyle_segment": "one of: luxury_premium, family, adventure, sport, urban, budget, unknown",
    "emotional_tone": "e.g. aspirational_premium, adventurous, safe_reliable, sporty_aggressive, serene",
    "campaign_theme": "e.g. luxury_ev_technology, family_adventure, urban_mobility, off_road_capability",
    "people_present": true or false,
    "activity": "describe activity if people present, else null",
    "compositional_technique": "e.g. foreground_frame, rule_of_thirds, symmetrical, leading_lines",
    "notes": "2-3 sentences about creative intent, target audience, advertising context"
  },
  "format": {
    "ad_format": "one of: static, animated, 360, video_still",
    "aspect_ratio": "e.g. 16:9, 4:3, 1:1, 2:1, 9:16",
    "platform_target": "one of: web_gallery, social, display_ad, print, unknown",
    "orientation": "one of: landscape, portrait, square",
    "copy_present": true or false,
    "logo_badge_visible": "brand name if logo visible e.g. cadillac_crest, bmw_roundel, none"
  },
  "raw_labels": ["flat list of 10-20 single descriptive tags about everything visible in the image"]
}"""

# ═══════════════════════════════════════════════════════════════════════════
# JSON parsing helpers
# ═══════════════════════════════════════════════════════════════════════════


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        first_newline = t.find("\n")
        if first_newline != -1:
            t = t[first_newline + 1:]
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


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
                return text[start: i + 1]
    return None


def _escape_invalid_backslashes(candidate: str) -> str:
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


def _parse_json_lenient(text: str) -> dict:
    if text is None or not str(text).strip():
        raise ValueError("Empty model response")

    cleaned = _strip_code_fences(str(text))
    candidates: list[str] = [cleaned]

    extracted = _extract_first_json_object(cleaned)
    if extracted and extracted not in candidates:
        candidates.append(extracted)

    errors: list[str] = []
    for candidate in candidates:
        variants = [candidate]
        escaped_variant = _escape_invalid_backslashes(candidate)
        if escaped_variant != candidate:
            variants.append(escaped_variant)

        for variant in variants:
            try:
                parsed = json.loads(variant)
                if isinstance(parsed, dict):
                    return parsed
                errors.append(f"Expected object, got {type(parsed).__name__}")
                continue
            except json.JSONDecodeError as exc:
                errors.append(str(exc))

            try:
                literal = ast.literal_eval(variant)
                if isinstance(literal, dict):
                    return literal
                errors.append(f"Expected dict literal, got {type(literal).__name__}")
            except Exception as literal_exc:
                errors.append(str(literal_exc))

    raise ValueError(f"Invalid JSON: {' | '.join(errors[-3:])}")


# ═══════════════════════════════════════════════════════════════════════════
# Normalization helpers
# ═══════════════════════════════════════════════════════════════════════════


def _normalize_keys(value):
    if isinstance(value, dict):
        return {
            (k.replace("\\_", "_").strip() if isinstance(k, str) else k): _normalize_keys(v)
            for k, v in value.items()
        }
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
        parts = [str(it).strip() for it in value if it is not None and str(it).strip()]
        return " ".join(parts) if parts else None
    if isinstance(value, dict):
        text_parts = []
        for k, v in value.items():
            if v is None:
                continue
            piece = str(v).strip()
            if piece:
                text_parts.append(f"{k}: {piece}")
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
    return float(match.group(0)) if match else 0.0


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


def _normalize_section(
    payload: dict,
    section_name: str,
    string_fields: list[str],
    bool_fields: list[str],
    list_fields: list[str],
) -> None:
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


def _normalize_for_schema(payload: dict, source_filename: Optional[str]) -> dict:
    payload = _normalize_keys(payload)

    if not payload.get("image_id"):
        payload["image_id"] = Path(source_filename).stem if source_filename else "unknown_image"
    payload["image_id"] = _coerce_optional_string(payload.get("image_id")) or "unknown_image"

    if not payload.get("source_file"):
        payload["source_file"] = source_filename or "unknown_file"
    payload["source_file"] = _coerce_optional_string(payload.get("source_file")) or "unknown_file"

    if not payload.get("classified_at"):
        payload["classified_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        payload["classified_at"] = _coerce_optional_string(payload.get("classified_at"))

    payload["confidence_score"] = _coerce_confidence_score(payload.get("confidence_score"))
    payload["raw_labels"] = _coerce_optional_string_list(payload.get("raw_labels"))

    _normalize_section(
        payload, "vehicle",
        ["brand", "model", "category", "trim_tier", "exterior_color", "view_angle", "vehicle_count", "notes"],
        [], [],
    )
    _normalize_section(
        payload, "location",
        ["environment_type", "terrain", "road_type", "region_inference", "background_feature", "notes"],
        ["is_studio"], [],
    )
    _normalize_section(payload, "scene", ["time_of_day", "season", "weather", "sky_prominence", "notes"], [], [])
    _normalize_section(
        payload, "visual",
        ["color_palette", "saturation", "brightness", "shot_type", "depth_of_field", "motion_energy", "production_type", "notes"],
        [], ["dominant_hues"],
    )
    _normalize_section(
        payload, "creative",
        ["lifestyle_segment", "emotional_tone", "campaign_theme", "activity", "compositional_technique", "notes"],
        ["people_present"], [],
    )
    _normalize_section(
        payload, "format",
        ["ad_format", "aspect_ratio", "platform_target", "orientation", "logo_badge_visible"],
        ["copy_present"], [],
    )
    return payload


# ═══════════════════════════════════════════════════════════════════════════
# Image helpers
# ═══════════════════════════════════════════════════════════════════════════


def _pil_to_base64(image: Image.Image) -> tuple[str, str]:
    buf = io.BytesIO()
    if image.mode in ("RGBA", "LA", "PA"):
        fmt, mime = "PNG", "image/png"
    else:
        image = image.convert("RGB")
        fmt, mime = "JPEG", "image/jpeg"
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii"), mime


# ═══════════════════════════════════════════════════════════════════════════
# Provider API calls
# ═══════════════════════════════════════════════════════════════════════════


async def _call_groq(image_b64: str, mime_type: str, model: str) -> dict:
    payload = {
        "model": model or GROQ_MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                ],
            }
        ],
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{GROQ_BASE_URL}/chat/completions", headers=headers, json=payload)
    if resp.status_code >= 400:
        raise RuntimeError(f"Groq API {resp.status_code}: {resp.text}")
    text = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    if not str(text).strip():
        raise RuntimeError("Groq returned empty response")
    return _parse_json_lenient(text)


async def _call_openai(image_b64: str, mime_type: str, model: str) -> dict:
    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                ],
            }
        ],
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenAI API {resp.status_code}: {resp.text}")
    text = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    if not str(text).strip():
        raise RuntimeError("OpenAI returned empty response")
    return _parse_json_lenient(text)


async def _call_anthropic(image_b64: str, mime_type: str, model: str) -> dict:
    payload = {
        "model": model,
        "max_tokens": 2000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": mime_type, "data": image_b64},
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
        resp = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
    if resp.status_code >= 400:
        raise RuntimeError(f"Anthropic API {resp.status_code}: {resp.text}")
    text_parts = []
    for part in resp.json().get("content", []):
        if isinstance(part, dict) and part.get("text"):
            text_parts.append(str(part["text"]))
    text = "".join(text_parts).strip()
    if not text:
        raise RuntimeError("Anthropic returned empty response")
    return _parse_json_lenient(text)


async def _call_ollama(image_b64: str, mime_type: str, model: str) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": SYSTEM_PROMPT,
                "images": [image_b64],
            }
        ],
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
    if resp.status_code >= 400:
        raise RuntimeError(f"Ollama API {resp.status_code}: {resp.text}")
    text = resp.json().get("message", {}).get("content", "")
    if not str(text).strip():
        raise RuntimeError("Ollama returned empty response")
    return _parse_json_lenient(text)


# ═══════════════════════════════════════════════════════════════════════════
# Provider registry
# ═══════════════════════════════════════════════════════════════════════════

PROVIDERS: dict[str, dict] = {
    "groq": {
        "models": [],
        "call": _call_groq,
        "needs_key": "GROQ_API_KEY",
    },
    "ollama": {
        "models": ["llava:7b", "llava:13b", "llava-phi3"],
        "call": _call_ollama,
        "needs_key": "OLLAMA_BASE_URL",
    },
    "openai": {
        "models": ["gpt-4o", "gpt-4o-mini", "o4-mini"],
        "call": _call_openai,
        "needs_key": "OPENAI_API_KEY",
    },
    "anthropic": {
        "models": ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-haiku-4-5-20251001"],
        "call": _call_anthropic,
        "needs_key": "ANTHROPIC_API_KEY",
    },
}

_KEY_MAP: dict[str, str] = {
    "GROQ_API_KEY": GROQ_API_KEY,
    "OLLAMA_BASE_URL": OLLAMA_BASE_URL,
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
}


def _key_status(env_var: str) -> str:
    return "\u2705" if _KEY_MAP.get(env_var) else "\u274c"


# ═══════════════════════════════════════════════════════════════════════════
# Async helper — Streamlit doesn't natively run async callbacks
# ═══════════════════════════════════════════════════════════════════════════


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# ═══════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Car Image Classifier", page_icon="\U0001F697", layout="wide")

st.title("\U0001F697 Car Image Classifier")
st.caption("Upload a car image \u2192 get structured JSON classification via LLM. No backend needed.")

# --- Sidebar: provider + model ---
with st.sidebar:
    st.header("Provider")

    provider_options = list(PROVIDERS.keys())
    provider_labels = {
        p: f"{_key_status(PROVIDERS[p]['needs_key'])} {p}" for p in provider_options
    }

    selected_provider = st.selectbox(
        "Choose provider",
        options=provider_options,
        format_func=lambda p: provider_labels[p],
        index=0,
    )

    cfg = PROVIDERS[selected_provider]

    selected_model = None
    if cfg["models"]:
        selected_model = st.selectbox("Model", options=cfg["models"], index=0)
    else:
        st.markdown(f"**Model:** `{GROQ_MODEL}`")

    st.divider()
    st.markdown("**API key status**")
    for pname, pcfg in PROVIDERS.items():
        st.text(f"  {_key_status(pcfg['needs_key'])} {pname}")

# --- Main area ---
uploaded_file = st.file_uploader("Upload a car image", type=["png", "jpg", "jpeg", "webp", "gif"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption=uploaded_file.name, width=300)

classify_pressed = st.button("\U0001F50D Classify", type="primary", use_container_width=True)

if classify_pressed:
    if not uploaded_file:
        st.error("Please upload an image first.")
    else:
        api_key = _KEY_MAP.get(cfg["needs_key"], "")
        if not api_key:
            st.error(f"Missing API key: set **{cfg['needs_key']}** in your `.env` file.")
        elif cfg["models"] and not selected_model:
            st.error(f"Please select a model for **{selected_provider}**.")
        else:
            image = Image.open(uploaded_file)
            image_b64, mime_type = _pil_to_base64(image)
            effective_model = selected_model or ""

            with st.spinner(f"Classifying via **{selected_provider}**" + (f" / {effective_model}" if effective_model else "") + " ..."):
                try:
                    raw = _run_async(cfg["call"](image_b64, mime_type, effective_model))
                    normalized = _normalize_for_schema(raw, uploaded_file.name)
                    pretty = json.dumps(normalized, indent=2, ensure_ascii=False)
                    st.success(f"Done — **{selected_provider}**" + (f" / {effective_model}" if effective_model else ""))
                    st.code(pretty, language="json")
                except Exception as exc:
                    st.error(f"Provider error ({selected_provider}): {exc}")

