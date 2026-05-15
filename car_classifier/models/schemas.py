from datetime import datetime
from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, BeforeValidator

from car_classifier.config import (
    ANTHROPIC_MODEL_HAIKU,
    ANTHROPIC_MODEL_OPUS,
    ANTHROPIC_MODEL_SONNET,
    OPENAI_MODEL_GPT4O,
    OPENAI_MODEL_GPT4O_MINI,
    OPENAI_MODEL_O4_MINI,
)


def _coerce_notes_field(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if x is not None and str(x).strip()]
        return " ".join(parts) if parts else None
    return str(value)


OptionalNotesStr = Annotated[Optional[str], BeforeValidator(_coerce_notes_field)]


class AnthropicModelChoice(str, Enum):
    claude_sonnet_4_20250514 = ANTHROPIC_MODEL_SONNET
    claude_opus_4_20250514 = ANTHROPIC_MODEL_OPUS
    claude_haiku_4_5_20251001 = ANTHROPIC_MODEL_HAIKU


class OpenAIModelChoice(str, Enum):
    gpt_4o = OPENAI_MODEL_GPT4O
    gpt_4o_mini = OPENAI_MODEL_GPT4O_MINI
    o4_mini = OPENAI_MODEL_O4_MINI


class CarClassificationRequest(BaseModel):
    image: bytes
    model: Optional[str] = None


class VehicleSchema(BaseModel):
    brand: Optional[str] = None
    model: Optional[str] = None
    category: Optional[str] = None
    trim_tier: Optional[str] = None
    exterior_color: Optional[str] = None
    view_angle: Optional[str] = None
    vehicle_count: Optional[str] = None
    notes: OptionalNotesStr = None


class LocationSchema(BaseModel):
    environment_type: Optional[str] = None
    terrain: Optional[str] = None
    road_type: Optional[str] = None
    region_inference: Optional[str] = None
    background_feature: Optional[str] = None
    is_studio: Optional[bool] = None
    notes: OptionalNotesStr = None


class SceneSchema(BaseModel):
    time_of_day: Optional[str] = None
    season: Optional[str] = None
    weather: Optional[str] = None
    sky_prominence: Optional[str] = None
    notes: OptionalNotesStr = None


class VisualSchema(BaseModel):
    color_palette: Optional[str] = None
    dominant_hues: Optional[list[str]] = None
    saturation: Optional[str] = None
    brightness: Optional[str] = None
    shot_type: Optional[str] = None
    depth_of_field: Optional[str] = None
    motion_energy: Optional[str] = None
    production_type: Optional[str] = None
    notes: OptionalNotesStr = None


class CreativeSchema(BaseModel):
    lifestyle_segment: Optional[str] = None
    emotional_tone: Optional[str] = None
    campaign_theme: Optional[str] = None
    people_present: Optional[bool] = None
    activity: Optional[str] = None
    compositional_technique: Optional[str] = None
    notes: OptionalNotesStr = None


class FormatSchema(BaseModel):
    ad_format: Optional[str] = None
    aspect_ratio: Optional[str] = None
    platform_target: Optional[str] = None
    orientation: Optional[str] = None
    copy_present: Optional[bool] = None
    logo_badge_visible: Optional[str] = None


class CarClassificationResponse(BaseModel):
    image_id: str
    source_file: str
    classified_at: datetime
    confidence_score: float
    vehicle: VehicleSchema
    location: LocationSchema
    scene: SceneSchema
    visual: VisualSchema
    creative: CreativeSchema
    format: FormatSchema
    raw_labels: list[str]
