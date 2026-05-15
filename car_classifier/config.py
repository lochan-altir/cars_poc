import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

ANTHROPIC_MODEL_SONNET = "claude-sonnet-4-20250514"
ANTHROPIC_MODEL_OPUS = "claude-opus-4-20250514"
ANTHROPIC_MODEL_HAIKU = "claude-haiku-4-5-20251001"

OPENAI_MODEL_GPT4O = "gpt-4o"
OPENAI_MODEL_GPT4O_MINI = "gpt-4o-mini"
OPENAI_MODEL_O4_MINI = "o4-mini"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL_LLAVA_7B = "llava:7b"
OLLAMA_MODEL_LLAVA_13B = "llava:13b"
OLLAMA_MODEL_LLAVA_PHI3 = "llava-phi3"

GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL_LLAMA4_SCOUT = os.getenv("GROQ_MODEL_LLAMA4_SCOUT", "meta-llama/llama-4-scout-17b-16e-instruct")
