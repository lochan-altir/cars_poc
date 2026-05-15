from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from car_classifier.routers.anthropic import router as anthropic_router
from car_classifier.routers.groq import router as groq_router
from car_classifier.routers.openai import router as openai_router
from car_classifier.routers.ollama import router as ollama_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(anthropic_router, prefix="/classify", tags=["AI providers"])
app.include_router(openai_router, prefix="/classify", tags=["AI providers"])
app.include_router(ollama_router, prefix="/classify", tags=["Open source"])
app.include_router(groq_router, prefix="/classify", tags=["Open source"])


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
