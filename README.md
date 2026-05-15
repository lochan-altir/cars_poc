# cars_poc

Classify car listing images with multiple LLM providers. Two ways to run:

| Entry | Purpose |
|--------|---------|
| **`app.py`** (Streamlit) | Default: single process, UI + direct API calls to Groq / Ollama / OpenAI / Anthropic. No separate backend. |
| **`car_classifier/`** (FastAPI) | Optional REST API (`/classify/*`) for integrations or Docker. |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

Or: `make install`

## Run the UI (recommended)

```bash
source .venv/bin/activate
python3 -m streamlit run app.py
```

Or: `make run-app`

Open the URL printed in the terminal (usually `http://localhost:8501`). Put `.env` in the project root (or parent directory); keys are documented in `.env.example`.

## Run the API only

```bash
source .venv/bin/activate
uvicorn car_classifier.main:app --reload
```

Or: `make run-backend` — then `http://127.0.0.1:8000/docs`

## Project layout

```
cars_poc/
├── app.py                 # Streamlit app (standalone)
├── requirements.txt
├── Makefile
├── Dockerfile             # Builds FastAPI image only
├── car_classifier/        # FastAPI package
│   ├── main.py
│   ├── config.py
│   ├── core/              # prompt, json_utils, image_utils
│   ├── models/            # Pydantic schemas
│   └── routers/           # per-provider HTTP handlers
└── .env.example
```

## Docker (API)

```bash
docker build -t cars-poc-api .
docker run --rm -p 8000:8000 --env-file .env cars-poc-api
```

The image runs **uvicorn** on port **8000**; it does not include the Streamlit app.
# cars_poc
# cars_poc
