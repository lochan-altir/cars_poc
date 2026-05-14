# cars_poc

FastAPI service for car-image classification across multiple providers.

## Layout (scalable)

- Root (project-level):
  - `requirements.txt`
  - `Makefile`
  - `.env` (local only, **gitignored**)
  - `.env.example` (template)
  - `.gitignore`
  - `README.md`
- Application package:
  - `car_classifier/` (all Python app code)
    - `main.py` (FastAPI `app`)
    - `routers/` (provider endpoints)
    - `core/` (shared prompt + utilities)
    - `models/` (Pydantic schemas)

## Setup (single environment for backend + frontend)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Or with Make:

```bash
make install
```

## Run backend

```bash
source .venv/bin/activate
uvicorn car_classifier.main:app --reload
```

or:

```bash
make run-backend
```

## Run frontend (Gradio)

```bash
source .venv/bin/activate
python fronrend/app.py
```

or:

```bash
make run-frontend
```

Frontend URL: `http://127.0.0.1:7860`
Backend URL: `http://127.0.0.1:8000`

## Run both from root

```bash
make run-all
```

## Docker

Build:

```bash
docker build -t cars_poc .
```

Run (loads your local `.env` at runtime):

```bash
docker run --rm -p 8000:8000 --env-file .env cars_poc
```

# cars_poc
