PYTHON ?= python3

.PHONY: install run-app run-backend

install:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

# Streamlit UI (calls providers directly; no FastAPI required)
run-app:
	. .venv/bin/activate && $(PYTHON) -m streamlit run app.py

# Optional: OpenAPI REST API (Docker / integrations)
run-backend:
	. .venv/bin/activate && uvicorn car_classifier.main:app --reload
