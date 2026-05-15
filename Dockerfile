FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY car_classifier /app/car_classifier

EXPOSE 8000

CMD ["uvicorn", "car_classifier.main:app", "--host", "0.0.0.0", "--port", "8000"]

