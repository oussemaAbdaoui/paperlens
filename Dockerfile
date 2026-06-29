# PaperLens — single image runs either the API or the Streamlit UI.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for PyMuPDF / faiss / torch wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY eval ./eval
COPY scripts ./scripts
COPY pytest.ini ./

ENV PYTHONPATH=/app/backend:/app
ENV PAPERLENS_DATA_DIR=/data
RUN mkdir -p /data

EXPOSE 8000 8501

# Default: run the API. The frontend service overrides this command.
CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
