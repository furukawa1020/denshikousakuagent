FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=7860

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements-cpu.txt /tmp/requirements-cpu.txt
RUN pip install --no-cache-dir -r /tmp/requirements-cpu.txt

COPY ai_models ./ai_models
COPY backend ./backend
COPY frontend ./frontend
COPY data ./data
COPY runs ./runs

EXPOSE 7860
CMD ["python", "backend/server.py"]
