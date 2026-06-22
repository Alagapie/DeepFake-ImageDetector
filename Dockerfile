FROM python:3.11-slim

RUN apt-get update && apt-get install -y libglib2.0-0 libgomp1 libxcb1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=300 \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --default-timeout=300 -r requirements.txt

COPY app/ ./app/

RUN mkdir -p uploads reports

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
